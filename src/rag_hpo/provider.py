from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from rag_hpo.config import ProviderConfig, ResponseMode
from rag_hpo.models import StrictModel

T = TypeVar("T", bound=BaseModel)
RETRYABLE_STATUSES = {408, 409, 429, *range(500, 600)}


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._jitter = jitter
        timeout = httpx.Timeout(
            connect=config.connect_timeout,
            read=config.read_timeout,
            write=30.0,
            pool=10.0,
        )
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def request(
        self,
        *,
        system_message: str,
        user_message: str,
        response_model: type[T],
        temperature: float = 0.2,
    ) -> tuple[T, str]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
        }
        response_format = self._response_format(response_model)
        if response_format is not None:
            payload["response_format"] = response_format

        response = self._post_with_retries(payload)
        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("completion content is not text")
            return response_model.model_validate_json(content), content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise ProviderError(
                "invalid_response",
                "provider returned an invalid response",
            ) from exc

    def health_check(self) -> None:
        class HealthResponse(StrictModel):
            ok: bool

        self.request(
            system_message='Return JSON containing only {"ok": true}.',
            user_message="Health check.",
            response_model=HealthResponse,
            temperature=0.0,
        )

    def _response_format(self, response_model: type[BaseModel]) -> dict[str, Any] | None:
        if self.config.response_mode is ResponseMode.PROMPT_ONLY:
            return None
        if self.config.response_mode is ResponseMode.JSON_OBJECT:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__.lower(),
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }

    def _post_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.client.post(
                    str(self.config.base_url),
                    headers=headers,
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                if attempt == self.config.max_attempts:
                    raise ProviderError("timeout", "provider request timed out") from exc
                self._sleep(self._delay(attempt, None))
                continue
            except httpx.HTTPError as exc:
                raise ProviderError("network_error", "provider request failed") from exc

            if response.is_success:
                return response
            if response.status_code not in RETRYABLE_STATUSES:
                raise ProviderError(
                    "provider_rejected",
                    f"provider rejected the request with HTTP {response.status_code}",
                    response.status_code,
                )
            if attempt == self.config.max_attempts:
                raise ProviderError(
                    "retry_exhausted",
                    f"provider failed after {attempt} attempts with HTTP {response.status_code}",
                    response.status_code,
                )
            self._sleep(self._delay(attempt, response.headers.get("Retry-After")))
        raise AssertionError("unreachable")

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    seconds = parsedate_to_datetime(retry_after).timestamp() - time.time()
                    return min(60.0, max(0.0, seconds))
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(60.0, float(2 ** (attempt - 1)) + self._jitter())

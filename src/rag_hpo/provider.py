from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError


def _extract_json_payload(raw_text: str) -> str:
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", cleaned, re.DOTALL)
    json_str = match.group(1).strip() if match else cleaned
    if not match:
        match_raw = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match_raw:
            json_str = match_raw.group(1).strip()

    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "phenotypes" in data and isinstance(data["phenotypes"], list):
            new_phenotypes = []
            for item in data["phenotypes"]:
                if isinstance(item, str):
                    new_phenotypes.append({"phrase": item, "category": "Abnormal"})
                elif isinstance(item, dict):
                    if "phrase" in item and "category" not in item:
                        item["category"] = "Abnormal"
                    new_phenotypes.append(item)
            data["phenotypes"] = new_phenotypes
            return json.dumps(data)
    except Exception:
        pass

    return json_str


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
        self.usage: dict[str, int] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

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
        self.usage["requests"] += 1
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
            if response_format.get("type") == "json_object":
                if "json" not in system_message.lower() and "json" not in user_message.lower():
                    payload["messages"][0]["content"] += " Return valid JSON."

        response = self._post_with_retries(payload)
        try:
            envelope = response.json()
            usage = envelope.get("usage", {})
            for source, target in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = usage.get(source)
                if isinstance(value, int):
                    self.usage[target] += value
            content = envelope["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("completion content is not text")
            clean_json = _extract_json_payload(content)
            return response_model.model_validate_json(clean_json), content
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
                    f"provider rejected the request with HTTP {response.status_code}: {response.text}",
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

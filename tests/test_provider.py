from __future__ import annotations

import json

import httpx
import pytest

from rag_hpo.config import ProviderConfig, ResponseMode
from rag_hpo.models import PhenotypeExtraction
from rag_hpo.provider import OpenAICompatibleProvider, ProviderError


def _config(**values: object) -> ProviderConfig:
    return ProviderConfig(
        api_key="secret-for-tests",  # pragma: allowlist secret
        base_url="https://provider.test/v1/chat/completions",
        model="test-model",
        **values,
    )


def _success(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def test_strict_success_does_not_expose_key() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer secret-for-tests"
        return _success('{"phenotypes":[{"phrase":"fever","category":"Abnormal"}]}')

    provider = OpenAICompatibleProvider(
        _config(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    parsed, raw = provider.request(
        system_message="system",
        user_message="note",
        response_model=PhenotypeExtraction,
    )
    assert parsed.phenotypes[0].phrase == "fever"
    assert raw.startswith("{")
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "secret-for-tests" not in json.dumps(body)


def test_json_object_and_prompt_only_modes() -> None:
    formats: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        formats.append(json.loads(request.content).get("response_format"))
        return _success('{"phenotypes":[]}')

    for mode in (ResponseMode.JSON_OBJECT, ResponseMode.PROMPT_ONLY):
        provider = OpenAICompatibleProvider(
            _config(response_mode=mode),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        provider.request(
            system_message="system",
            user_message="note",
            response_model=PhenotypeExtraction,
        )
    assert formats == [{"type": "json_object"}, None]


def test_malformed_json_is_rejected() -> None:
    provider = OpenAICompatibleProvider(
        _config(),
        client=httpx.Client(transport=httpx.MockTransport(lambda _: _success("bad"))),
    )
    with pytest.raises(ProviderError, match="invalid response") as caught:
        provider.request(
            system_message="system",
            user_message="note",
            response_model=PhenotypeExtraction,
        )
    assert caught.value.code == "invalid_response"


@pytest.mark.parametrize("status", [400, 401, 403])
def test_nonretryable_status_fails_immediately(status: int) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    provider = OpenAICompatibleProvider(
        _config(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderError) as caught:
        provider.request(
            system_message="system",
            user_message="note",
            response_model=PhenotypeExtraction,
        )
    assert caught.value.code == "provider_rejected"
    assert calls == 1


def test_rate_limit_honors_retry_after_and_then_succeeds() -> None:
    calls = 0
    delays: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return _success('{"phenotypes":[]}')

    provider = OpenAICompatibleProvider(
        _config(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=delays.append,
    )
    value, _ = provider.request(
        system_message="system",
        user_message="note",
        response_model=PhenotypeExtraction,
    )
    assert value.phenotypes == []
    assert calls == 3
    assert delays == [0.25, 0.25]


def test_retry_exhaustion() -> None:
    provider = OpenAICompatibleProvider(
        _config(),
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))),
        sleep=lambda _: None,
        jitter=lambda: 0.0,
    )
    with pytest.raises(ProviderError) as caught:
        provider.request(
            system_message="system",
            user_message="note",
            response_model=PhenotypeExtraction,
        )
    assert caught.value.code == "retry_exhausted"


def test_timeout_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    provider = OpenAICompatibleProvider(
        _config(max_attempts=2),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderError) as caught:
        provider.request(
            system_message="system",
            user_message="note",
            response_model=PhenotypeExtraction,
        )
    assert caught.value.code == "timeout"
    assert calls == 2

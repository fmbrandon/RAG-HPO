from __future__ import annotations

import os
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class ResponseMode(StrEnum):
    STRICT = "strict"
    JSON_OBJECT = "json-object"
    PROMPT_ONLY = "prompt-only"


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr
    base_url: str = Field(default=DEFAULT_BASE_URL)
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    response_mode: ResponseMode = ResponseMode.STRICT
    connect_timeout: float = Field(default=10.0, gt=0)
    read_timeout: float = Field(default=90.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        response_mode: ResponseMode | str | None = None,
    ) -> ProviderConfig:
        key = os.environ.get("RAG_HPO_API_KEY", "")
        if not key:
            raise ValueError("RAG_HPO_API_KEY is required")
        return cls(
            api_key=SecretStr(key),
            base_url=base_url or os.environ.get("RAG_HPO_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.environ.get("RAG_HPO_MODEL", DEFAULT_MODEL),
            response_mode=ResponseMode(
                response_mode or os.environ.get("RAG_HPO_RESPONSE_MODE", ResponseMode.STRICT)
            ),
        )

    @field_validator("base_url")
    @classmethod
    def require_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("provider base URL must use HTTPS")
        return value

    def redacted(self) -> dict[str, object]:
        return {
            "base_url": str(self.base_url),
            "model": self.model,
            "response_mode": self.response_mode.value,
            "api_key": "configured",  # pragma: allowlist secret
        }

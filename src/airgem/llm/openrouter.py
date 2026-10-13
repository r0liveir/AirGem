from __future__ import annotations

import os

from .provider import LLMProvider, message_payload, post_json


class OpenRouterProvider:
    """OpenAI-compatible OpenRouter adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter")

    def complete(self, *, system: str, user: str) -> str:
        response = post_json(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": message_payload(system, user),
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected OpenRouter response: {response}") from error


def provider() -> LLMProvider:
    return OpenRouterProvider()


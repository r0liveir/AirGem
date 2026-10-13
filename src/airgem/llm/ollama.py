from __future__ import annotations

import os

from .provider import LLMProvider, message_payload, post_json


class OllamaProvider:
    """Ollama chat adapter. Ollama stays local by default."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def complete(self, *, system: str, user: str) -> str:
        response = post_json(
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": message_payload(system, user),
                "stream": False,
                "format": "json",
            },
            timeout=self.timeout,
        )
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"Unexpected Ollama response: {response}") from error


def provider() -> LLMProvider:
    return OllamaProvider()


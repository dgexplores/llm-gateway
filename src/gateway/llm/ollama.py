"""Local Ollama provider. No API key required.

This is the default because it lets the gateway be demonstrated end-to-end with no
credentials at all, which is the difference between a portfolio project someone can
run and one they read about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import Completion, Message
from .http import http_client, post_json

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


@dataclass
class OllamaProvider:
    """Talks to a local Ollama server.

    Non-retryable failures are limited to an unknown model, which will never resolve
    on its own. A stopped server surfaces as a transport error and is retried, because
    the most common cause is the user not having started it yet.
    """

    name: str = "ollama"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self._client = http_client(self.base_url, self.timeout_s)

    @classmethod
    def from_env(cls) -> OllamaProvider:
        return cls(
            model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL),
        )

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        data = post_json(
            self._client,
            "/api/chat",
            {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {"temperature": temperature},
            },
        )

        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            # Ollama reports cumulative counts, which for a single call is what we want.
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    def healthy(self) -> bool:
        try:
            self._client.get("/api/tags", timeout=2.0)
        except Exception:
            return False
        return True

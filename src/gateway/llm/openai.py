"""OpenAI-compatible chat completions provider.

Speaks the OpenAI wire format, which also covers vLLM, Together, Groq, LM Studio and
Ollama's compatibility endpoint. Local servers need no key, so the same class serves
both the hosted API and a self-hosted deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import Completion, Message, ProviderError
from .http import count_tokens_from_payload, http_client, post_json

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class OpenAIProvider:
    """Chat completions against an OpenAI-compatible endpoint.

    An absent key is not an error here: local servers ignore it. The request fails at
    the server instead, which produces a diagnostic with the actual cause rather than
    a client-side guess about which configuration is wrong.
    """

    name: str = "openai"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self._client = http_client(self.base_url, self.timeout_s)

    @classmethod
    def from_env(cls) -> OpenAIProvider:
        key = os.environ.get("OPENAI_API_KEY")
        return cls(
            model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            api_key=key or None,
        )

    @property
    def _is_local(self) -> bool:
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0]
        return host.startswith(("localhost", "127.0.0.1", "0.0.0.0", "::1"))

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        if not self.api_key and not self._is_local:
            raise ProviderError("OPENAI_API_KEY is not set", retryable=False)

        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else None
        data = post_json(
            self._client,
            "/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
            },
            headers=headers,
        )

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("response contained no choices", retryable=True)

        prompt_tokens, completion_tokens = count_tokens_from_payload(data)
        return Completion(
            text=choices[0].get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def healthy(self) -> bool:
        if not self.api_key and not self._is_local:
            return False
        try:
            self._client.get("/models", timeout=5.0)
        except Exception:
            return False
        return True

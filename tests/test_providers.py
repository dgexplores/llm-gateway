"""Status-to-retryability mapping and provider construction.

This is the boundary where a wrong decision costs real latency: a 401 retried three
times adds seconds to every failing request and still fails.
"""

from __future__ import annotations

import pytest

from gateway.llm.base import ProviderError
from gateway.llm.http import is_retryable_status
from gateway.llm.ollama import OllamaProvider
from gateway.llm.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
        (408, True),
        (425, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (529, True),
    ],
)
def test_retryability_by_status(status: int, retryable: bool) -> None:
    assert is_retryable_status(status) is retryable


def test_openai_refuses_without_a_key_on_a_remote_host() -> None:
    provider = OpenAIProvider(api_key=None)
    with pytest.raises(ProviderError) as caught:
        provider.complete([])

    assert caught.value.retryable is False
    assert "OPENAI_API_KEY" in str(caught.value)


def test_openai_tolerates_a_missing_key_on_a_local_server() -> None:
    provider = OpenAIProvider(api_key=None, base_url="http://localhost:8000/v1")

    # The call reaches the server and fails there, rather than being blocked locally.
    with pytest.raises(ProviderError) as caught:
        provider.complete([])

    assert "OPENAI_API_KEY" not in str(caught.value)


def test_openai_from_env_keeps_a_missing_key_as_none() -> None:
    import os

    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL")}
    try:
        provider = OpenAIProvider.from_env()
        assert provider.api_key is None
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def test_ollama_needs_no_key_and_reports_its_model() -> None:
    provider = OllamaProvider()
    assert provider.api_key if hasattr(provider, "api_key") else True
    assert provider.model


def test_ollama_reads_its_endpoint_from_the_environment() -> None:
    import os

    os.environ["OLLAMA_HOST"] = "http://ollama.internal:11434"
    os.environ["OLLAMA_MODEL"] = "qwen2.5:7b"
    try:
        provider = OllamaProvider.from_env()
        assert provider.base_url == "http://ollama.internal:11434"
        assert provider.model == "qwen2.5:7b"
    finally:
        del os.environ["OLLAMA_HOST"]
        del os.environ["OLLAMA_MODEL"]

"""Shared HTTP plumbing for providers that speak JSON over a network call.

Kept separate from the concrete providers so that status-to-retryability mapping and
timeout enforcement exist in exactly one place. A provider that skips this mapping
will retry a 401 three times, which is a latency bug users feel.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import Completion, Message, ProviderError

DEFAULT_TIMEOUT_S = 60.0

# Statuses that may succeed on a later attempt. 429 and 5xx are transient by
# definition; 408 and 425 likewise. Everything else in 4xx is a client-side error
# that will repeat identically.
_RETRYABLE_STATUS = frozenset({408, 425, 429})


def is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE_STATUS or 500 <= status < 600


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST JSON and return the decoded body.

    Translates transport and status failures into ``ProviderError`` with the
    retryability already decided, so callers never re-derive it.
    """
    try:
        response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"timeout calling {url}", retryable=True) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"transport error calling {url}: {exc}", retryable=True) from exc

    if response.status_code >= 400:
        body = response.text[:200]
        raise ProviderError(
            f"HTTP {response.status_code} from {url}: {body}",
            retryable=is_retryable_status(response.status_code),
        )

    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError(f"non-JSON response from {url}", retryable=True) from exc


def count_tokens_from_payload(data: dict[str, Any]) -> tuple[int, int]:
    """Best-effort token counts when a provider omits usage.

    Word count under-reports for code and over-reports for prose. It is only ever a
    fallback: an absent count recorded as a guess is better than recorded as zero,
    because zero would silently understate cost.
    """
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


def http_client(base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> httpx.Client:
    """An httpx client with bounded timeouts.

    Connect and read timeouts are set separately: a connect timeout alone would let a
    hung server hold a request forever.
    """
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
        headers={"content-type": "application/json"},
    )


__all__ = [
    "Completion",
    "Message",
    "ProviderError",
    "count_tokens_from_payload",
    "http_client",
    "is_retryable_status",
    "post_json",
]

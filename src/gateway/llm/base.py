"""Provider protocol and response types.

One protocol, three implementations. Adding a provider means implementing
``complete``; no call site changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)


@dataclass(frozen=True)
class Completion:
    """A single provider response."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Provider call failed. ``retryable`` drives the retry decision."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class ProviderExhausted(ProviderError):
    """Every configured provider failed. Raised instead of returning a fabricated answer."""

    def __init__(self, errors: dict[str, str]):
        detail = "; ".join(f"{name}: {msg}" for name, msg in errors.items())
        super().__init__(f"all providers failed ({detail})", retryable=False)
        self.errors = errors


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn messages into a completion.

    Contract, checked by tests rather than asserted at runtime:

    - Must raise ``ProviderError`` on failure. Anything else escaping ``complete`` is
      a bug in the provider, not a condition the client can retry against.
    - ``retryable=False`` means "do not retry": a 401 or a malformed request will not
      succeed on a second identical attempt, and retrying only adds latency.
    - ``model`` must be stable for the lifetime of the instance, because it keys the
      cost table and the cache namespace.
    """

    name: str

    @property
    def model(self) -> str: ...

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion: ...

    def healthy(self) -> bool:
        """Whether this provider should be attempted. Circuit breakers override this."""
        return True


def require_api_key(*names: str) -> str:
    """Read the first present API key from the environment.

    Raises rather than returning an empty string: a missing key should fail at the
    boundary, not surface as a 401 from a remote call three retries later.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ProviderError(
        f"no API key found (looked for {', '.join(names)})",
        retryable=False,
    )

"""Cache.

Hash mode is the default because it costs nothing. Semantic mode costs an embedding
call per query, so it is opt-in and its hit rate is measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from .llm.base import Completion, Message


class Cache(Protocol):
    def get(self, key: str) -> Completion | None: ...

    def put(self, key: str, value: Completion) -> None: ...

    def stats(self) -> dict[str, int]: ...


class InMemoryCache:
    """Process-local cache. Correct behaviour for single-node deployments and tests."""

    def __init__(self) -> None:
        self._store: dict[str, Completion] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def key_for(messages: list[Message], model: str) -> str:
        """Deterministic hash of the conversation and model.

        Model is included so a cache built for one model is never served to another.
        """
        payload = json.dumps(
            [[m.role, m.content] for m in messages] + [["__model__", model]],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Completion | None:
        found = self._store.get(key)
        if found is None:
            self._misses += 1
        else:
            self._hits += 1
        return found

    def put(self, key: str, value: Completion) -> None:
        self._store[key] = value

    def stats(self) -> dict[str, int]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


class NullCache:
    """Always misses. The correct behaviour when caching is off."""

    def get(self, key: str) -> Completion | None:
        return None

    def put(self, key: str, value: Completion) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": 0, "hit_rate": 0.0}

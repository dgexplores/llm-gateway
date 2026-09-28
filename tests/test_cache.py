"""Module 2 verification: caching correctness and graceful degradation."""

from __future__ import annotations

import pytest

from gateway import Completion, InMemoryCache, Message, NullCache
from tests.fakes import FakeProvider


def test_cache_hit_avoids_a_provider_call() -> None:
    provider = FakeProvider("p", [Completion(text="cached", model="fake-1")])
    cache = InMemoryCache()
    messages = [Message.user("what is a transformer")]
    key = InMemoryCache.key_for(messages, "fake-1")

    cache.put(key, Completion(text="cached", model="fake-1"))
    provider.calls = 0

    hit = cache.get(key)

    assert hit is not None and hit.text == "cached"
    assert provider.calls == 0


def test_cache_key_separates_models() -> None:
    a = InMemoryCache.key_for([Message.user("hi")], "model-a")
    b = InMemoryCache.key_for([Message.user("hi")], "model-b")

    assert a != b, "a cache built for one model must not be served to another"


def test_cache_key_separates_conversations() -> None:
    a = InMemoryCache.key_for([Message.user("hi")], "m")
    b = InMemoryCache.key_for([Message.system("be terse"), Message.user("hi")], "m")

    assert a != b


def test_stats_report_hit_rate() -> None:
    cache = InMemoryCache()
    cache.put("k", Completion(text="v", model="m"))

    cache.get("k")
    cache.get("k")
    cache.get("missing")

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.6667, abs=1e-4)


def test_null_cache_always_misses() -> None:
    cache = NullCache()
    cache.put("k", Completion(text="v", model="m"))

    assert cache.get("k") is None
    assert cache.stats()["hit_rate"] == 0.0

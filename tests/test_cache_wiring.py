"""The cache has to be wired into the client, not merely implemented.

`InMemoryCache` and `NullCache` were written, exported, and unit-tested while nothing in
the pipeline ever constructed one. Every test passed, the project's stated cache hit rate
was a claim about dead code, and nothing reported a problem — because a unit test of a
class cannot tell you whether anything calls it.

These tests go through `LLMClient.complete`, which is the only place the cache is
reachable from. A regression that removes the wiring fails here.
"""

from __future__ import annotations

import pytest

from gateway.accounting import UsageRecorder
from gateway.cache import InMemoryCache, NullCache
from gateway.llm.base import Completion, Message
from gateway.llm.client import LLMClient


class CountingProvider:
    name = "counting"

    def __init__(self, model: str = "stub-1") -> None:
        self.calls = 0
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages, *, request_id="", temperature=0.0) -> Completion:
        self.calls += 1
        return Completion(text="ok", model=self._model, prompt_tokens=10, completion_tokens=2)


def _client(cache=None, model: str = "stub-1") -> tuple[LLMClient, CountingProvider]:
    provider = CountingProvider(model)
    recorder = UsageRecorder(stream=open("/dev/null", "w"))
    return LLMClient([provider], recorder, cache=cache), provider


def _messages(text: str = "hello") -> list[Message]:
    return [Message.system("s"), Message.user(text)]


def test_a_repeated_call_is_served_from_the_cache() -> None:
    cache = InMemoryCache()
    client, provider = _client(cache)

    client.complete(_messages(), request_id="r1")
    client.complete(_messages(), request_id="r2")

    assert provider.calls == 1, "the second identical call reached the provider"
    assert cache.stats()["hits"] == 1


def test_different_prompts_are_not_confused() -> None:
    cache = InMemoryCache()
    client, provider = _client(cache)

    client.complete(_messages("first"), request_id="r1")
    client.complete(_messages("second"), request_id="r2")

    assert provider.calls == 2


def test_caching_is_off_by_default() -> None:
    """Opt-in, so a caller who did not ask for a cache cannot get one by accident."""
    client, provider = _client()

    client.complete(_messages(), request_id="r1")
    client.complete(_messages(), request_id="r2")

    assert provider.calls == 2
    assert isinstance(client.cache, NullCache)


def test_a_null_cache_never_returns_anything() -> None:
    client, provider = _client(NullCache())

    client.complete(_messages(), request_id="r1")
    client.complete(_messages(), request_id="r2")

    assert provider.calls == 2


def test_two_models_do_not_share_entries() -> None:
    """A cache built for one model must never be served to another, or the model field
    in the results table lies about which model produced the text."""
    shared = InMemoryCache()
    first, first_provider = _client(shared, model="model-a")
    second, second_provider = _client(shared, model="model-b")

    first.complete(_messages(), request_id="r1")
    second.complete(_messages(), request_id="r2")

    assert first_provider.calls == 1
    assert second_provider.calls == 1, "model-b was served model-a's cached completion"


def test_temperature_is_not_part_of_the_key() -> None:
    """Documented consequence, pinned so it cannot change silently: two calls differing
    only in temperature share an entry, so the second gets the first's text. Worth
    knowing rather than discovering."""
    cache = InMemoryCache()
    client, provider = _client(cache)

    client.complete(_messages(), request_id="r1", temperature=0.0)
    client.complete(_messages(), request_id="r2", temperature=0.9)

    assert provider.calls == 1


def test_a_failure_is_not_cached() -> None:
    """Caching a failure would make one bad provider call permanent for the lifetime of
    the process, which is far worse than an extra call."""
    from gateway.llm.base import ProviderError, ProviderExhausted

    class Failing(CountingProvider):
        name = "failing"

        def complete(self, messages, *, request_id="", temperature=0.0) -> Completion:
            self.calls += 1
            raise ProviderError("down", retryable=False)

    cache = InMemoryCache()
    provider = Failing()
    client = LLMClient([provider], UsageRecorder(stream=open("/dev/null", "w")), cache=cache)

    for i in range(2):
        with pytest.raises(ProviderExhausted):
            client.complete(_messages(), request_id=f"r{i}")

    assert cache.stats()["hits"] == 0
    assert len(cache._store) == 0


def test_a_cache_hit_is_recorded_as_one() -> None:
    """Accounting is the reason the cache exists. A hit that costs tokens and money in
    the usage record makes the hit rate a lie in the other direction."""
    seen: list[bool] = []

    class Recorder(UsageRecorder):
        def record(self, usage):  # type: ignore[override]
            seen.append(usage.cache_hit)
            super().record(usage)

    provider = CountingProvider()
    client = LLMClient(
        [provider], Recorder(stream=open("/dev/null", "w")), cache=InMemoryCache()
    )

    client.complete(_messages(), request_id="r1")
    client.complete(_messages(), request_id="r2")

    assert seen == [False, True]


def test_the_cache_hit_flag_is_a_label_not_a_short_circuit() -> None:
    """``cache_hit=True`` means the caller already holds a cached completion and wants the
    usage record to say so. It must not consult this cache — that would record the request
    twice — but it still has to return a Completion, so the provider is called.

    Pinned because "the flag means the value came from a cache" reads like it should skip
    the provider, and writing that test is a trap worth naming."""
    cache = InMemoryCache()
    client, provider = _client(cache)

    client.complete(_messages(), request_id="r1", cache_hit=True)

    assert cache.stats()["hits"] == 0, "the flag must not trigger a second cache lookup"
    assert provider.calls == 1, "the flag labels the accounting; it does not supply text"


def test_a_labelled_cache_hit_is_not_stored_again() -> None:
    """A completion the caller already had cached does not need writing back."""
    cache = InMemoryCache()
    client, _provider = _client(cache)

    client.complete(_messages(), request_id="r1", cache_hit=True)

    assert len(cache._store) == 0

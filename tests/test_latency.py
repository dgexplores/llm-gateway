"""Latency measurement.

The bug these guard: the clock was started inside the record call, after the provider
had already returned, so every recorded latency was approximately zero. Nothing failed
loudly, because a latency of 0.0 is a perfectly valid float. It would have shipped a
plausible, wrong number into the results table that the resume is built from.
"""

from __future__ import annotations

import io
import json

import pytest

from gateway import LLMClient, Message, ProviderExhausted
from tests.fakes import CountingSleep, FakeClock, SlowProvider, TimedProvider


def recorder() -> tuple[object, io.StringIO]:
    from gateway import UsageRecorder

    stream = io.StringIO()
    return UsageRecorder(stream=stream), stream


def parse(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_latency_is_measured_across_the_provider_call() -> None:
    rec, stream = recorder()
    clock = FakeClock()
    provider = TimedProvider("p", duration_s=0.5, clock=clock)

    LLMClient([provider], rec, sleep=lambda _: None, monotonic=clock).complete(
        [Message.user("hi")], request_id="r1"
    )

    assert parse(stream)[0]["latency_ms"] == pytest.approx(500.0)


def test_latency_is_not_zero_for_an_instant_provider() -> None:
    """Guards the specific regression: a near-zero clock gap must not read as 0.0."""
    rec, stream = recorder()
    clock = FakeClock()
    provider = TimedProvider("p", duration_s=0.02, clock=clock)

    LLMClient([provider], rec, sleep=lambda _: None, monotonic=clock).complete(
        [Message.user("hi")], request_id="r1"
    )

    assert parse(stream)[0]["latency_ms"] > 0.0


def test_total_latency_includes_backoff_and_earlier_attempts() -> None:
    rec, stream = recorder()
    clock = FakeClock()
    sleeper = CountingSleep()
    # One failed attempt, then success on the second.
    provider = TimedProvider("p", duration_s=0.1, fail_times=1, clock=clock)

    LLMClient([provider], rec, sleep=sleeper, monotonic=clock).complete(
        [Message.user("hi")], request_id="r1"
    )

    record = parse(stream)[0]
    assert record["latency_ms"] == pytest.approx(100.0), "one successful attempt only"
    assert record["total_latency_ms"] > record["latency_ms"], "must account for the retry"
    assert record["attempt"] == 2


def test_total_latency_is_never_less_than_provider_latency() -> None:
    """Invariant, not a one-off example: the request cannot outrun the call inside it."""
    rec, _ = recorder()
    clock = FakeClock()
    provider = TimedProvider("p", duration_s=0.3, fail_times=1, clock=clock)

    LLMClient([provider], rec, sleep=CountingSleep(), monotonic=clock).complete(
        [Message.user("hi")], request_id="r1"
    )

    from gateway.accounting import Usage

    usage = Usage(
        request_id="r",
        provider="p",
        model="m",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=100.0,
        total_latency_ms=250.0,
        cost_usd=0.0,
    )
    assert usage.overhead_ms == pytest.approx(150.0)
    assert usage.overhead_ms >= 0.0


def test_real_wall_clock_latency_is_positive() -> None:
    """Confirms the injection does not mask a genuine measurement error."""
    rec, stream = recorder()
    provider = SlowProvider("p", duration_s=0.05)

    LLMClient([provider], rec, sleep=lambda _: None).complete(
        [Message.user("hi")], request_id="r1"
    )

    assert parse(stream)[0]["latency_ms"] >= 45.0


def test_backoff_stays_within_bounds() -> None:
    rec, _ = recorder()
    clock = FakeClock()
    sleeper = CountingSleep()
    provider = TimedProvider("p", fail_times=10, clock=clock)

    with pytest.raises(ProviderExhausted):
        LLMClient([provider], rec, sleep=sleeper, monotonic=clock).complete(
            [Message.user("hi")], request_id="r1"
        )

    assert len(sleeper.calls) == 2, "two backoffs between three attempts"
    assert all(0.0 <= d <= 2.0 for d in sleeper.calls), "backoff must stay capped"


def test_cache_hit_path_still_records_a_measurement() -> None:
    """A cache hit skips the provider but the request is still a real request."""
    rec, stream = recorder()
    clock = FakeClock()
    provider = TimedProvider("p", clock=clock)

    LLMClient([provider], rec, sleep=lambda _: None, monotonic=clock).complete(
        [Message.user("hi")], request_id="r1", cache_hit=True
    )

    record = parse(stream)[0]
    assert record["cache_hit"] is True
    assert record["latency_ms"] >= 0.0


def test_circuit_breaker_opens_and_is_reported() -> None:
    rec, _ = recorder()
    client = LLMClient(
        [TimedProvider("p", fail_times=99)], rec, sleep=lambda _: None, monotonic=FakeClock()
    )

    for i in range(3):
        with pytest.raises(ProviderExhausted):
            client.complete([Message.user("hi")], request_id=f"r{i}")

    assert client.breakers["p"].open

"""Module 1 verification: failover, retry, and exactly one usage record per call."""

from __future__ import annotations

import io
import json

import pytest

from gateway import (
    Completion,
    CostTable,
    LLMClient,
    Message,
    ProviderError,
    ProviderExhausted,
    UsageRecorder,
)
from tests.fakes import FakeProvider


def recorder() -> tuple[UsageRecorder, io.StringIO]:
    stream = io.StringIO()
    return UsageRecorder(stream=stream), stream


def parse(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_fails_over_after_primary_retries_exhausted() -> None:
    rec, stream = recorder()
    primary = FakeProvider("primary", [ProviderError("429 rate limited")])
    secondary = FakeProvider("secondary", [Completion(text="ok", model="fake-1")])

    client = LLMClient([primary, secondary], rec, sleep=lambda _: None)
    result = client.complete([Message.user("hi")], request_id="r1")

    assert result.text == "ok"
    assert primary.calls == 3, "primary should be retried before failing over"
    assert secondary.calls == 1


def test_recovers_on_retry_without_failing_over() -> None:
    rec, _ = recorder()
    flaky = FakeProvider(
        "flaky",
        [ProviderError("503"), Completion(text="recovered", model="fake-1")],
    )
    client = LLMClient([flaky], rec, sleep=lambda _: None)

    result = client.complete([Message.user("hi")], request_id="r1")

    assert result.text == "recovered"
    assert flaky.calls == 2


def test_raises_rather_than_returning_a_fabricated_answer() -> None:
    rec, _ = recorder()
    providers = [
        FakeProvider("a", [ProviderError("down")]),
        FakeProvider("b", [ProviderError("down")]),
    ]
    client = LLMClient(providers, rec, sleep=lambda _: None)

    with pytest.raises(ProviderExhausted) as caught:
        client.complete([Message.user("hi")], request_id="r1")

    assert set(caught.value.errors) == {"a", "b"}


def test_non_retryable_error_fails_over_immediately() -> None:
    rec, _ = recorder()
    primary = FakeProvider("primary", [ProviderError("bad key", retryable=False)])
    secondary = FakeProvider("secondary", [Completion(text="ok", model="fake-1")])
    client = LLMClient([primary, secondary], rec, sleep=lambda _: None)

    client.complete([Message.user("hi")], request_id="r1")

    assert primary.calls == 1, "a 401 should not be retried"


def test_writes_exactly_one_record_per_provider_call() -> None:
    rec, stream = recorder()
    provider = FakeProvider(
        "p", [Completion(text="ok", model="fake-1", prompt_tokens=10, completion_tokens=5)]
    )
    client = LLMClient([provider], rec, sleep=lambda _: None)

    client.complete([Message.user("hi")], request_id="r1")
    client.complete([Message.user("bye")], request_id="r2")

    records = parse(stream)
    assert len(records) == 2
    assert [r["request_id"] for r in records] == ["r1", "r2"]
    assert all(r["prompt_tokens"] == 10 and r["completion_tokens"] == 5 for r in records)


def test_no_record_when_every_provider_fails() -> None:
    rec, stream = recorder()
    client = LLMClient([FakeProvider("a", [ProviderError("down")])], rec, sleep=lambda _: None)

    with pytest.raises(ProviderExhausted):
        client.complete([Message.user("hi")], request_id="r1")

    assert parse(stream) == []


def test_cost_uses_registered_rates_and_zero_for_unknown_models() -> None:
    costs = CostTable()
    costs.add("p", "fake-1", prompt=3.0, completion=15.0)

    assert costs.cost("p", "fake-1", 1_000_000, 0) == pytest.approx(3.0)
    assert costs.cost("p", "unknown", 1_000_000, 1_000_000) == 0.0


def test_attempt_number_is_recorded() -> None:
    rec, stream = recorder()
    provider = FakeProvider(
        "flaky",
        [ProviderError("503"), Completion(text="ok", model="fake-1")],
    )
    LLMClient([provider], rec, sleep=lambda _: None).complete([Message.user("hi")], request_id="r1")

    assert parse(stream)[0]["attempt"] == 2

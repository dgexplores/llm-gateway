"""A clock and providers whose timing the test controls.

Timing is injected rather than real so latency assertions are exact. A test that waits
on wall time is slow when it passes and flaky when it fails, which is unacceptable in
the one place that is supposed to guard the metrics.
"""

from __future__ import annotations

from gateway.llm.base import Completion, Message, ProviderError


class FakeClock:
    """Monotonic clock that advances only when told to."""

    def __init__(self, step_s: float = 0.0):
        self._t = 0.0
        self._step = step_s

    def __call__(self) -> float:
        self._t += self._step
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class TimedProvider:
    """A provider whose ``complete`` advances an injected clock.

    The advance represents the provider round trip, so the recorded ``latency_ms`` is
    the elapsed time across the call. Using the same clock the client reads keeps the
    measurement internally consistent.
    """

    def __init__(
        self,
        name: str,
        *,
        duration_s: float = 0.25,
        result: Completion | None = None,
        fail_times: int = 0,
        clock: FakeClock | None = None,
        model: str = "fake-1",
    ):
        self.name = name
        self.model = model
        self.duration_s = duration_s
        self.result = result or Completion(text="ok", model=model)
        self.fail_times = fail_times
        self.clock = clock or FakeClock()
        self.calls = 0

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        self.calls += 1
        self.clock.advance(self.duration_s)
        if self.calls <= self.fail_times:
            raise ProviderError("simulated transient failure")
        return self.result

    def healthy(self) -> bool:
        return True


class SlowProvider(TimedProvider):
    """Real wall-clock delay, for tests asserting on genuinely elapsed time."""

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        import time

        time.sleep(self.duration_s)
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("simulated transient failure")
        return self.result


class FakeProvider:
    """Returns a scripted sequence of results.

    Each item is either a ``Completion`` to return, or an exception to raise. Once the
    script is exhausted the last item repeats, which is how "always fails" is
    expressed.
    """

    def __init__(self, name: str, script: list[Completion | Exception], model: str = "fake-1"):
        self.name = name
        self.model = model
        self._script = script
        self.calls = 0

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        index = min(self.calls, len(self._script) - 1)
        self.calls += 1
        result = self._script[index]
        if isinstance(result, Exception):
            raise result
        return result

    def healthy(self) -> bool:
        return True


class CountingSleep:
    """Records backoff durations instead of waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)

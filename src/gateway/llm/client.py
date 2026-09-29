"""The single client every model call goes through.

Provider selection, retries, and accounting are the default rather than something
each call site opts into.
"""

from __future__ import annotations

import random
import time

from ..accounting import CostTable, Usage, UsageRecorder
from .base import Completion, Message, Provider, ProviderError, ProviderExhausted

# Retried up to this many times per provider before failing over.
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 0.2
BACKOFF_CAP_S = 2.0


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures and stays open for ``cooldown_s``.

    Prevents a dead provider from adding its retry latency to every request. Half-open
    after the cooldown: one request is allowed through to test recovery.
    """

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    @property
    def open(self) -> bool:
        return self._opened_at is not None


class LLMClient:
    """Retries, then fails over, then refuses to answer.

    Order of operations per request: for each provider in order, attempt up to
    ``MAX_ATTEMPTS`` times with jittered backoff. If all providers fail, raise
    ``ProviderExhausted`` rather than returning the last error's text.
    """

    def __init__(
        self,
        providers: list[Provider],
        recorder: UsageRecorder,
        costs: CostTable | None = None,
        *,
        sleep=time.sleep,
        monotonic=time.perf_counter,
        rng: random.Random | None = None,
    ):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.recorder = recorder
        self.costs = costs or CostTable()
        self.breakers = {p.name: CircuitBreaker() for p in providers}
        # Injected so latency assertions are deterministic in tests. A real sleep in
        # the test path would make the suite slow and flaky in exactly the place
        # where a flaky test is least welcome.
        self._sleep = sleep
        self._monotonic = monotonic
        self._rng = rng or random.Random()

    def complete(
        self,
        messages: list[Message],
        *,
        request_id: str,
        temperature: float = 0.0,
        cache_hit: bool = False,
    ) -> Completion:
        errors: dict[str, str] = {}
        request_started = self._monotonic()

        for provider in self.providers:
            breaker = self.breakers[provider.name]

            if not breaker.allow():
                errors[provider.name] = "circuit open"
                continue

            for attempt in range(1, MAX_ATTEMPTS + 1):
                # The clock must wrap the call itself. Starting it after the call
                # returns yields a latency of approximately zero, which is worse than
                # no metric at all: it looks like a real number in a results table.
                attempt_started = self._monotonic()
                try:
                    completion = provider.complete(messages, temperature=temperature)
                except ProviderError as exc:
                    if not exc.retryable:
                        errors[provider.name] = str(exc)
                        break
                    if attempt == MAX_ATTEMPTS:
                        errors[provider.name] = str(exc)
                        break
                    self._sleep(self._backoff(attempt))
                    continue

                breaker.record_success()
                self._record(
                    request_id,
                    provider,
                    completion,
                    cache_hit,
                    attempt,
                    latency_ms=(self._monotonic() - attempt_started) * 1000,
                    total_latency_ms=(self._monotonic() - request_started) * 1000,
                )
                return completion

            breaker.record_failure()

        raise ProviderExhausted(errors)

    @property
    def name(self) -> str:
        """The provider chain, in the order it will be attempted.

        A client fronts several providers so it has no single identity; the chain is
        what a user needs to see when an answer came from the wrong place.
        """
        return "→".join(p.name for p in self.providers) or "none"

    def _backoff(self, attempt: int) -> float:
        ceiling = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
        return self._rng.uniform(0, ceiling)

    def _record(
        self,
        request_id: str,
        provider: Provider,
        completion: Completion,
        cache_hit: bool,
        attempt: int,
        *,
        latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        self.recorder.record(
            Usage(
                request_id=request_id,
                provider=provider.name,
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                latency_ms=latency_ms,
                total_latency_ms=total_latency_ms,
                cost_usd=self.costs.cost(
                    provider.name,
                    completion.model,
                    completion.prompt_tokens,
                    completion.completion_tokens,
                ),
                cache_hit=cache_hit,
                attempt=attempt,
            )
        )

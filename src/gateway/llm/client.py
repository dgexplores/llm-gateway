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
        rng: random.Random | None = None,
    ):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.recorder = recorder
        self.costs = costs or CostTable()
        self.breakers = {p.name: CircuitBreaker() for p in providers}
        self._sleep = sleep
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

        for provider in self.providers:
            breaker = self.breakers[provider.name]

            if not breaker.allow():
                errors[provider.name] = "circuit open"
                continue

            for attempt in range(1, MAX_ATTEMPTS + 1):
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
                self._record(request_id, provider, completion, cache_hit, attempt)
                return completion

            breaker.record_failure()

        raise ProviderExhausted(errors)

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
    ) -> None:
        started = time.perf_counter()
        self.recorder.record(
            Usage(
                request_id=request_id,
                provider=provider.name,
                model=completion.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                latency_ms=(time.perf_counter() - started) * 1000,
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

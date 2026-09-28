"""Usage accounting.

Every provider call produces exactly one record. Records are the only source of the
numbers in docs/RESULTS.md, so that file is generated from them rather than written
by hand.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class Usage:
    """One provider call."""

    request_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool = False
    verified: bool = False
    abstained: bool = False
    attempt: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class CostTable:
    """USD per million tokens, keyed by ``(provider, model)``.

    Unknown models cost 0.0 rather than raising: a missing price should not break a
    request, but it should also show up as zero in the results so the gap is visible.
    """

    _USD_PER_TOKEN = 1e-6

    def __init__(self, table: dict[tuple[str, str], tuple[float, float]] | None = None):
        self._table = table or {}

    def add(self, provider: str, model: str, prompt: float, completion: float) -> None:
        """Register USD per million tokens for a model."""
        self._table[(provider, model)] = (prompt, completion)

    def cost(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        try:
            prompt_rate, completion_rate = self._table[(provider, model)]
        except KeyError:
            return 0.0
        return (
            prompt_tokens * prompt_rate + completion_tokens * completion_rate
        ) * self._USD_PER_TOKEN


class UsageRecorder:
    """Append-only JSONL recorder.

    One writer, append mode, flushed per record so a crash loses at most the record
    being written.
    """

    def __init__(self, stream: TextIO | None = None, path: Path | None = None):
        if (stream is None) == (path is None):
            raise ValueError("supply exactly one of stream or path")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a", encoding="utf-8")
        self._stream = stream

    def record(self, usage: Usage) -> None:
        self._stream.write(usage.to_json() + "\n")
        self._stream.flush()

    @contextmanager
    def timed(self, **fields: object) -> Iterator[dict[str, object]]:
        """Measure a provider call and fill in latency.

        Usage:

            with recorder.timed(request_id=..., provider=..., model=...) as slot:
                response = client.complete(...)
                slot["prompt_tokens"] = response.prompt_tokens
                slot["completion_tokens"] = response.completion_tokens
        """
        slot: dict[str, object] = dict(fields)
        started = time.perf_counter()
        try:
            yield slot
        finally:
            slot.setdefault("latency_ms", (time.perf_counter() - started) * 1000)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> UsageRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

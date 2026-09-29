"""What the response cache is actually worth.

A cache is claimed to cut cost and latency, and both are easy to assert and awkward to
measure. This measures them on a workload with a stated shape, because the hit rate is a
property of the workload, not of the cache: 100% on a repeated question and 0% on unique
questions are both correct numbers for different traffic.

The workload is a Zipf-ish mix — one question asked many times, a long tail asked once —
because that is what documentation traffic looks like, and because a uniform workload
would report a hit rate that describes no real system.

Usage:
    uv run python scripts/cache_benchmark.py --json docs/results/cache.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gateway.accounting import UsageRecorder
from gateway.cache import InMemoryCache
from gateway.llm.base import Completion, Message
from gateway.llm.client import LLMClient


class CountingProvider:
    """A provider that does the least possible work.

    Deliberately not DemoProvider: this script measures the cache, and a provider that
    parses a source block would be measuring something else. The stub also makes the
    latency floor obvious, because there is almost nothing to save.
    """

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model(self) -> str:
        return "stub-1"

    def complete(self, messages, *, request_id="", temperature=0.0) -> Completion:
        self.calls += 1
        return Completion(
            text=messages[-1].content[:40],
            model="stub-1",
            prompt_tokens=len(messages[-1].content),
            completion_tokens=8,
        )

# Weight 0 is asked 40 times, weight 1 five times, and so on. Real traffic is skewed;
# a uniform workload flatters the cache.
def workload(questions: list[str]) -> list[str]:
    out: list[str] = []
    for index, question in enumerate(questions):
        out.extend([question] * max(1, 40 // (index + 1) if index < 4 else 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    questions = [json.loads(line)["question"] for line in
                 Path("evals/goldens.jsonl").read_text().splitlines() if line.strip()]
    calls = workload(questions)

    messages = [[Message.system("s"), Message.user(q)] for q in calls]
    recorder = UsageRecorder(stream=open("/dev/null", "w"))
    cache = InMemoryCache()
    provider = CountingProvider()
    client = LLMClient([provider], recorder, cache=cache)

    started = time.perf_counter()
    for batch in messages:
        client.complete(batch, request_id="bench")
    cached_elapsed = (time.perf_counter() - started) * 1000

    # Same workload with no cache, for the latency it replaces.
    uncached_provider = CountingProvider()
    uncached_client = LLMClient([uncached_provider], recorder)
    started = time.perf_counter()
    for batch in messages:
        uncached_client.complete(batch, request_id="bench")
    uncached_elapsed = (time.perf_counter() - started) * 1000

    total = len(calls)
    hits = cache.stats()["hits"]
    rate = hits / total if total else 0.0

    print(f"workload          {total} calls over {len(questions)} distinct questions")
    print(f"cache hits        {hits} ({rate:.1%})")
    print(f"with cache        {cached_elapsed:8.1f} ms total")
    print(f"without cache     {uncached_elapsed:8.1f} ms total")
    saved = 1 - (cached_elapsed / uncached_elapsed) if uncached_elapsed else 0.0
    print(f"latency avoided   {saved:8.1%}")
    print(
        "\nThe provider is a stub that does almost nothing, so absolute latency is not\n"
        "representative of a network model. The hit rate is a property of the workload and\n"
        "is real; the latency saving is a floor, not an estimate of a production gain."
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "calls": total,
                    "distinct_questions": len(questions),
                    "cache_hits": hits,
                    "hit_rate": round(rate, 4),
                    "with_cache_ms": round(cached_elapsed, 1),
                    "without_cache_ms": round(uncached_elapsed, 1),
                    "latency_avoided": round(saved, 4),
                    "provider_calls_made": provider.calls,
                    "provider_calls_avoided": total - provider.calls,
                    "note": "stub provider; the hit rate is real, the latency saving is a floor",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

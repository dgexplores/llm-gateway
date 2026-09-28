"""Evaluation.

Metrics are split by whether they are deterministic, because that determines what may
gate a build. A non-deterministic metric used as a gate produces flaky failures that
teams learn to retry past, which is worse than having no gate at all: the gate still
costs time while catching nothing.

    retrieval metrics  deterministic        gate CI
    answer metrics     LLM-judged           report only
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .retriever import ScoredChunk


@dataclass(frozen=True)
class Golden:
    """One labelled question.

    ``relevant_doc_ids`` may name whole documents rather than individual chunks, which
    keeps labelling cheap while still supporting document-level recall.
    """

    id: str
    question: str
    answer: str
    relevant_doc_ids: tuple[str, ...]
    difficulty: str = "single-hop"
    notes: str = ""


def load_goldens(path: Path) -> list[Golden]:
    goldens: list[Golden] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        record = json.loads(line)
        goldens.append(
            Golden(
                id=record["id"],
                question=record["question"],
                answer=record.get("answer", ""),
                relevant_doc_ids=tuple(record.get("relevant_doc_ids", ())),
                difficulty=record.get("difficulty", "single-hop"),
                notes=record.get("notes", ""),
            )
        )
    return goldens


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(results: Sequence[ScoredChunk], relevant: set[str], k: int = 10) -> float:
    """Normalised discounted cumulative gain over document ids.

    Returns 0.0 when nothing is relevant, which is the correct reading: an unanswerable
    question has no credit to award.

    Bounded to 1.0. Repeated chunks from one relevant document each earn the full
    position discount, so a retriever that returns ten chunks of a single document
    would otherwise score above the ideal ranking of distinct documents. The bound
    keeps the metric interpretable as "how good is this ordering" rather than
    "how aggressively did this retriever duplicate results".
    """
    if not relevant:
        return 0.0
    gains = [1.0 if s.doc_id in relevant else 0.0 for s in results[:k]]
    ideal = [1.0] * min(len(relevant), k)
    denominator = _dcg(ideal)
    if not denominator:
        return 0.0
    return min(1.0, _dcg(gains) / denominator)


def recall_at_k(results: Sequence[ScoredChunk], relevant: set[str], k: int = 10) -> float:
    """Fraction of relevant documents that appear in the top ``k``.

    Averages recall over *questions*, not documents: one question with five relevant
    docs should not outweigh three questions with one each.
    """
    if not relevant:
        return 1.0
    found = {s.doc_id for s in results[:k]}
    return len(found & relevant) / len(relevant)


def mrr(results: Sequence[ScoredChunk], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result. 0.0 if none is in the list."""
    for position, scored in enumerate(results, start=1):
        if scored.doc_id in relevant:
            return 1.0 / position
    return 0.0


def hit_rate_at_k(results: Sequence[ScoredChunk], relevant: set[str], k: int = 10) -> float:
    return 1.0 if any(s.doc_id in relevant for s in results[:k]) else 0.0


@dataclass
class Metrics:
    """Aggregated retrieval quality for one retriever over one golden set."""

    retriever: str
    n_questions: int
    ndcg_at_10: float
    recall_at_10: float
    mrr: float
    hit_rate_at_10: float
    p95_retrieval_ms: float = 0.0
    mean_candidates: float = 0.0
    per_difficulty: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        return {
            "retriever": self.retriever,
            "ndcg@10": round(self.ndcg_at_10, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "hit@10": round(self.hit_rate_at_10, 4),
            "p95_ms": round(self.p95_retrieval_ms, 1),
            "n": self.n_questions,
        }


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile.

    Chosen over interpolation because with tens of samples an interpolated p95 implies
    precision the data does not have.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def evaluate_retriever(
    name: str,
    retrieve: Callable[[str, int], list[ScoredChunk]],
    goldens: Iterable[Golden],
    *,
    k: int = 10,
    clock: Callable[[], float] | None = None,
) -> Metrics:
    """Score a retrieval function against labelled questions.

    ``clock`` is injected so latency is measurable deterministically in CI. Pass
    ``time.perf_counter`` for a real measurement.
    """
    import time

    now = clock or time.perf_counter

    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    hits: list[float] = []
    latencies: list[float] = []
    candidates: list[float] = []
    by_difficulty: dict[str, list[float]] = {}

    goldens = list(goldens)
    for golden in goldens:
        relevant = set(golden.relevant_doc_ids)
        started = now()
        results = retrieve(golden.question, k)
        latencies.append((now() - started) * 1000)

        ndcg = ndcg_at_k(results, relevant, k)
        ndcgs.append(ndcg)
        recalls.append(recall_at_k(results, relevant, k))
        mrrs.append(mrr(results, relevant))
        hits.append(hit_rate_at_k(results, relevant, k))
        candidates.append(float(len(results)))
        by_difficulty.setdefault(golden.difficulty, []).append(ndcg)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731

    return Metrics(
        retriever=name,
        n_questions=len(goldens),
        ndcg_at_10=mean(ndcgs),
        recall_at_10=mean(recalls),
        mrr=mean(mrrs),
        hit_rate_at_10=mean(hits),
        p95_retrieval_ms=percentile(latencies, 0.95),
        mean_candidates=mean(candidates),
        per_difficulty={key: mean(values) for key, values in sorted(by_difficulty.items())},
    )


def regression_gate(
    candidate: Metrics, baseline: Metrics, *, max_drop: float = 0.02
) -> tuple[bool, str]:
    """Fail the build if ``candidate`` regresses against ``baseline``.

    The threshold is absolute rather than relative so a large baseline cannot buy a
    proportionally large regression.
    """
    drop = baseline.ndcg_at_10 - candidate.ndcg_at_10
    if drop > max_drop:
        return (
            False,
            f"nDCG@10 regressed by {drop:.4f} "
            f"({baseline.ndcg_at_10:.4f} -> {candidate.ndcg_at_10:.4f}), limit {max_drop}",
        )
    return True, f"nDCG@10 {candidate.ndcg_at_10:.4f} vs baseline {baseline.ndcg_at_10:.4f}"


def render_table(rows: Sequence[Metrics]) -> str:
    """Markdown table. The form the README and RESULTS.md both consume."""
    if not rows:
        return "_no results_"
    header = "| retriever | nDCG@10 | recall@10 | MRR | hit@10 | p95 ms | n |\n"
    header += "|---|---|---|---|---|---|---|\n"
    for metrics in rows:
        row = metrics.to_row()
        header += (
            f"| {row['retriever']} | {row['ndcg@10']} | {row['recall@10']} | "
            f"{row['mrr']} | {row['hit@10']} | {row['p95_ms']} | {row['n']} |\n"
        )
    return header


def save_metrics(path: Path, rows: Sequence[Metrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n", encoding="utf-8")

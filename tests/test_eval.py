"""Metric definitions and the CI gate.

These are the numbers a resume is built from, so each one is pinned against a case
where a plausible alternative implementation would give a different answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.rag import (
    Chunk,
    Golden,
    ScoredChunk,
    evaluate_retriever,
    load_goldens,
    ndcg_at_k,
    recall_at_k,
    regression_gate,
    render_table,
)
from gateway.rag.eval import hit_rate_at_k, mrr, percentile


def scored(doc_id: str, score: float = 1.0, chunk_id: str = "") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id or f"{doc_id}-0", doc_id=doc_id, text=doc_id),
        score=score,
    )


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    results = [scored("a.md"), scored("b.md")]
    assert ndcg_at_k(results, {"a.md", "b.md"}, k=10) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_is_retrieved() -> None:
    assert ndcg_at_k([], {"a.md"}, k=10) == 0.0


def test_ndcg_penalises_the_wrong_order() -> None:
    """Position matters: a relevant doc at rank 2 scores less than at rank 1."""
    right_first = ndcg_at_k([scored("a.md"), scored("x.md")], {"a.md"}, k=10)
    right_last = ndcg_at_k([scored("x.md"), scored("a.md")], {"a.md"}, k=10)

    assert right_first > right_last


def test_ndcg_discounts_ranks() -> None:
    at_1 = ndcg_at_k([scored("a.md")] + [scored("x.md")] * 20, {"a.md"}, k=10)
    at_5 = ndcg_at_k([scored("x.md")] * 4 + [scored("a.md")] + [scored("y.md")] * 5, {"a.md"}, k=10)

    assert at_1 > at_5


def test_ndcg_with_no_relevant_documents_is_zero() -> None:
    """An unanswerable question has no credit to award."""
    assert ndcg_at_k([scored("a.md")], set(), k=10) == 0.0


def test_ndcg_respects_the_k_cutoff() -> None:
    results = [scored("x.md")] * 5 + [scored("a.md")]
    assert ndcg_at_k(results, {"a.md"}, k=5) == 0.0
    assert ndcg_at_k(results, {"a.md"}, k=10) > 0.0


def test_ndcg_cannot_exceed_one_by_duplicating_chunks() -> None:
    """Ten chunks of one document each earn a full position discount. Without the
    bound, a retriever that returns duplicates scores above the ideal ordering of
    distinct documents, which is not a ranking quality at all."""
    duplicated = [scored("a.md", chunk_id=f"c{i}") for i in range(10)]

    assert ndcg_at_k(duplicated, {"a.md"}, k=10) == 1.0


def test_ndcg_stays_bounded_across_shapes() -> None:
    results = [scored("a.md", chunk_id=f"c{i}") for i in range(10)] + [scored("b.md")]

    assert 0.0 <= ndcg_at_k(results, {"a.md", "b.md"}, k=10) <= 1.0


def test_recall_counts_distinct_documents() -> None:
    """Two chunks from one document is not two relevant documents."""
    results = [scored("a.md", chunk_id="c1"), scored("a.md", chunk_id="c2")]
    assert recall_at_k(results, {"a.md", "b.md"}, k=10) == 0.5


def test_recall_is_one_when_everything_relevant_is_found() -> None:
    assert recall_at_k([scored("a.md"), scored("b.md")], {"a.md", "b.md"}, k=10) == 1.0


def test_mrr_rewards_early_relevant_results() -> None:
    results = [scored("x.md"), scored("y.md"), scored("a.md")]
    assert mrr(results, {"a.md"}) == pytest.approx(1 / 3)


def test_mrr_is_zero_without_a_hit() -> None:
    assert mrr([scored("x.md")], {"a.md"}) == 0.0


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k([scored("x.md"), scored("a.md")], {"a.md"}, k=2) == 1.0
    assert hit_rate_at_k([scored("x.md")], {"a.md"}, k=2) == 0.0


def test_percentile_uses_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.95) == 5.0
    assert percentile(values, 0.5) == 3.0


def test_percentile_of_nothing_is_zero() -> None:
    assert percentile([], 0.95) == 0.0


def test_evaluate_retriever_aggregates_over_questions() -> None:
    goldens = [
        Golden("q1", "query one", "a", ("a.md",)),
        Golden("q2", "query two", "b", ("b.md",)),
    ]

    def retrieve(query: str, k: int):
        return [scored("a.md")] if query == "query one" else [scored("wrong.md")]

    metrics = evaluate_retriever("fake", retrieve, goldens, clock=lambda: 0.0)

    assert metrics.n_questions == 2
    assert metrics.ndcg_at_10 == pytest.approx(0.5)
    assert metrics.hit_rate_at_10 == pytest.approx(0.5)


def test_evaluate_retriever_breaks_down_by_difficulty() -> None:
    goldens = [
        Golden("q1", "easy", "", ("a.md",), difficulty="single-hop"),
        Golden("q2", "hard", "", ("b.md",), difficulty="multi-hop"),
    ]

    def retrieve(query: str, k: int):
        return [scored("a.md")] if query == "easy" else []

    metrics = evaluate_retriever("fake", retrieve, goldens, clock=lambda: 0.0)

    assert metrics.per_difficulty["single-hop"] == pytest.approx(1.0)
    assert metrics.per_difficulty["multi-hop"] == 0.0


def test_evaluate_retriever_with_no_goldens_is_all_zeroes() -> None:
    metrics = evaluate_retriever("fake", lambda q, k: [], [], clock=lambda: 0.0)

    assert metrics.n_questions == 0
    assert metrics.ndcg_at_10 == 0.0


def test_regression_gate_passes_when_equal() -> None:
    baseline = evaluate_retriever("b", lambda q, k: [], [], clock=lambda: 0.0)
    candidate = evaluate_retriever("c", lambda q, k: [], [], clock=lambda: 0.0)

    ok, reason = regression_gate(candidate, baseline)
    assert ok
    assert "0.0000" in reason


def test_regression_gate_fails_on_a_drop() -> None:
    """The gate must actually fail, or it is decoration."""
    goldens = [Golden(f"q{i}", f"q{i}", "", ("a.md",)) for i in range(10)]

    good = evaluate_retriever("good", lambda q, k: [scored("a.md")], goldens, clock=lambda: 0.0)
    bad = evaluate_retriever("bad", lambda q, k: [scored("z.md")], goldens, clock=lambda: 0.0)

    ok, reason = regression_gate(bad, good, max_drop=0.02)
    assert not ok
    assert "regressed" in reason


def test_regression_gate_tolerates_a_small_drop() -> None:
    baseline = evaluate_retriever("b", lambda q, k: [], [], clock=lambda: 0.0)
    baseline.ndcg_at_10 = 0.5
    candidate = evaluate_retriever("c", lambda q, k: [], [], clock=lambda: 0.0)
    candidate.ndcg_at_10 = 0.49

    ok, _ = regression_gate(candidate, baseline, max_drop=0.02)
    assert ok


def test_regression_gate_allows_improvement() -> None:
    baseline = evaluate_retriever("b", lambda q, k: [], [], clock=lambda: 0.0)
    baseline.ndcg_at_10 = 0.3
    candidate = evaluate_retriever("c", lambda q, k: [], [], clock=lambda: 0.0)
    candidate.ndcg_at_10 = 0.6

    ok, _ = regression_gate(candidate, baseline)
    assert ok


def test_render_table_produces_markdown() -> None:
    goldens = [Golden("q1", "q", "", ("a.md",))]
    metrics = evaluate_retriever("bm25", lambda q, k: [scored("a.md")], goldens, clock=lambda: 0.0)

    table = render_table([metrics])
    assert "| bm25 |" in table
    assert "nDCG@10" in table


def test_render_table_with_nothing() -> None:
    assert render_table([]) == "_no results_"


def test_load_goldens_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "goldens.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "How?",
                        "answer": "Like this",
                        "relevant_doc_ids": ["a.md", "b.md"],
                        "difficulty": "multi-hop",
                    }
                ),
                "// a comment line",
                "",
                json.dumps({"id": "q2", "question": "Why?", "answer": "Because"}),
            ]
        ),
        encoding="utf-8",
    )

    goldens = load_goldens(path)

    assert len(goldens) == 2, "comments and blank lines must be skipped"
    assert goldens[0].relevant_doc_ids == ("a.md", "b.md")
    assert goldens[1].difficulty == "single-hop", "must default"

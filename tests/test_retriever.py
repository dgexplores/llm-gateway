"""Retrieval contract: what a retriever must promise, and the edge cases that break it."""

from __future__ import annotations

import pytest

from gateway.rag import BM25Retriever, Chunk, NullRetriever, ScoredChunk, normalize, top_k


def chunk(chunk_id: str, doc_id: str, text: str = "text") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text)


def test_null_retriever_returns_nothing() -> None:
    """With RAG off the gateway must answer without context, not invent a source."""
    assert NullRetriever().retrieve("anything", k=10) == []


def test_results_respect_k() -> None:
    scored = normalize(
        [
            ScoredChunk(chunk=Chunk(f"c{i}", f"d{i}", f"text {i}"), score=float(20 - i))
            for i in range(20)
        ]
    )
    assert len(top_k(scored, 5)) == 5
    assert len(top_k(scored, 20)) == 20


def test_top_k_clamps_a_negative_k() -> None:
    """A misconfigured caller must get an empty list, not an inverted slice."""
    scored = normalize(
        [
            ScoredChunk(chunk=Chunk("c1", "d1", "a"), score=0.5),
            ScoredChunk(chunk=Chunk("c2", "d2", "b"), score=0.9),
        ]
    )
    assert top_k(scored, -5) == []


def test_normalize_is_a_noop_when_all_scores_are_equal() -> None:
    """Equal scores would divide by zero; returning them unchanged avoids NaN."""
    scored = [ScoredChunk(chunk=Chunk("c1", "d1", "a"), score=0.5)]
    assert normalize(scored)[0].score == 0.5


def test_normalize_rescales_to_unit_range() -> None:
    scored = [
        ScoredChunk(chunk=Chunk("c1", "d1", "a"), score=2.0),
        ScoredChunk(chunk=Chunk("c2", "d2", "b"), score=6.0),
    ]
    values = [s.score for s in normalize(scored)]
    assert values == [0.0, 1.0]


def test_normalize_handles_empty_input() -> None:
    assert normalize([]) == []


def test_scored_chunk_exposes_chunk_fields() -> None:
    scored = ScoredChunk(chunk=Chunk("c1", "doc.md", "hello", title="Doc"), score=1.0)
    assert scored.chunk_id == "c1"
    assert scored.doc_id == "doc.md"
    assert scored.text == "hello"


@pytest.mark.parametrize("k", [0, 1, 3, 100])
def test_retriever_respects_any_k(k: int) -> None:
    retriever = BM25Retriever.build(
        [Chunk(f"c{i}", f"d{i}", f"term{i} unique content {i}") for i in range(5)]
    )
    assert len(retriever.retrieve("unique", k=k)) <= k

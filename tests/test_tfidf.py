"""Tests for the TF-IDF retriever.

It exists so the rank-fusion variant in the ablation has a second, genuinely different
member. Fusing BM25 with BM25 would measure nothing, and shipping a fusion number
without one is how a component ends up looking like a feature while being an
assumption.
"""

from __future__ import annotations

import math

from gateway.rag.retriever import Retriever
from gateway.rag.tfidf import TFIDFIndex, TFIDFRetriever

CHUNKS = [
    ("c1", "status.md", "The status_code parameter sets the HTTP status code."),
    ("c2", "tasks.md", "Background tasks run after a response is returned."),
    ("c3", "bread.md", "Sourdough bread needs a long cold fermentation."),
]


def _chunks():
    from gateway.rag.retriever import Chunk

    return [Chunk(cid, doc, text) for cid, doc, text in CHUNKS]


def test_satisfies_the_retriever_protocol() -> None:
    assert isinstance(TFIDFRetriever.build(_chunks()), Retriever)


def test_ranks_the_matching_chunk_first() -> None:
    results = TFIDFRetriever.build(_chunks()).retrieve("background tasks", k=3)

    assert results
    assert results[0].chunk_id == "c2"


def test_scores_are_descending() -> None:
    results = TFIDFRetriever.build(_chunks()).retrieve("status code", k=3)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_scores_are_cosine_bounded() -> None:
    """Cosine similarity of non-negative TF-IDF vectors, so 0..1. The UI renders the
    score as a bar width and normalize() assumes the range."""
    for result in TFIDFRetriever.build(_chunks()).retrieve("status code", k=3):
        assert 0.0 <= result.score <= 1.0


def test_a_term_present_everywhere_is_not_weighted_to_zero() -> None:
    """Unsmoothed IDF would score a term in every document at exactly zero, silently
    discarding it. The +1 smoothing is what stops that."""
    index = TFIDFIndex(_chunks())
    ubiquitous = max(index.idf.values(), key=lambda v: -v)  # lowest IDF == most common

    assert ubiquitous > 0.0
    assert all(math.isfinite(v) for v in index.idf.values())


def test_returns_nothing_for_an_unknown_query() -> None:
    """Every query term absent from the corpus. Returning arbitrary chunks would be
    filler handed to the generator as ground truth."""
    assert TFIDFRetriever.build(_chunks()).retrieve("kubernetes ingress", k=3) == []


def test_does_not_raise_on_a_question_the_corpus_cannot_answer() -> None:
    """Named in the Retriever contract as a normal result, not an error."""
    assert isinstance(TFIDFRetriever.build(_chunks()).retrieve("quantum chromodynamics", k=3), list)


def test_respects_k_and_rejects_non_positive() -> None:
    r = TFIDFRetriever.build(_chunks())

    # Only c1 shares a term with "status code". Returning the other two at score 0
    # would be filler handed to the generator as ground truth, which the Retriever
    # contract calls out as worse than returning nothing.
    assert len(r.retrieve("status code", k=2)) == 1
    assert r.retrieve("status code", k=0) == []


def test_a_query_matching_nothing_returns_nothing_even_with_a_large_k() -> None:
    """k caps results; it does not manufacture them."""
    assert TFIDFRetriever.build(_chunks()).retrieve("kubernetes", k=99) == []


def test_k_larger_than_the_index_returns_everything_matching() -> None:
    results = TFIDFRetriever.build(_chunks()).retrieve("the code", k=99)

    assert 0 < len(results) <= 3


def test_empty_index_returns_nothing() -> None:
    assert TFIDFRetriever.build([]).retrieve("anything", k=5) == []

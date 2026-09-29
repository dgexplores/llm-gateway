"""Tests for the LangChain retriever adapter.

These run against the real ``langchain-core`` rather than a stand-in, because the two
bugs this adapter exists to prevent — inverted scores and an unimplemented relevance
hook — are both properties of the actual library. A fake would have agreed with whatever
the adapter assumed.

Retrieval *quality* is not tested, and cannot be here. The only embedding available
without a network or an API key is ``DeterministicFakeEmbedding``, whose vectors are
random: measured against it, a query about status codes ranks a background-tasks
passage first. Asserting that would assert noise. What is asserted is that the adapter
faithly reports whatever the store says.
"""

from __future__ import annotations

import math

import pytest

from gateway.rag.langchain import LangChainRetriever, _to_similarity
from gateway.rag.retriever import Chunk, ScoredChunk, normalize

pytest.importorskip("langchain_core", reason="langchain extra not installed")


def _chunks() -> list[Chunk]:
    return [
        Chunk("c1", "status.md", "The status_code parameter sets the HTTP status code.", "Status"),
        Chunk("c2", "tasks.md", "Background tasks run after a response is returned.", "Tasks"),
        Chunk("c3", "bread.md", "Sourdough bread needs a long fermentation.", "Bread"),
    ]


@pytest.fixture
def retriever() -> LangChainRetriever:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return LangChainRetriever.from_chunks(
        _chunks(), DeterministicFakeEmbedding(size=64), name="lc-test"
    )


# --- the score direction, which is the whole point of the adapter


def test_distance_is_inverted_into_similarity() -> None:
    """LangChain ranks by distance where lower is better. ScoredChunk means the
    opposite, and normalize() maps min to 0. Passing distance through would rank
    correctly and draw the confidence bar backwards.

    The property is direction, not a particular value at zero: the logistic is a
    bounded transform, so 0 maps to the midpoint rather than to 1."""
    assert _to_similarity(1.0) < _to_similarity(0.0)
    assert _to_similarity(5.0) < _to_similarity(1.0)
    assert _to_similarity(0.0) == pytest.approx(0.5)


def test_negative_distance_does_not_produce_a_score_above_one() -> None:
    """Cosine distance is negative for dissimilar vectors, so the naive 1 - d would
    exceed 1.0 and invert the bar again."""
    assert _to_similarity(-0.5) <= 1.0
    assert _to_similarity(-0.5) > 0


def test_similarity_is_strictly_decreasing_in_distance() -> None:
    """Ordering must be preserved exactly for the sort in retrieve() to be correct.

    Stated pairwise: a larger distance must always score lower. Asserting "the list is
    sorted" instead would only be true if the input were pre-sorted, which says nothing
    about the function."""
    distances = [-2.0, -0.5, 0.0, 0.25, 3.0, 10.0]
    scores = [_to_similarity(d) for d in distances]

    for nearer, farther in zip(distances, distances[1:], strict=False):
        assert _to_similarity(nearer) > _to_similarity(farther)

    assert len(set(scores)) == len(scores)


def test_scores_come_back_highest_first(retriever: LangChainRetriever) -> None:
    results = retriever.retrieve("status code", k=3)

    assert results, "a non-empty index must return something"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_every_score_is_within_zero_and_one(retriever: LangChainRetriever) -> None:
    """normalize() assumes this, and the UI renders it as a bar width."""
    for result in retriever.retrieve("anything", k=3):
        assert 0.0 <= result.score <= 1.0


# --- contract compliance


def test_satisfies_the_project_retriever_protocol(retriever: LangChainRetriever) -> None:
    from gateway.rag.retriever import Retriever

    assert isinstance(retriever, Retriever)


def test_respects_k(retriever: LangChainRetriever) -> None:
    assert len(retriever.retrieve("status code", k=2)) == 2


def test_k_larger_than_the_index_returns_everything(retriever: LangChainRetriever) -> None:
    assert len(retriever.retrieve("status code", k=99)) == 3


def test_non_positive_k_returns_nothing(retriever: LangChainRetriever) -> None:
    assert retriever.retrieve("status code", k=0) == []
    assert retriever.retrieve("status code", k=-3) == []


def test_a_query_with_no_overlap_does_not_raise(retriever: LangChainRetriever) -> None:
    """Named in the Retriever contract as a normal result, not an error. A semantic
    store always returns its nearest neighbours, which is the real reason this matters:
    there is no such thing as a lexical miss here."""
    results = retriever.retrieve("quantum chromodynamics", k=3)

    assert isinstance(results, list)


def test_empty_index_returns_empty() -> None:
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=8))
    empty = LangChainRetriever(store)

    assert empty.retrieve("anything", k=5) == []


# --- metadata round-trip


def test_metadata_survives_the_round_trip(retriever: LangChainRetriever) -> None:
    results = retriever.retrieve("status code", k=3)

    for result in results:
        assert result.chunk_id
        assert result.doc_id
        assert result.text


def test_doc_ids_are_preserved(retriever: LangChainRetriever) -> None:
    """Evaluation is computed per doc_id. A wrapper that drops it silently breaks the
    metric rather than failing loudly."""
    results = retriever.retrieve("status code", k=3)
    found = {r.doc_id for r in results}

    assert found <= {"status.md", "tasks.md", "bread.md"}
    assert found


def test_title_is_preserved(retriever: LangChainRetriever) -> None:
    results = retriever.retrieve("status code", k=3)

    assert all(r.chunk.title for r in results)


# --- the API trap


def test_falls_back_when_the_store_cannot_provide_relevance_scores(
    retriever: LangChainRetriever,
) -> None:
    """InMemoryVectorStore inherits similarity_search_with_relevance_scores from the
    base class and then raises NotImplementedError, because it does not implement
    _select_relevance_score_fn. Calling it unconditionally breaks on the default store."""
    with pytest.raises(NotImplementedError):
        retriever._store.similarity_search_with_relevance_scores("x", k=1)

    # The adapter must survive it and still return ordered results.
    assert retriever.retrieve("status code", k=2)


def test_prefers_native_relevance_scores_when_available() -> None:
    """A store that does implement the hook gets its own 0..1 score, not the
    transformed distance, so a well-calibrated store is not second-guessed."""
    from langchain_core.documents import Document

    class Store:
        def similarity_search_with_relevance_scores(self, query, k):
            return [
                (Document(page_content="a", metadata={"chunk_id": "1", "doc_id": "d"}), 0.91),
                (Document(page_content="b", metadata={"chunk_id": "2", "doc_id": "d"}), 0.42),
            ]

        def similarity_search_with_score(self, query, k):
            raise AssertionError("must not be called when relevance scores are available")

    results = LangChainRetriever(Store()).retrieve("q", k=2)

    assert [r.score for r in results] == [0.91, 0.42]


# --- interop with the rest of the project


def test_output_feeds_normalize_for_display(retriever: LangChainRetriever) -> None:
    results = retriever.retrieve("status code", k=3)
    rescaled = normalize(results)

    assert all(isinstance(s, ScoredChunk) for s in rescaled)
    assert max(s.score for s in rescaled) == pytest.approx(1.0)


def test_name_records_the_embedding_model() -> None:
    """The name lands in the results table, so it has to say which embedding produced
    the number. A column of rows all reading 'semantic' is not comparable."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    r = LangChainRetriever.from_chunks(_chunks(), DeterministicFakeEmbedding(size=64), name="lc")

    assert r.name.startswith("lc-")
    assert "Fake" in r.name or "fake" in r.name


@pytest.mark.parametrize("distance", [-1e6, -50.0, -1.0, -0.118, 0.0, 0.118, 1.0, 50.0, 1e6])
def test_similarity_stays_bounded_at_both_extremes(distance: float) -> None:
    """The first version of this used 1 / (1 + d) and returned 1.13 for a distance of
    -0.118, because cosine distance is negative for dissimilar vectors. Bounded output
    is part of the contract, not a detail."""
    score = _to_similarity(distance)

    assert 0.0 <= score <= 1.0
    assert math.isfinite(score)


def test_extreme_negative_distance_does_not_overflow() -> None:
    """A naive exp(d) returns inf for a large negative distance, and inf would poison
    every sort and the average behind it."""
    # Past the clamp a negative distance is treated as maximally near, so the score
    # saturates near 1 rather than at 0.
    score = _to_similarity(-1e6)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
    # Saturates at 1.0 rather than overflowing, which is the point.
    assert score >= 0.99

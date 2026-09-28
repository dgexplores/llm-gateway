"""BM25 behaviour, focused on the cases that silently produce bad results."""

from __future__ import annotations

from gateway.rag import BM25Retriever, Chunk


def build(docs: dict[str, str]) -> BM25Retriever:
    return BM25Retriever.build(
        [Chunk(chunk_id=cid, doc_id=cid, text=text, title=cid) for cid, text in docs.items()]
    )


def test_ranks_the_lexically_matching_document_first() -> None:
    retriever = build(
        {
            "a.md": "This page describes background tasks and scheduling.",
            "b.md": "This page explains OAuth2 authentication scopes.",
            "c.md": "This page covers response model serialization.",
        }
    )

    results = retriever.retrieve("OAuth2 authentication scopes", k=3)

    assert results[0].doc_id == "b.md"


def test_returns_nothing_for_a_query_with_no_overlap() -> None:
    """Empty is correct. Returning low-scoring filler is worse: the generator treats
    the chunk list as ground truth."""
    retriever = build({"a.md": "background tasks and scheduling"})

    assert retriever.retrieve("quantum chromodynamics", k=5) == []


def test_empty_index_returns_nothing() -> None:
    assert BM25Retriever.build([]).retrieve("anything", k=5) == []


def test_empty_query_returns_nothing() -> None:
    retriever = build({"a.md": "some content about caching"})
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("!!! ???", k=5) == []


def test_title_matches_outrank_body_matches() -> None:
    """A term in a heading is stronger evidence than one buried in body text."""
    retriever = BM25Retriever.build(
        [
            Chunk("c1", "a.md", "filler text here", title="Caching strategies"),
            Chunk("c2", "b.md", "caching is mentioned once in passing", title="Unrelated"),
        ]
    )

    results = retriever.retrieve("caching", k=2)
    assert results[0].doc_id == "a.md"


def test_scores_are_positive_and_ordered() -> None:
    retriever = build(
        {
            "a.md": "dependency injection container resolves services",
            "b.md": "dependency injection is discussed briefly",
            "c.md": "unrelated content about static files",
        }
    )

    results = retriever.retrieve("dependency injection", k=3)
    scores = [s.score for s in results]

    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_a_term_in_every_document_does_not_beat_a_rare_term() -> None:
    """The reason IDF exists. A stopword-like match must not dominate."""
    retriever = build(
        {
            "a.md": "the guide covers many topics",
            "b.md": "the guide covers many topics",
            "c.md": "the guide covers many topics",
            "rare.md": "the guide covers retryexhaustion handling",
        }
    )

    results = retriever.retrieve("the retryexhaustion", k=4)
    assert results[0].doc_id == "rare.md"


def test_chunk_ids_break_ties_deterministically() -> None:
    """Stable ordering matters: without it, two runs produce different tables."""
    retriever = build({"b.md": "same text", "a.md": "same text", "c.md": "same text"})

    first = [s.chunk_id for s in retriever.retrieve("same", k=3)]
    second = [s.chunk_id for s in retriever.retrieve("same", k=3)]

    assert first == second == sorted(first)


def test_index_is_built_once_and_reused() -> None:
    retriever = build({"a.md": "content about timeouts"})

    assert retriever.index.size == 1
    assert retriever.index.avg_length > 0


def test_punctuation_and_case_do_not_affect_matching() -> None:
    retriever = build({"a.md": "Use Depends() for dependencies."})

    assert retriever.retrieve("use depends for dependencies", k=1)
    assert retriever.retrieve("USE DEPENDS() FOR DEPENDENCIES", k=1)

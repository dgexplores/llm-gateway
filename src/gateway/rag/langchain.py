"""LangChain retriever adapter.

Wraps any LangChain ``VectorStore`` or ``BaseRetriever`` behind this project's
``Retriever`` contract, so a semantic index can be benchmarked against BM25 without
either side knowing about the other.

Two things here are not plumbing, and both were found by running the real library
rather than reading its source:

**LangChain returns distance; this project returns similarity.** Every LangChain
vector store ranks by distance, where *lower is better*. ``ScoredChunk.score`` means
the opposite, and ``retriever.normalize`` maps min to 0 and max to 1 for the UI. Passing
distance straight through would therefore rank correctly but display the confidence bar
backwards. Scores are converted on the way in, not trusted.

**``similarity_search_with_relevance_scores`` does not work everywhere.** The base class
exposes it, and it is the obvious call to make. ``InMemoryVectorStore`` raises
``NotImplementedError`` because it does not implement ``_select_relevance_score_fn``. The
adapter prefers it when a store supports it and falls back to distance otherwise, so the
same code works across stores.

**What the tests do and do not prove.** They exercise the mechanics against the real
library: ordering, the distance-to-similarity direction, metadata mapping, k-clamping,
and the empty result. They do not demonstrate retrieval *quality*, because the only
embedding available without a network or an API key is ``DeterministicFakeEmbedding``,
whose vectors are random by construction. Measured against it, a query about status
codes ranks a passage about background tasks first. A semantic-quality claim needs a
real embedding model, and none of the numbers in ``docs/RESULTS.md`` come from here.
"""

from __future__ import annotations

import math
from typing import Any

from .retriever import Chunk, ScoredChunk

# Below the point where math.exp overflows a float.
_EXP_LIMIT = 700.0


def _to_similarity(distance: float) -> float:
    """Convert a distance to a higher-is-better score inside (0, 1).

    A logistic in the distance: strictly decreasing, so ordering is preserved exactly,
    and bounded, so ``normalize`` and the UI confidence bar can both rely on the range.

    The obvious alternatives are both wrong here. ``1 - d`` exceeds 1.0 for the negative
    cosine distances that a store returns for dissimilar vectors. ``1 / (1 + d)`` does
    too, and also breaks at ``d = -1`` — a first attempt at this function did exactly
    that, which is why the bound is now asserted rather than assumed.

    Computed in the two stable branches, because a plain ``exp(d)`` overflows for a
    large negative distance and would return ``inf``.

    The negative branch is ``exp(-d) / (1 + exp(-d))``. That is the numerically stable
    rewrite of the *same* function: multiply numerator and denominator by ``exp(-d)``.
    Writing ``exp(d) / (1 + exp(d))`` instead is the complement, and it inverts the
    ranking for every negative distance. An earlier version of this file did exactly
    that, which ranked the least similar passages first for most of a real cosine
    distance range — and no test caught it at the time, because the retriever re-sorts
    by score, so a confidently wrong score still produced a self-consistent-looking
    list. The ordering property is now asserted directly.
    """
    # math.exp overflows above ~709, and a cosine distance can be arbitrarily large in
    # either direction. Clamping keeps the result finite; every distance past the clamp
    # is infinitely far relative to the other and collapsing them is correct.
    d = max(-_EXP_LIMIT, min(_EXP_LIMIT, float(distance)))
    if d >= 0.0:
        return 1.0 / (1.0 + math.exp(d))
    e = math.exp(-d)
    return e / (1.0 + e)


def _search_with_relevance(store: Any, query: str, k: int) -> list[tuple[Any, float]] | None:
    """Use the store's own 0..1 relevance score when it has one.

    Returns ``None`` when the store does not implement the hook, which is the documented
    behaviour of ``InMemoryVectorStore`` and of several hosted stores.
    """
    try:
        return store.similarity_search_with_relevance_scores(query, k=k)
    except NotImplementedError:
        return None


class LangChainRetriever:
    """Adapts a LangChain vector store to this project's retriever contract.

    ``name`` shows up in the UI and in the results files, so it should say what the
    index actually is — "langchain-inmemory-fake" rather than a vague "semantic", so a
    reader can tell from the table which embedding produced the number.
    """

    def __init__(self, store: Any, *, name: str = "langchain", model: str = "") -> None:
        self._store = store
        self.name = name
        self.model = model

    @classmethod
    def from_chunks(
        cls, chunks: list[Chunk], embeddings: Any, *, name: str = "langchain"
    ) -> LangChainRetriever:
        """Build an in-memory LangChain index over this project's ``Chunk`` type.

        ``embeddings`` is any LangChain ``Embeddings``. The split is deliberate: choosing
        the embedding model is the caller's decision, because it is the one choice that
        determines whether semantic retrieval beats lexical here.
        """
        from langchain_core.documents import Document
        from langchain_core.vectorstores import InMemoryVectorStore

        documents = [
            Document(
                page_content=chunk.text,
                metadata={"doc_id": chunk.doc_id, "chunk_id": chunk.chunk_id, "title": chunk.title},
            )
            for chunk in chunks
        ]
        store = InMemoryVectorStore.from_documents(documents, embeddings)
        model = getattr(embeddings, "model", None) or type(embeddings).__name__
        return cls(store, name=f"{name}-{model}", model=str(model))

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        from langchain_core.documents import Document

        if k <= 0:
            return []

        results = _search_with_relevance(self._store, query, k)
        if results is not None:
            # Already 0..1, higher is better. Used as-is.
            pairs = [(doc, float(score)) for doc, score in results]
        else:
            # Distance, lower is better. Inverted here so downstream never sees it.
            pairs = [
                (doc, _to_similarity(score))
                for doc, score in self._store.similarity_search_with_score(query, k=k)
            ]

        found: list[ScoredChunk] = []
        for doc, score in pairs:
            if not isinstance(doc, Document):
                continue
            metadata = doc.metadata or {}
            found.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=str(metadata.get("chunk_id", "")),
                        doc_id=str(metadata.get("doc_id", "")),
                        text=doc.page_content,
                        title=str(metadata.get("title", "")),
                    ),
                    score=round(score, 6),
                )
            )

        # The store may return results in its own order; the contract promises
        # descending score, and a display bar that disagrees with the list order is
        # worse than no bar.
        found.sort(key=lambda s: s.score, reverse=True)
        return found[:k]

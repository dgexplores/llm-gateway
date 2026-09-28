"""Retrieval contract.

The boundary between "find relevant text" and "answer the question" is where RAG
systems usually blur. Keeping it explicit here means a retriever can be swapped,
benchmarked, and compared without touching generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text.

    ``doc_id`` is stable across chunking runs so evaluation results remain
    reproducible; ``chunk_id`` is unique within a document.
    """

    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with a relevance score.

    The score is meaningful only relative to other scores *from the same retriever*.
    Comparing 0.42 from BM25 against 0.81 from a cross-encoder is a category error,
    and the type says so rather than leaving it to a comment nobody reads.
    """

    chunk: Chunk
    score: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def text(self) -> str:
        return self.chunk.text


@runtime_checkable
class Retriever(Protocol):
    """Anything that ranks chunks against a query.

    Contract:

    - ``retrieve`` returns at most ``k`` results, ordered by descending score.
    - Returns an empty list when nothing is relevant. Returning low-scoring
      filler is worse than returning nothing, because the generator treats the
      chunk list as ground truth.
    - Must not raise on a query with no lexical overlap; that is a normal result.
    - Scores need not be normalized across retrievers, only within one.
    """

    name: str

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]: ...


class NullRetriever:
    """Returns nothing. The correct behaviour when retrieval is disabled.

    Not a stub: with RAG off, the gateway must answer without context rather than
    invent a context source.
    """

    name = "null"

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return []


def normalize(scored: list[ScoredChunk]) -> list[ScoredChunk]:
    """Rescale scores to 0..1 within a single result set.

    Useful for displaying a confidence bar in the UI. Applied per result set
    because cross-retriever comparison is not meaningful.
    """
    if not scored:
        return []
    values = [s.score for s in scored]
    low, high = min(values), max(values)
    if high == low:
        return scored
    span = high - low
    return [ScoredChunk(chunk=s.chunk, score=round((s.score - low) / span, 4)) for s in scored]


def top_k(scored: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    """Truncate to ``k``, preserving the retriever's ordering.

    ``k`` is clamped to a positive value so a misconfigured caller cannot request
    a negative slice and silently get inverted results.
    """
    return scored[: max(0, k)]

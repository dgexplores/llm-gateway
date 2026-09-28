"""BM25 retrieval.

The first real retriever, and the baseline every later one must beat. Building the
strongest retriever first and then comparing would leave nothing to measure against,
and a project that cannot show a delta is indistinguishable from a tutorial.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .retriever import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 parameters from the original Okapi paper. Exposed so the effect of changing
# them is measurable rather than a matter of taste.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# Below this, a match is treated as noise. Without a floor, a query with one rare
# term matched in every document returns the entire corpus.
MIN_SCORE = 0.0


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class BM25Index:
    """Sparse lexical index over a fixed chunk set.

    Built once and queried many times. Construction is O(total tokens); each query is
    O(query terms), which is why this is a credible baseline rather than a toy.
    """

    chunks: list[Chunk] = field(default_factory=list)
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    def __post_init__(self) -> None:
        self._tokens: list[list[str]] = []
        self._frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[int]] = {}
        self._build()

    def _build(self) -> None:
        total = 0
        for position, chunk in enumerate(self.chunks):
            # Title terms are repeated: a match in a heading is stronger evidence
            # than a match buried in body text, and this is the cheapest way to
            # express that without a learned weighting model.
            tokens = tokenize(chunk.text) + tokenize(chunk.title) * 2
            counts = Counter(tokens)
            self._tokens.append(tokens)
            self._frequencies.append(counts)
            self._lengths.append(len(tokens))
            total += len(tokens)
            for term in counts:
                self._postings.setdefault(term, []).append(position)

        self._avg_length = (total / len(self.chunks)) if self.chunks else 0.0
        self._n = len(self.chunks)

    @property
    def size(self) -> int:
        return self._n

    @property
    def avg_length(self) -> float:
        """Mean tokens per chunk. Exposed because a corpus of near-empty chunks is a
        common ingest bug that silently degrades length normalization."""
        return self._avg_length

    def _idf(self, term: str) -> float:
        """Robertson's IDF with the +1 smoothing that avoids negatives.

        A term appearing in every document is genuinely uninformative, and a negative
        weight would let it actively penalise matches.
        """
        df = len(self._postings.get(term, ()))
        if df == 0:
            return 0.0
        import math

        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or self._n == 0:
            return []

        scores: dict[int, float] = {}
        for term in set(terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for position in self._postings.get(term, ()):
                freq = self._frequencies[position][term]
                length = self._lengths[position]
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (length / self._avg_length if self._avg_length else 1.0)
                )
                contribution = idf * (freq * (self.k1 + 1)) / (denominator or 1.0)
                scores[position] = scores.get(position, 0.0) + contribution

        ranked = [
            ScoredChunk(chunk=self.chunks[position], score=round(score, 6))
            for position, score in scores.items()
            if score > MIN_SCORE
        ]
        ranked.sort(key=lambda s: (-s.score, s.chunk_id))
        return ranked[:k]


@dataclass
class BM25Retriever:
    name: str = "bm25"

    index: BM25Index = field(default_factory=BM25Index)

    @classmethod
    def build(cls, chunks: list[Chunk], **kwargs: object) -> BM25Retriever:
        return cls(index=BM25Index(chunks=chunks, **kwargs))  # type: ignore[arg-type]

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return self.index.search(query, k=k)

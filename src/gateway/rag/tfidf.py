"""TF-IDF cosine retrieval.

A second lexical ranker, which exists so that :class:`~gateway.rag.hybrid.HybridRetriever`
can be measured rather than assumed. Rank fusion needs members whose rankings genuinely
differ; fusing BM25 with BM25 measures nothing.

It differs from BM25 in the two ways that matter for a fusion: BM25 saturates term
frequency and applies a length normalisation that rewards short documents, while this
is plain cosine over an L2-normalised TF-IDF vector. Results overlap heavily, which is
the honest expectation and the reason the ablation reports the number instead of
predicting a gain.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .bm25 import tokenize
from .retriever import Chunk, ScoredChunk


@dataclass
class TFIDFIndex:
    """A sparse-free TF-IDF matrix held as parallel vectors.

    The corpus is 1,448 chunks, so the dense form is small enough to be simple and the
    clarity is worth more than the memory. This would not survive a million chunks; the
    note is here so nobody reaches for it at that size.
    """

    chunks: list[Chunk]
    vectors: list[dict[str, float]] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        documents = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        count = len(documents)

        document_frequency: Counter[str] = Counter()
        for terms in documents:
            document_frequency.update(terms.keys())

        # Smoothed IDF. The +1 keeps a term present in every document from scoring zero
        # weight, which would silently make a term that appears everywhere worthless.
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }

        self.vectors = []
        for terms in documents:
            vector = {term: freq * self.idf[term] for term, freq in terms.items()}
            norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
            self.vectors.append({term: value / norm for term, value in vector.items()})

    def _query_vector(self, query: str) -> dict[str, float]:
        terms = Counter(tokenize(query))
        vector = {
            term: freq * self.idf[term]
            for term, freq in terms.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if not norm:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def search(self, query: str, k: int = 10) -> list[ScoredChunk]:
        query_vector = self._query_vector(query)
        if not query_vector:
            # Every query term is unknown to the corpus. Returning nothing is the
            # contract; returning arbitrary chunks would hand the generator filler.
            return []

        scored: list[ScoredChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            if len(query_vector) < len(vector):
                total = sum(weight * vector.get(term, 0.0) for term, weight in query_vector.items())
            else:
                total = sum(weight * vector.get(term, 0.0) for term, weight in vector.items())
            if total > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=round(total, 6)))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


@dataclass
class TFIDFRetriever:
    index: TFIDFIndex
    name: str = "tfidf"

    @classmethod
    def build(cls, chunks: list[Chunk], *, name: str = "tfidf") -> TFIDFRetriever:
        return cls(index=TFIDFIndex(list(chunks)), name=name)

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        return self.index.search(query, k=max(0, k))

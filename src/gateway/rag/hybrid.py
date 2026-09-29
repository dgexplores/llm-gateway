"""Hybrid retrieval by reciprocal rank fusion.

BM25 alone is blind to paraphrases. A dense retriever alone is blind to exact
identifiers: a query for ``status_code`` must find the document containing
``status_code``, and an embedding averages that away. Fusing the two ranks is the
cheap fix, and it is cheap in a way that matters here — it needs no model, so it runs
in CI and on a clean clone.

Reciprocal rank fusion (Cormack et al.) is used rather than score blending because the
two retrievers' scores are not on a comparable scale. RRF uses only rank, so no
normalisation is required and no calibration can be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import Retriever, ScoredChunk

# 60 is the value from the original paper. It is not a tuned constant here: tuning it
# on the golden set would be fitting the evaluation, which is the thing the gate exists
# to prevent.
RRF_K = 60


@dataclass
class HybridRetriever:
    """Fuses several retrievers over their rankings.

    ``name`` includes the members so a results table records exactly what produced a
    row, rather than a label that could mean different things across runs.
    """

    retrievers: list[Retriever] = field(default_factory=list)
    k: int = RRF_K
    weights: dict[str, float] = field(default_factory=dict)
    rrf_k: int = RRF_K

    def __post_init__(self) -> None:
        if not self.retrievers:
            raise ValueError("a hybrid retriever needs at least one member")
        if self.k <= 0:
            raise ValueError("k must be positive")
        self.name = "+".join(r.name for r in self.retrievers)

    def _weight(self, retriever: Retriever) -> float:
        return self.weights.get(retriever.name, 1.0)

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        # Each member is asked for more than k, because a chunk ranked 8th by one
        # retriever can be in the fused top k when the other disagrees. Asking only
        # for k would make fusion a no-op on everything past k.
        depth = max(k * 3, 30)
        scores: dict[str, float] = {}
        chunks: dict[str, ScoredChunk] = {}

        for retriever in self.retrievers:
            weight = self._weight(retriever)
            for rank, hit in enumerate(retriever.retrieve(query, k=depth), start=1):
                key = hit.chunk_id
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank)
                chunks.setdefault(key, hit)

        ranked = sorted(
            (
                ScoredChunk(chunk=hit.chunk, score=round(score, 6))
                for key, score in scores.items()
                if (hit := chunks.get(key)) is not None
            ),
            key=lambda s: (-s.score, s.chunk_id),
        )
        return ranked[:k]


def explain_fusion(
    query: str,
    hybrid: HybridRetriever,
    chunk_id: str,
    depth: int = 30,
) -> list[dict[str, object]]:
    """Why a chunk ranked where it did.

    Without this, a hybrid result is unauditable: a chunk can surface for reasons no
    reader can reconstruct. Returns each member's rank and its contribution.
    """
    rows: list[dict[str, object]] = []
    for retriever in hybrid.retrievers:
        hits = retriever.retrieve(query, k=depth)
        weight = hybrid._weight(retriever)
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id == chunk_id:
                rows.append(
                    {
                        "retriever": retriever.name,
                        "rank": rank,
                        "weight": weight,
                        "contribution": round(weight / (hybrid.rrf_k + rank), 6),
                        "raw_score": round(hit.score, 4),
                    }
                )
                break
        else:
            rows.append(
                {"retriever": retriever.name, "rank": None, "contribution": 0.0}
            )
    return rows

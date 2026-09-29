"""Relevance gating: refuse to answer a question the passages do not address.

Everything else in this project checks *grounding* — that a sentence in the answer appears
in the retrieved passages. That is not the same question. The corpus can be irrelevant and
grounding still passes, which is why the measured abstention rate on unanswerable
questions was 0%: the extractive provider quotes the passages by construction, so every
sentence it produces is trivially grounded, and nothing was looking at whether the
passages had anything to do with the question.

Two lexical gates were measured first and both rejected, because the answerable and
unanswerable score distributions overlap. See ``docs/RESULTS.md``. This is the third
attempt and the first that separates, because it is a different question: a cross-encoder
trained for query-passage relevance rather than a word-overlap heuristic.

**It is opt-in and off by default**, deliberately. A threshold that catches 75% of
unanswerable questions also abstains on 11% of answerable ones, and it costs a model
download and roughly 10 ms per query. That is a defensible trade for a tool whose thesis is
knowing when it does not know, and an indefensible one to impose on someone who just wants
BM25. The numbers are in ``docs/RESULTS.md`` so the choice can be made deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chosen from the measured trade-off curve, not picked by feel. End to end, through the
# real pipeline and over the best of k retrieved passages:
#
#   threshold   answerable answered   unanswerable caught
#   -9.0        63/63  (100%)         6/8  (75%)
#   -3.0        56/63  ( 89%)         8/8  (100%)
#
# -9.0 is the default because it costs nothing measured: it answers every question the
# corpus can address while catching three quarters of the ones it cannot. -3.0 catches the
# rest at the price of withholding one answer in nine, which is a defensible choice for a
# tool whose thesis is knowing when it does not know, and not one to impose by default.
#
# An earlier estimate of this curve, taken by scoring only the top-ranked passage,
# suggested -9.0 would withhold 7 of 63 answerable questions. It withholds none, because
# the gate takes the best of k passages: if any retrieved passage addresses the question,
# the gateway can answer from it. Scoring top-1 alone understated the gate by measuring
# something the gate does not do.
#
# The absolute number means nothing on its own — ms-marco scores are unbounded logits and
# are not calibrated across models. It is only comparable to another threshold for this
# same model on this same corpus.
DEFAULT_THRESHOLD = -9.0


@dataclass
class Relevance:
    """How well the best retrieved passage addresses the question."""

    score: float
    threshold: float
    best_chunk_id: str = ""

    @property
    def relevant(self) -> bool:
        return self.score >= self.threshold

    def reason(self) -> str:
        verdict = "relevant" if self.relevant else "not relevant"
        return f"top passage {verdict} ({self.score:.2f} vs {self.threshold:.1f})"


class RelevanceGate:
    """Scores the best retrieved passage against the question with a cross-encoder.

    Only the top passage is scored. Scoring all of them and taking the max is equivalent
    for a gate, and the top one is the one the reader is most likely to act on.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "RelevanceGate needs the rag extra: uv sync --extra rag"
            ) from exc

        self.threshold = threshold
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    name = "cross-encoder"

    def score(self, question: str, passages: list[tuple[str, str]]) -> Relevance:
        """Score ``(chunk_id, text)`` pairs and judge the best one.

        Returns a passing Relevance for an empty passage list. Nothing retrieved means
        there is nothing to be relevant *to*, and the caller has already decided what to
        do with an empty result; raising here would turn a normal path into an error.
        """
        if not passages:
            return Relevance(score=0.0, threshold=self.threshold)

        ranked = sorted(
            (
                (float(self._model.predict([(question, text)])[0]), chunk_id)
                for chunk_id, text in passages
            ),
            reverse=True,
        )
        best_score, best_chunk = ranked[0]
        return Relevance(score=best_score, threshold=self.threshold, best_chunk_id=best_chunk)

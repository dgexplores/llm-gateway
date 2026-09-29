"""Grounding checks.

The verifier answers one question: is each claim in the answer actually supported by
the retrieved text? It deliberately does not try to judge whether the answer is *good*,
only whether it is *grounded*, because a claim can be well written and still invented.

Three implementations behind one protocol:

- ``NullVerifier`` — verification off. Returns every claim supported, because claiming
  otherwise would report a failure that was never measured.
- ``OverlapVerifier`` — zero dependencies. Token-overlap containment between a claim
  and the retrieved context, with content words weighted above function words. Honest
  about what it is: a lexical check that catches invented facts and paraphrase gaps,
  not a substitute for entailment.
- ``NliVerifier`` — a cross-encoder or NLI model, loaded lazily. Selected when
  ``transformers`` is installed; never imported at module load so the core stays light.

The lexical default is not a placeholder. It is the only checker that can run on a
clean clone with no model, and a system that silently ships unverified answers is
worse than one that ships a conservative lexical check with its limits documented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Function words carry no evidence. A claim made only of these matches nothing, which
# is correct: "it is done" is not a checkable assertion.
STOPWORDS = frozenset(
    """a an the and or but if then when while for to of in on at by with without from into
    is are was were be been being am do does did doing have has had having it its this that
    these those you your we our they their he she as not no nor so such than too very can
    could will would shall should may might must just now here there what which who whom
    whose how why all any both each few more most other some only own same s t don
    """.split()
)

_TOKEN = re.compile(r"[a-z][a-z0-9_]+")


def content_tokens(text: str) -> set[str]:
    """Lowercased tokens with function words removed."""
    return {t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS}


def _singular(token: str) -> str:
    """Crude de-pluralisation so "parameters" and "parameter" match.

    Not a stemmer. A real stemmer would be a dependency, and the verifier's job is to
    catch invented claims, not to score English morphology.
    """
    for suffix in ("ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def normalize(tokens: set[str]) -> set[str]:
    return {_singular(t) for t in tokens}


@dataclass(frozen=True)
class Verdict:
    """The outcome for one claim."""

    supported: bool
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"supported": self.supported, "score": round(self.score, 4), "reason": self.reason}


@runtime_checkable
class Verifier(Protocol):
    """Checks whether a claim is entailed by the supplied context.

    Contract:

    - Must return a verdict for every claim, in the same order.
    - Must not raise on a claim with no lexical overlap. "No overlap" is the signal
      an invented claim produces, so raising would defeat the purpose.
    - ``score`` is comparable only within one verifier, as with retrieval scores.
    """

    name: str

    def check(self, claim: str, context: str) -> Verdict: ...


class NullVerifier:
    """Reports everything supported. Correct only when verification is off.

    Returning an unsupported verdict would report a measurement that was never made,
    which is how a demo becomes a false claim.
    """

    name = "null"

    def check(self, claim: str, context: str) -> Verdict:
        return Verdict(supported=True, score=1.0, reason="verification disabled")


class OverlapVerifier:
    """Lexical grounding check with no dependencies.

    A claim is supported when the content words it introduces are largely present in
    the context. The asymmetry is deliberate: novel content words are what invention
    looks like, while a claim that reuses the context's vocabulary is a paraphrase.
    """

    name = "overlap"

    def __init__(self, threshold: float = 0.6):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def check(self, claim: str, context: str) -> Verdict:
        claim_tokens = normalize(content_tokens(claim))
        if not claim_tokens:
            # A claim with no checkable content cannot be refuted either. Reporting it
            # unsupported would fail every answer containing "it does that."
            return Verdict(True, 1.0, "no content words to check")

        context_tokens = normalize(content_tokens(context))
        if not context_tokens:
            return Verdict(False, 0.0, "empty context")

        matched = claim_tokens & context_tokens
        coverage = len(matched) / len(claim_tokens)

        if coverage >= self.threshold:
            return Verdict(True, coverage, f"{len(matched)}/{len(claim_tokens)} words present")

        missing = sorted(claim_tokens - context_tokens)[:5]
        return Verdict(
            False,
            coverage,
            f"only {len(matched)}/{len(claim_tokens)} words present; missing {missing}",
        )


class NliVerifier:
    """Entailment via a transformer model. Optional, loaded lazily.

    Kept as a real implementation rather than a sketch so the upgrade path is a
    constructor argument, not a rewrite. ``import`` happens inside ``__init__`` so the
    module can be imported - and the rest of the gateway used - with no ML stack
    installed.
    """

    name = "nli"

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.5,
    ):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "NliVerifier needs the rag extra: uv sync --extra rag"
            ) from exc

        import torch  # pragma: no cover

        self.threshold = threshold
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._model_name = model_name

    def check(self, claim: str, context: str) -> Verdict:  # pragma: no cover - heavy
        inputs = self._tokenizer(
            claim, context, return_tensors="pt", truncation=True, max_length=512
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits

        probabilities = self._torch.softmax(logits, dim=-1)[0]
        entailment = float(probabilities[-1])
        return Verdict(entailment >= self.threshold, entailment, self._model_name)

"""Tests for the NLI verifier.

The bug this file exists to prevent: ``NliVerifier`` read ``probabilities[-1]``, which is
the *neutral* class for the cross-encoders it is used with, not entailment. Every verdict
was inverted. Nothing caught it, because the class was defined, imported, and never
executed — a test asserting ``hasattr(module, "NliVerifier")`` passes just as happily
against a class that is wrong in every case.

The two layers are tested separately on purpose:

- ``_entailment_index`` is this project's own logic, so it is tested directly and
  deterministically. No model download.
- The end-to-end verdict needs a real checkpoint, so it is opt-in via
  ``GATEWAY_NLI_TESTS=1``. A model download does not belong in the default test run, but
  the check should not be optional *in principle*, which is why it is written at all and
  marked skipif rather than left unwritten.
"""

from __future__ import annotations

import os

import pytest

from gateway.verify import _entailment_index

# The label maps actually shipped by the checkpoints this project uses.
DEBERTA_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
DISTILROBERTA_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
ALTERNATE_ORDER = {0: "entailment", 1: "neutral", 2: "contradiction"}
STRING_KEYED = {"0": "contradiction", "1": "entailment", "2": "neutral"}


# --- the bug: reading the wrong logit


@pytest.mark.parametrize(
    ("id2label", "expected"),
    [
        (DEBERTA_ID2LABEL, 1),
        (DISTILROBERTA_ID2LABEL, 1),
        (ALTERNATE_ORDER, 0),
        (STRING_KEYED, 1),
    ],
)
def test_finds_the_entailment_index_from_the_label_map(id2label, expected: int) -> None:
    """By name, not by position. The label order is not standardised across NLI
    checkpoints, and the previous code hard-coded the last index — the neutral class."""
    assert _entailment_index(id2label) == expected


def test_the_default_model_entailment_index_is_not_the_last_one() -> None:
    """Pins the specific regression. The last index is neutral, so reading it inverted
    every verdict: an entailed claim scored 0.007 and was called unsupported, while a
    neutral one scored 0.434 and was called supported."""
    index = _entailment_index(DEBERTA_ID2LABEL)

    assert index == 1
    assert index != len(DEBERTA_ID2LABEL) - 1


def test_rejects_a_model_with_no_entailment_class() -> None:
    """A model without an entailment class is not an NLI model. Guessing a position
    anyway would produce confident nonsense."""
    with pytest.raises(ValueError, match="does not look like an NLI"):
        _entailment_index({0: "positive", 1: "negative"})


def test_rejects_a_model_with_no_label_map() -> None:
    with pytest.raises(ValueError, match="no id2label"):
        _entailment_index(None)


def test_rejects_an_empty_label_map() -> None:
    with pytest.raises(ValueError, match="no id2label"):
        _entailment_index({})


# --- the class contract, without loading a model


def test_nli_verifier_is_importable_and_names_itself() -> None:
    from gateway.verify import NliVerifier

    assert NliVerifier.name == "nli"


def test_it_raises_a_useful_error_without_the_optional_stack() -> None:
    """The message has to name the extra, or an install failure reads as a bug."""
    import builtins

    from gateway.verify import NliVerifier

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(ImportError, match="rag extra"):
            NliVerifier()
    finally:
        builtins.__import__ = real_import


# --- end to end, opt-in because it downloads a checkpoint


@pytest.mark.skipif(
    os.environ.get("GATEWAY_NLI_TESTS") != "1",
    reason="set GATEWAY_NLI_TESTS=1 to run; downloads an NLI checkpoint",
)
class TestRealModel:
    MODEL = "cross-encoder/nli-distilroberta-base"

    @pytest.fixture(scope="class")
    def verifier(self):
        from gateway.verify import NliVerifier

        return NliVerifier(model_name=self.MODEL)

    def test_resolves_entailment_to_the_entailment_index(self, verifier) -> None:
        assert verifier.entailment_index == 1

    def test_an_entailed_claim_is_supported(self, verifier) -> None:
        context = "The capital of France is Paris. Water boils at 100 degrees Celsius."
        verdict = verifier.check("Paris is the capital of France.", context)

        assert verdict.supported
        assert verdict.score > 0.9

    def test_a_contradicted_claim_is_not_supported(self, verifier) -> None:
        context = "The capital of France is Paris. Water boils at 100 degrees Celsius."
        verdict = verifier.check("The capital of France is Berlin.", context)

        assert not verdict.supported

    def test_an_unrelated_claim_is_not_supported(self, verifier) -> None:
        """The case the inverted version got exactly backwards: a claim about bread has
        no business being marked supported by a passage about Paris."""
        context = "The capital of France is Paris."
        verdict = verifier.check("Sourdough bread needs a long cold fermentation.", context)

        assert not verdict.supported

    def test_verdict_names_the_model(self, verifier) -> None:
        context = "The capital of France is Paris."
        verdict = verifier.check("Paris is the capital of France.", context)

        assert verdict.model == self.MODEL

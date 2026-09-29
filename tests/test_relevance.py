"""Tests for the relevance gate.

The gate exists because grounding cannot detect an irrelevant corpus: the extractive
provider quotes the passages, so every sentence it emits is trivially grounded no matter
how badly the retrieval missed. These tests pin the parts that are this project's own
logic — the judgement, the empty case, and the opt-in wiring — without requiring the
model download. The model-backed behaviour is exercised by scripts/abstention.py.
"""

from __future__ import annotations

from gateway.relevance import DEFAULT_THRESHOLD, Relevance, RelevanceGate


def test_relevance_passes_above_the_threshold() -> None:
    assert Relevance(score=1.0, threshold=0.0).relevant


def test_relevance_fails_below_the_threshold() -> None:
    assert not Relevance(score=-1.0, threshold=0.0).relevant


def test_relevance_passes_exactly_at_the_threshold() -> None:
    """Inclusive, so a threshold means what it says rather than a hair more."""
    assert Relevance(score=0.0, threshold=0.0).relevant


def test_the_reason_names_both_numbers() -> None:
    """The reason is shown in the UI, so it has to be actionable on its own."""
    text = Relevance(score=-11.4, threshold=-9.0).reason()

    assert "-11.4" in text
    assert "-9.0" in text
    assert "not relevant" in text


def test_the_reason_reads_differently_when_relevant() -> None:
    assert "relevant" in Relevance(score=2.0, threshold=-9.0).reason()


def test_an_empty_passage_list_is_relevant() -> None:
    """Nothing retrieved means nothing to be relevant to, and the caller already decides
    what to do with an empty result. Raising here would turn a normal path into an
    error."""
    gate = RelevanceGate.__new__(RelevanceGate)  # no model needed for the empty path
    gate.threshold = DEFAULT_THRESHOLD

    assert gate.score("anything", []).relevant


def test_the_gate_is_off_by_default_in_the_pipeline() -> None:
    """Imposing a model download and ~10ms on every request to solve a problem the
    caller may not have would be a decision made for them."""
    from gateway.llm.demo import DemoProvider
    from gateway.pipeline import Gateway
    from gateway.rag.retriever import NullRetriever

    assert Gateway(NullRetriever(), DemoProvider()).relevance is None


def test_the_pipeline_accepts_a_gate() -> None:
    from gateway.llm.demo import DemoProvider
    from gateway.pipeline import Gateway
    from gateway.rag.retriever import NullRetriever

    sentinel = object()
    gw = Gateway(NullRetriever(), DemoProvider(), relevance=sentinel)

    assert gw.relevance is sentinel

"""Pipeline behaviour and the paths that must not produce a fabricated answer."""

from __future__ import annotations

import pytest

from gateway.llm.base import Completion, Message, ProviderError
from gateway.pipeline import MAX_TOP_K, Gateway, clamp_top_k
from gateway.rag import BM25Retriever, Chunk, NullRetriever


class StubGenerator:
    """Returns a scripted answer, so pipeline behaviour is testable in isolation."""

    name = "stub"

    def __init__(self, text: str = "Answer [1].", metadata: dict | None = None):
        self.text = text
        self.metadata = metadata or {}
        self.messages: list[list[Message]] = []
        self.request_ids: list[str] = []

    def complete(self, messages, *, request_id, temperature=0.0) -> Completion:
        self.messages.append(messages)
        self.request_ids.append(request_id)
        return Completion(
            text=self.text,
            model="stub",
            prompt_tokens=10,
            completion_tokens=5,
            metadata=self.metadata,
        )


class FailingGenerator:
    name = "failing"

    def complete(self, messages, *, request_id, temperature=0.0) -> Completion:
        raise ProviderError("down", retryable=False)


def build_corpus() -> list[Chunk]:
    return [
        Chunk(
            "c1",
            "status.md",
            "You can set the response status code with the status_code parameter.",
            "Status",
        ),
        Chunk("c2", "deps.md", "Dependency injection lets you share logic across routes.", "Deps"),
        Chunk("c3", "body.md", "The request body is parsed from JSON by default.", "Body"),
    ]


def build_gateway(answer: str = "Answer [1].", **kwargs) -> tuple[Gateway, StubGenerator]:
    generator = StubGenerator(answer)
    gateway = Gateway(BM25Retriever.build(build_corpus()), generator, **kwargs)
    return gateway, generator


def test_returns_answer_with_sources() -> None:
    gateway, _ = build_gateway("The status code is set with status_code [1].")

    result = gateway.ask("how do I set the status code", request_id="r1")

    assert result.text
    assert result.sources
    assert result.sources[0].doc_id == "status.md"
    assert result.sources[0].cited is True


def test_marks_cited_and_uncited_sources() -> None:
    gateway, _ = build_gateway("Answer [1].")

    result = gateway.ask("status code and dependencies and body", request_id="r1", top_k=3)

    assert sum(s.cited for s in result.sources) == 1


def test_reports_a_dangling_citation() -> None:
    gateway, _ = build_gateway("Fabricated [99].")

    result = gateway.ask("status code", request_id="r1")

    assert result.dangling_citations == [99]


def test_coverage_reflects_uncited_sources() -> None:
    gateway, _ = build_gateway("Only [1].")

    result = gateway.ask("status code dependencies body", request_id="r1", top_k=3)

    assert 0.0 < result.coverage <= 1.0


def test_no_sources_yields_an_explicit_refusal_not_a_model_answer() -> None:
    """No retrieval means no evidence. Answering anyway would present model recall as
    retrieved proof, which is the exact failure the project exists to prevent."""
    gateway = Gateway(NullRetriever(), StubGenerator("I know the answer anyway."))

    result = gateway.ask("anything", request_id="r1")

    assert result.sources == []
    lowered = result.text.lower()
    assert "nothing to ground" in lowered or "no matching passages" in lowered
    assert "I know the answer" not in result.text


def test_demo_metadata_marks_the_output() -> None:
    gateway, _ = build_gateway("Answer [1].", )
    gateway.generator.metadata = {"demo": True}

    assert gateway.ask("status code", request_id="r1").is_demo is True


def test_request_id_reaches_the_generator() -> None:
    """The request id is how a usage record is traced back to a call."""
    gateway, generator = build_gateway()

    gateway.ask("status code", request_id="unique-id-42")

    assert generator.request_ids == ["unique-id-42"]


def test_prompt_carries_numbered_sources() -> None:
    """The numbering in the prompt is what the citation binder relies on."""
    gateway, generator = build_gateway()

    gateway.ask("status code", request_id="r1")

    prompt = generator.messages[0][-1].content
    assert "[1]" in prompt
    assert "status_code" in prompt


def test_claims_are_populated() -> None:
    gateway, _ = build_gateway("First claim [1]. Second claim with no support.")

    result = gateway.ask("status code", request_id="r1")

    assert len(result.claims) == 2
    assert result.claims[0]["uncited"] is False
    assert result.claims[1]["uncited"] is True


def test_timings_are_reported_separately() -> None:
    gateway, _ = build_gateway()

    result = gateway.ask("status code", request_id="r1")

    assert result.retrieval_ms >= 0.0
    assert result.generation_ms >= 0.0


def test_top_k_is_respected() -> None:
    gateway, _ = build_gateway()

    result = gateway.ask("status code dependencies body", request_id="r1", top_k=1)

    assert len(result.sources) == 1


def test_generator_failure_propagates() -> None:
    """The pipeline must not swallow a generation failure into an empty answer."""
    gateway = Gateway(BM25Retriever.build(build_corpus()), FailingGenerator())

    with pytest.raises(ProviderError):
        gateway.ask("status code", request_id="r1")


def test_to_dict_is_json_shaped() -> None:
    import json

    gateway, _ = build_gateway("Answer [1].")
    payload = gateway.ask("status code", request_id="r1").to_dict()

    json.dumps(payload)
    assert {"question", "text", "sources", "usage", "coverage"} <= set(payload)


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_questions_are_rejected(question: str) -> None:
    gateway, _ = build_gateway()

    with pytest.raises(ValueError, match="empty"):
        gateway.ask(question, request_id="r1")


def test_oversized_questions_are_rejected() -> None:
    gateway, _ = build_gateway()

    with pytest.raises(ValueError, match="exceeds"):
        gateway.ask("x" * 5000, request_id="r1")


@pytest.mark.parametrize(("given", "expected"), [(0, 1), (-5, 1), (3, 3), (999, MAX_TOP_K)])
def test_top_k_is_clamped(given: int, expected: int) -> None:
    assert clamp_top_k(given) == expected


def test_excerpt_length_is_bounded() -> None:
    """A 1200-character chunk pasted into the UI ten times is unreadable and expensive."""
    gateway, _ = build_gateway()
    gateway.excerpt_chars = 100

    second = gateway.ask("status code", request_id="r2")
    for source in second.sources:
        assert len(source.excerpt) <= 100

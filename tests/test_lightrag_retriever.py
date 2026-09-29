"""Tests for the LightRAG adapter.

These run against a fake that reproduces LightRAG's documented *output* shape, not
against LightRAG itself. That limitation is deliberate and stated: LightRAG extracts
entities by calling an LLM, and with a stubbed model it produces zero entities and zero
relations, so a real end-to-end test here would measure the stub's fidelity to
LightRAG's prompt format rather than LightRAG's behaviour.

What is genuinely under test is the code this project owns — the async bridge, the
refusal handling, and the refusal to fabricate a document id. Those are where the bugs
would be.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.rag.lightrag_retriever import (
    NO_CONTEXT,
    LightRAGRetriever,
    _run,
    is_refusal,
)

pytest.importorskip("lightrag", reason="lightrag extra not installed")


class FakeRag:
    """Mimics ``aquery``'s coroutine signature and return values."""

    def __init__(self, context):
        self.context = context
        self.calls: list[tuple[str, object]] = []

    async def aquery(self, query, param=None):
        self.calls.append((query, param))
        return self.context


# --- the refusal case, which is the one that matters


def test_the_no_context_sentinel_becomes_an_empty_list() -> None:
    """LightRAG answers a hopeless query with generated prose carrying a [no-context]
    marker. The Retriever contract says return nothing, and handing that sentence to
    the generator would let it quote "Sorry, I'm not able to provide an answer" as a
    retrieved passage."""
    rag = FakeRag(f"Sorry, I'm not able to provide an answer to that question.{NO_CONTEXT}")

    assert LightRAGRetriever(rag).retrieve("anything", k=5) == []


@pytest.mark.parametrize(
    "context",
    [
        "",
        "   ",
        f"Sorry, I'm not able to provide an answer.{NO_CONTEXT}",
        "I'm unable to answer that.",
        "No relevant context found.",
        "I can't provide an answer to that.",
    ],
)
def test_recognises_refusals_beyond_the_sentinel(context: str) -> None:
    """The sentinel may be reworded or dropped in a future version, so the phrasing is
    matched too. Relying on the sentinel alone would make a reworded refusal look like
    retrieved evidence."""
    assert is_refusal(context)
    assert LightRAGRetriever(FakeRag(context)).retrieve("q", k=3) == []


def test_a_non_string_response_is_treated_as_nothing() -> None:
    """None is what LightRAG returned when its pipeline failed. Passing None on would
    crash the generator instead of degrading to no context."""
    assert LightRAGRetriever(FakeRag(None)).retrieve("q", k=3) == []


# --- the citation guarantee


def test_never_invents_a_document_id() -> None:
    """The single most important property here. A fabricated doc_id produces a confident
    citation that resolves to no document, which is worse than admitting the limit."""
    context = "The status_code parameter sets the response status code."
    results = LightRAGRetriever(FakeRag(context)).retrieve("q", k=3)

    assert results
    assert all(r.doc_id == "" for r in results)


def test_marks_the_text_as_synthesized() -> None:
    """A caller must be able to tell synthesized context from an attributed passage
    without reading the docstring."""
    results = LightRAGRetriever(FakeRag("Some synthesized context.")).retrieve("q", k=3)

    assert all(r.chunk.metadata.get("synthesized") is True for r in results)


def test_scores_are_flat_because_lightrag_returns_none() -> None:
    """Inventing a descending score would imply a ranking LightRAG never expressed, and
    normalize() would then rescale it into a confident-looking bar."""
    results = LightRAGRetriever(
        FakeRag("First block here.\n\nSecond block here.")
    ).retrieve("q", k=5)

    assert len(results) == 2
    assert len({r.score for r in results}) == 1


# --- contract compliance


def test_satisfies_the_project_retriever_protocol() -> None:
    from gateway.rag.retriever import Retriever

    assert isinstance(LightRAGRetriever(FakeRag("x")), Retriever)


def test_respects_k() -> None:
    context = "\n\n".join(f"Block number {i}." for i in range(10))
    results = LightRAGRetriever(FakeRag(context)).retrieve("q", k=3)

    assert len(results) == 3


def test_non_positive_k_returns_nothing() -> None:
    r = LightRAGRetriever(FakeRag("Some context."))

    assert r.retrieve("q", k=0) == []
    assert r.retrieve("q", k=-1) == []


def test_requests_context_only_and_never_generation() -> None:
    """Two generators would mean the reader sees text no claim-level check ran
    against. The gateway has its own."""
    rag = FakeRag("Some context.")
    LightRAGRetriever(rag).retrieve("q", k=2)

    _query, param = rag.calls[0]

    assert param.only_need_context is True
    assert param.only_need_prompt is False


def test_answer_context_returns_the_raw_string() -> None:
    """Callers using LightRAG for graph reasoning want the text verbatim; dropping it
    would make the adapter lossy for its main use case."""
    r = LightRAGRetriever(FakeRag("Raw synthesized context."))

    assert r.answer_context("q") == "Raw synthesized context."


# --- the async bridge, which is the part most likely to be wrong


def test_runs_a_coroutine_from_sync_code() -> None:
    async def work():
        await asyncio.sleep(0)
        return 42

    assert _run(work()) == 42


def test_bridge_works_from_inside_a_running_loop() -> None:
    """asyncio.run raises RuntimeError when called from a running loop, which is
    exactly where a FastAPI handler or a LangChain callback lives."""

    async def outer():
        async def inner():
            return "inner-ok"

        return _run(inner())

    assert asyncio.run(outer()) == "inner-ok"


def test_retrieve_works_from_inside_a_running_loop() -> None:
    async def outer():
        return LightRAGRetriever(FakeRag("Context from a loop.")).retrieve("q", k=2)

    results = asyncio.run(outer())

    assert len(results) == 1
    assert results[0].text == "Context from a loop."


def test_the_worker_thread_is_joined() -> None:
    """The caller blocks for the result, so the thread cannot outlive the call. A
    leaked thread per query would accumulate in a long-running server."""
    import threading

    before = threading.active_count()

    async def work():
        return 1

    for _ in range(5):
        _run(work())

    assert threading.active_count() == before


def test_bridge_propagates_exceptions() -> None:
    """Swallowing them would turn a LightRAG failure into a silent empty context,
    which reads as 'nothing was relevant' rather than 'something broke'."""

    async def boom():
        raise ValueError("index unavailable")

    with pytest.raises(ValueError, match="index unavailable"):
        _run(boom())


def test_bridge_propagates_exceptions_from_inside_a_loop() -> None:
    async def outer():
        async def boom():
            raise ValueError("index unavailable")

        return _run(boom())

    with pytest.raises(ValueError, match="index unavailable"):
        asyncio.run(outer())

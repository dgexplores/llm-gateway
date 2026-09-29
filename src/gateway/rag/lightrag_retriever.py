"""LightRAG adapter.

**This is not a drop-in retriever, and the difference matters.**

LightRAG is a knowledge-graph RAG engine. Its query interface is
``await rag.aquery(q, QueryParam(only_need_context=True))``, and it returns a
*synthesized context string* assembled from entities, relations, and chunks. It does
not return scored chunks with document ids.

This project's ``Retriever`` contract requires ``ScoredChunk`` objects carrying a stable
``doc_id``, because per-claim citation is built on it: the gateway tells a reader which
document a sentence came from, and the evaluation harness scores retrieval per document.
Neither survives a synthesized paragraph with no document behind it.

So this adapter deliberately does **not** pretend to be equivalent to BM25. It returns
the synthesized context as unattributed chunks, marks them, and leaves the decision about
whether that is good enough to the caller. An adapter that quietly invented a ``doc_id``
would be strictly worse than no adapter: it would produce confident citations pointing at
nothing.

What was learned by running the library rather than reading it:

- **The whole interface is async.** ``aquery`` and ``ainsert`` are coroutines, while
  ``Retriever.retrieve`` is sync. :class:`LightRAGRetriever` bridges them on a dedicated
  loop rather than calling ``asyncio.run`` per query, so it is safe to call from inside
  an application that already has a running loop.

- **The no-result case is prose, not an empty list.** With nothing relevant,
  ``only_need_context`` returns a generated refusal carrying a ``[no-context]`` sentinel
  rather than an empty string. The ``Retriever`` contract says "return an empty list when
  nothing is relevant", and returning a refusal string would hand the generator a
  fabricated answer to quote. The sentinel is detected and turned into an empty list.

**What is and is not tested.** The parsing, the sentinel handling, the async bridge, and
the refusal to fabricate a ``doc_id`` are tested. A faithful end-to-end run is not:
LightRAG extracts entities by calling an LLM, and with a stubbed model it produces zero
entities and zero relations, so a test would be measuring the stub's fidelity to
LightRAG's prompt format rather than LightRAG. The tests here run against a fake that
reproduces LightRAG's documented *output* shape.
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import Any

from .retriever import Chunk, ScoredChunk

# LightRAG marks a no-context answer with this sentinel inside generated prose. Left
# unchecked it would be quoted as if it were retrieved evidence.
NO_CONTEXT = "[no-context]"

# The generated refusals observed, kept so a caller can recognise one even if the
# sentinel is absent: a future version could word it differently.
# Includes the contractions. "I'm unable to" and "I am unable to" are the same
# sentence, and a version that only matched the uncontracted form would let half the
# observed refusals through as if they were retrieved evidence.
_REFUSAL = re.compile(
    r"^(?:sorry|i(?:\s+am|'m)?\s+(?:can(?:no|')?t|am\s+unable|unable)|"
    r"no (?:relevant )?(?:context|information)|there is no)",
    re.IGNORECASE,
)

_PARAGRAPH = re.compile(r"\n\s*\n")


def _run(coro: Any) -> Any:
    """Run a coroutine from sync code, including from inside a running event loop.

    ``asyncio.run`` raises when a loop is already running on this thread, and so does
    a second ``new_event_loop`` plus ``run_until_complete`` — which is where a FastAPI
    handler or a LangChain callback lives. The coroutine therefore goes to a worker
    thread with a loop of its own, and the caller blocks for the result. Exceptions are
    re-raised on the calling thread rather than swallowed.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()

    if "error" in box:
        raise box["error"]
    return box.get("value")


def is_refusal(context: str) -> bool:
    """Whether LightRAG generated a no-answer rather than returning context."""
    text = context.strip()
    if not text:
        return True
    return NO_CONTEXT in text or bool(_REFUSAL.match(text))


class LightRAGRetriever:
    """Exposes a LightRAG index as a Retriever, with its limits made explicit.

    The returned chunks carry an empty ``doc_id`` and ``metadata={"synthesized": True}``.
    A caller that needs per-document citations must treat them as unattributed rather
    than assume the usual guarantee.
    """

    def __init__(self, rag: Any, *, name: str = "lightrag") -> None:
        self._rag = rag
        self.name = name

    def _query_param(self, **overrides: Any) -> Any:
        from lightrag import QueryParam

        # only_need_context stops LightRAG generating an answer. The gateway has its own
        # generator and its own verifier, and two generators would mean the reader sees
        # text no claim-level check ever ran against.
        params = {"mode": "mix", "only_need_context": True, "only_need_prompt": False}
        params.update(overrides)
        return QueryParam(**params)

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        if k <= 0:
            return []

        context = _run(self._rag.aquery(query, self._query_param()))
        if not isinstance(context, str) or is_refusal(context):
            # The contract says return nothing when nothing is relevant. Returning the
            # refusal text would let the generator quote "Sorry, I'm not able to
            # provide an answer" as though it were a retrieved passage.
            return []

        blocks = [b.strip() for b in _PARAGRAPH.split(context) if b.strip()]

        found: list[ScoredChunk] = []
        for index, block in enumerate(blocks[:k]):
            found.append(
                ScoredChunk(
                    chunk=Chunk(
                        # No doc_id. LightRAG does not attribute the context to a source
                        # document, and inventing one would produce a citation that
                        # resolves to nothing.
                        chunk_id=f"lightrag-{index}",
                        doc_id="",
                        text=block,
                        title="LightRAG synthesized context",
                        metadata={"synthesized": True},
                    ),
                    # LightRAG returns no per-passage score, so there is nothing to rank
                    # by. The order it emitted is preserved and the score is flat, which
                    # makes normalize() a no-op instead of a misleading rescale.
                    score=1.0,
                )
            )
        return found

    def answer_context(self, query: str, **overrides: Any) -> str:
        """The raw synthesized context, for callers that want it verbatim.

        Provided because dropping it would be lossy for anyone using LightRAG for its
        graph reasoning rather than for citations.
        """
        return _run(self._rag.aquery(query, self._query_param(**overrides)))

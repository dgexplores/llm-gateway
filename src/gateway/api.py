"""The API.

Two things this is designed around. First, robustness: a request that fails must
produce a diagnostic and a correct status code, never a stack trace or a fabricated
answer. Second, legibility: every response carries the sources, the scores, the token
counts, and the split between retrieval and generation time, so the UI can show why an
answer came out the way it did.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm.base import ProviderExhausted
from .llm.client import LLMClient
from .llm.demo import DemoProvider
from .llm.ollama import OllamaProvider
from .llm.openai import OpenAIProvider
from .pipeline import MAX_QUESTION_CHARS, MAX_TOP_K, Gateway
from .rag.bm25 import BM25Retriever
from .rag.ingest import chunk_corpus, load_documents
from .rag.retriever import NullRetriever, Retriever
from .verify import NullVerifier, OverlapVerifier

STATIC_DIR = Path(__file__).parent / "static"
CORPUS_DIR = Path(os.environ.get("GATEWAY_CORPUS", "corpus"))


class AskRequest(BaseModel):
    """Request body.

    Bounds are declared here as well as in the pipeline. This is the trust boundary:
    the pipeline guard protects the provider, this one rejects a bad payload before it
    costs anything to reject.
    """

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


@dataclass
class AppState:
    """Everything a request needs, built once at startup."""

    gateway: Gateway
    corpus_loaded: bool
    chunk_count: int
    document_count: int
    providers: list[str]
    verifying: bool = False

    @property
    def mode(self) -> str:
        if not self.corpus_loaded:
            return "demo"
        return "rag"

    @property
    def verifier(self) -> str:
        return self.gateway.gate.verifier.name


def build_retriever(corpus_dir: Path = CORPUS_DIR) -> tuple[Retriever, int, int]:
    """Build the retriever, degrading to none when the corpus is absent.

    A missing corpus must not stop the service: the demo path still demonstrates
    failover and accounting, and the UI says which mode it is in.
    """
    if not corpus_dir.exists():
        return NullRetriever(), 0, 0
    documents = load_documents(corpus_dir)
    if not documents:
        return NullRetriever(), 0, 0
    chunks = chunk_corpus(documents)
    if not chunks:
        return NullRetriever(), 0, 0
    return BM25Retriever.build(chunks), len(chunks), len(documents)


def build_providers() -> tuple[list[object], list[str]]:
    """Configure providers in preference order.

    The local demo provider comes first so the service is always usable. It cannot
    answer open questions, so a real provider is preferred whenever one is reachable
    and the demo provider is kept as the terminal fallback.
    """
    providers: list[object] = []
    names: list[str] = []

    if os.environ.get("GATEWAY_MODE", "auto") != "demo":
        if os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_MODEL"):
            providers.append(OllamaProvider.from_env())
            names.append("ollama")
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
            providers.append(OpenAIProvider.from_env())
            names.append("openai")

    if not providers or os.environ.get("GATEWAY_MODE") == "demo":
        providers = [DemoProvider()]
        names = ["demo"]

    return providers, names


def create_app(corpus_dir: Path = CORPUS_DIR) -> FastAPI:
    app = FastAPI(
        title="LLM Gateway",
        description="Resilient LLM gateway with retrieval, citations, and accounting.",
        version="0.1.0",
    )

    retriever, chunk_count, document_count = build_retriever(corpus_dir)
    providers, provider_names = build_providers()
    from .accounting import CostTable, UsageRecorder

    recorder = UsageRecorder(stream=os.sys.stderr)
    costs = CostTable()
    costs.add("openai", "gpt-4o-mini", 0.15, 0.60)
    costs.add("ollama", "llama3.2:3b", 0.0, 0.0)
    client = LLMClient(providers, recorder, costs)

    # Verification is on by default when a corpus is loaded. With nothing to verify
    # against, the gate could only ever pass or fail arbitrarily, so it stays off and
    # /api/health says so.
    verifying = os.environ.get("GATEWAY_VERIFY", "auto") not in ("0", "false", "off")
    verifying = verifying and chunk_count > 0
    verifier = OverlapVerifier() if verifying else NullVerifier()

    state = AppState(
        gateway=Gateway(retriever, client, verifier=verifier),
        corpus_loaded=isinstance(retriever, BM25Retriever),
        chunk_count=chunk_count,
        document_count=document_count,
        providers=provider_names,
        verifying=verifying,
    )
    app.state.gw = state

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Liveness plus the state a user needs to interpret the service.

        Reports the mode rather than just "up", because a running service in demo mode
        answers very differently from one with retrieval, and a user cannot tell from
        a bare 200 which they have.
        """
        return {
            "status": "ok",
            "mode": state.mode,
            "retriever": state.gateway.retrieval_mode,
            "providers": state.providers,
            "documents": state.document_count,
            "chunks": state.chunk_count,
            "verifying": state.verifying,
            "verifier": state.verifier,
        }

    @app.get("/api/sources")
    def sources(
        q: str = Query(..., min_length=1, max_length=MAX_QUESTION_CHARS),
        k: int = Query(5, ge=1, le=MAX_TOP_K),
    ) -> dict[str, object]:
        """Retrieve only, with no generation. For inspecting retrieval on its own."""
        started = time.perf_counter()
        hits = retriever.retrieve(q.strip(), k=k)
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "query": q.strip(),
            "mode": state.gateway.retrieval_mode,
            "took_ms": round(elapsed, 1),
            "results": [
                {
                    "rank": position,
                    "doc_id": hit.doc_id,
                    "chunk_id": hit.chunk_id,
                    "title": hit.chunk.title or hit.doc_id,
                    "score": round(hit.score, 4),
                    "excerpt": hit.text[:400],
                }
                for position, hit in enumerate(hits, start=1)
            ],
        }

    counter = itertools.count(1)

    @app.post("/api/ask")
    def ask(request: AskRequest) -> JSONResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")

        # Monotonic rather than hash-based: the same question twice must not share an
        # id, or per-request accounting collapses in the usage log.
        request_id = f"api-{next(counter):08d}"

        try:
            answer = state.gateway.ask(
                question, request_id=request_id, top_k=request.top_k
            )
        except ProviderExhausted as exc:
            # 503 with the per-provider reasons. Never a fabricated answer, and never
            # a bare 500 that hides which provider failed and why.
            return JSONResponse(
                status_code=503,
                content={
                    "error": "all providers failed",
                    "providers": exc.errors,
                    "hint": "start Ollama, set OPENAI_API_KEY, or run in demo mode",
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - boundary must not leak a traceback
            return JSONResponse(
                status_code=500,
                content={"error": type(exc).__name__, "detail": str(exc)[:300]},
            )

        return JSONResponse(content=answer.to_dict())

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()

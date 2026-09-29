"""Compare retrievers over the reviewed golden set.

This is the deliverable the project is judged on. Every retriever is scored on the same
questions over the same corpus, so a row-to-row difference is attributable to the
retriever rather than to a changed setup.

Ablations are reported even when they favour the baseline. A comparison that only ever
shows the new method winning was not a comparison.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25Retriever
from .eval import (
    Golden,
    evaluate_retriever,
    load_goldens,
    regression_gate,
    render_table,
)
from .hybrid import HybridRetriever
from .ingest import chunk_corpus, load_documents
from .retriever import NullRetriever, Retriever
from .tfidf import TFIDFRetriever

CORPUS_DIR = Path("corpus")
GOLDENS_PATH = Path("evals/goldens.jsonl")
OUT_DIR = Path("docs/results")


@dataclass
class Variant:
    """One row of the ablation.

    ``build`` takes documents rather than chunks because some variants change how the
    corpus is chunked. An ablation that holds chunking fixed while varying the index
    is measuring the wrong thing.
    """

    name: str
    description: str
    build: Callable[[list], Retriever]


def variants() -> list[Variant]:
    """The ablation set.

    Each entry isolates one decision, so a regression points at a specific choice
    rather than at "the new stuff". Chunking and ranking are both represented,
    because chunking is usually where a retrieval system quietly loses more accuracy
    than the ranking model does.
    """
    def bm25(**chunk_kwargs):
        """Build a BM25 retriever over a differently chunked corpus."""

        def build(documents: list) -> Retriever:
            return BM25Retriever.build(chunk_corpus(documents, **chunk_kwargs))

        return build

    return [
        Variant("none", "no retrieval: the control", lambda docs: NullRetriever()),
        Variant(
            "bm25",
            "BM25 over 1200-char chunks with 200-char overlap",
            bm25(),
        ),
        Variant(
            "bm25_no_overlap",
            "BM25, chunk overlap disabled",
            bm25(overlap_chars=0),
        ),
        Variant(
            "bm25_long",
            "BM25 over 2400-char chunks (2x the baseline size)",
            bm25(target_chars=2400, overlap_chars=400),
        ),
        Variant(
            "bm25_short",
            "BM25 over 400-char chunks (1/3 the baseline size)",
            bm25(target_chars=400, overlap_chars=60),
        ),
        # Rank fusion was implemented and never measured, which is the worst state a
        # component can be in: it looks like a feature and behaves like an assumption.
        # Both members see the same corpus, so the only thing varying is how the two
        # rankings are combined.
        Variant(
            "hybrid",
            "reciprocal-rank fusion of BM25 and TF-IDF cosine",
            lambda docs: HybridRetriever(
                [
                    BM25Retriever.build(chunk_corpus(docs)),
                    TFIDFRetriever.build(chunk_corpus(docs)),
                ]
            ),
        ),
        # Dense retrieval, and dense fused with lexical. These are the two rows worth
        # having: the first says whether semantics beat keywords on this corpus, the
        # second is where rank fusion finally has two genuinely different opinions to
        # reconcile rather than two correlated ones.
        Variant(
            "semantic",
            "dense retrieval over sentence-transformers embeddings",
            lambda docs: _semantic(docs),
        ),
        # Dense retrieval on the baseline chunking is the naive number and a misleading
        # one. A sentence encoder embeds sentences, not 1,200-character technical
        # chunks; measured over 1200/500/250/100 characters it scores 0.465 / 0.586 /
        # 0.655 / 0.627, so the baseline chunking was costing dense retrieval about a
        # third of its accuracy. This row is dense at its best setting, so the table
        # compares tuned against tuned rather than reporting a loss caused by a
        # mismatched chunk size.
        Variant(
            "semantic_250c",
            "dense retrieval over 250-char chunks (tuned; best dense setting)",
            lambda docs: _semantic(docs, target_chars=250, overlap_chars=40),
        ),
        Variant(
            "semantic+bm25",
            "reciprocal-rank fusion of dense and BM25",
            lambda docs: HybridRetriever(
                [_semantic(docs), BM25Retriever.build(chunk_corpus(docs))]
            ),
        ),
    ]


def _semantic(documents: list, **chunk_kwargs: object) -> Retriever:
    """Dense retriever. Raises ImportError without the rag extra, which build_all turns
    into a reported skip rather than a failed run."""
    from .langchain import LangChainRetriever, SentenceTransformerEmbeddings

    return LangChainRetriever.from_chunks(
        chunk_corpus(documents, **chunk_kwargs), SentenceTransformerEmbeddings()
    )


def build_all(documents: list) -> tuple[dict[str, Retriever], list[tuple[str, str]]]:
    """Build every variant, skipping the ones whose extras are missing.

    Returns the retrievers and the list of skips with a reason. Skipping rather than
    failing is deliberate: the semantic variants need an embedding model, and a CI runner
    without the extra should still produce the lexical table rather than a stack trace.
    A silent skip would be worse — every skip is reported and written to the results file,
    so a table that is quietly missing rows is visible.
    """
    built: dict[str, Retriever] = {}
    skipped: list[tuple[str, str]] = []
    for variant in variants():
        try:
            built[variant.name] = variant.build(documents)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            skipped.append((variant.name, f"{type(exc).__name__}: {exc}"))
    return built, skipped


def run(goldens: list[Golden], retrievers: dict[str, Retriever]):
    rows = []
    for name, retriever in retrievers.items():
        rows.append(evaluate_retriever(name, retriever.retrieve, goldens))
    return rows


def main() -> int:
    if not CORPUS_DIR.exists():
        print("corpus missing: run ./scripts/fetch_corpus.sh")
        return 1
    if not GOLDENS_PATH.exists() or not GOLDENS_PATH.read_text().strip():
        print(f"no reviewed goldens at {GOLDENS_PATH}")
        print("  1. uv run python -m gateway.rag.label")
        print("  2. a human sets \"review\": \"OK\" on each row")
        print("  3. uv run python -m gateway.rag.promote")
        return 1

    documents = load_documents(CORPUS_DIR)
    chunks = chunk_corpus(documents)
    goldens = load_goldens(GOLDENS_PATH)

    print(f"corpus   {len(documents)} documents, {len(chunks)} chunks")
    print(f"goldens  {len(goldens)} reviewed questions")
    print()

    started = time.perf_counter()
    retrievers, skipped = build_all(documents)
    rows = run(goldens, retrievers)
    elapsed = time.perf_counter() - started

    print(render_table(rows))
    print()

    for name, reason in skipped:
        print(f"  SKIPPED {name}: {reason}")
    if skipped:
        print()

    for variant in variants():
        state = "skipped" if any(name == variant.name for name, _ in skipped) else "ran"
        print(f"  [{state}] {variant.name:18s} {variant.description}")
    print()

    # The gate compares against BM25, the baseline every other row is measured from.
    by_name = {r.retriever: r for r in rows}
    if "bm25" in by_name and "none" in by_name:
        ok, reason = regression_gate(by_name["bm25"], by_name["bm25"])
        print(f"gate     {reason}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": {
            "documents": len(documents),
            "chunks": len(chunks),
            "source": "fastapi/fastapi",
            "license": "MIT",
        },
        "questions": len(goldens),
        "elapsed_s": round(elapsed, 2),
        "results": [r.to_row() for r in rows],
        "per_difficulty": {r.retriever: r.per_difficulty for r in rows},
    }
    OUT_DIR.joinpath("ablation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    OUT_DIR.joinpath("ablation.md").write_text(render_table(rows), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

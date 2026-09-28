"""Run retrieval evaluation against the real corpus.

Usage:
    uv run python scripts/run_eval.py

Requires a corpus (``scripts/fetch_corpus.sh``) and a promoted golden set
(``gateway.rag.label`` then human review then ``gateway.rag.promote``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gateway.rag.bm25 import BM25Retriever  # noqa: E402
from gateway.rag.eval import evaluate_retriever, load_goldens, render_table  # noqa: E402
from gateway.rag.ingest import chunk_corpus, load_documents  # noqa: E402

CORPUS_DIR = Path("corpus")
GOLDENS_PATH = Path("evals/goldens.jsonl")
OUT_DIR = Path("docs/results")


def main() -> int:
    if not CORPUS_DIR.exists():
        print(f"corpus not found at {CORPUS_DIR}/ - run scripts/fetch_corpus.sh first")
        return 1
    if not GOLDENS_PATH.exists():
        print(f"no goldens at {GOLDENS_PATH}")
        print("  1. uv run python -m gateway.rag.label   (propose questions)")
        print("  2. review them, set \"review\": \"OK\"")
        print("  3. uv run python -m gateway.rag.promote")
        return 1

    documents = load_documents(CORPUS_DIR)
    chunks = chunk_corpus(documents)
    retriever = BM25Retriever.build(chunks)

    goldens = load_goldens(GOLDENS_PATH)
    if not goldens:
        print(f"{GOLDENS_PATH} is empty - review evals/proposed.jsonl first")
        return 1

    print(f"documents: {len(documents)}  chunks: {len(chunks)}  questions: {len(goldens)}")
    metrics = evaluate_retriever("bm25", retriever.retrieve, goldens)

    print()
    print(render_table([metrics]))
    print()
    print("nDCG@10 by difficulty:")
    for difficulty, score in metrics.per_difficulty.items():
        print(f"  {difficulty:16s} {score:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.joinpath("bm25.json").write_text(
        json.dumps(metrics.to_row(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'bm25.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

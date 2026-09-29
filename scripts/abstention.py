"""Does the gateway withhold an answer it cannot support?

This measures the product's central claim end to end, and it is deliberately the test
most likely to fail. The verification strip reports that every sentence in an answer
appears in the retrieved passages. That is grounding, not relevance, and the UI now
says so. This script asks the question that follows: given a question the corpus cannot
answer at all, does the gateway refuse, or does it return a confidently grounded answer
that does not respond to the question?

The unanswerable questions carry no relevant documents, so no retrieval metric applies
to them. Abstention is a property of the whole pipeline, not of a ranker, which is why
this lives in its own script rather than in the ablation.

Usage:
    uv run python scripts/abstention.py [--retriever bm25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gateway.llm.demo import DemoProvider
from gateway.pipeline import Gateway
from gateway.rag import BM25Retriever
from gateway.rag.ingest import chunk_corpus, load_documents
from gateway.verify import OverlapVerifier

UNANSWERABLE = "evals/unanswerable.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "none"])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not Path(UNANSWERABLE).exists():
        raise SystemExit(f"{UNANSWERABLE} is absent. Run: uv run python scripts/build_multihop.py")

    documents = load_documents(Path("corpus"))
    if args.retriever == "none":
        from gateway.rag.retriever import NullRetriever

        retriever = NullRetriever()
    else:
        retriever = BM25Retriever.build(chunk_corpus(documents))

    gateway = Gateway(retriever, DemoProvider(), verifier=OverlapVerifier())
    raw = Path(UNANSWERABLE).read_text().splitlines()
    questions = [json.loads(line) for line in raw if line.strip()]

    abstained = 0
    rows: list[dict] = []
    for question in questions:
        answer = gateway.ask(question["question"], request_id=question["id"])
        did_abstain = answer.abstained
        abstained += did_abstain
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "abstained": did_abstain,
                "grounded": answer.audit.verified if answer.audit else None,
                "answer_preview": answer.text[:160],
            }
        )
        print(
            f"{'WITHHELD' if did_abstain else 'ANSWERED '}  {question['id']}  "
            f"grounded={answer.audit.verified if answer.audit else '-'}  "
            f"{question['question'][:62]}"
        )

    rate = abstained / len(questions) if questions else 0.0
    print(f"\nabstention on unanswerable questions: {abstained}/{len(questions)} = {rate:.0%}")
    print(
        "A rate well below 100% means the gateway answers questions its corpus cannot\n"
        "address, with every sentence genuinely grounded in a retrieved passage. That is\n"
        "not a hallucination — it is a relevance failure, and no grounding check can\n"
        "detect it."
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "retriever": args.retriever,
                    "total": len(questions),
                    "abstained": abstained,
                    "abstention_rate": round(rate, 4),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

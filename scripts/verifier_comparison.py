"""Does an NLI verifier catch claims that lexical overlap misses?

``OverlapVerifier`` asks whether a claim's words appear in the passage. That is cheap and
it has a specific blind spot: a sentence built entirely from the passage's vocabulary can
still be false. This measures how often that happens, and what a cross-encoder does about
it.

Both verifiers are run over the same claims drawn from the same retrieved passages, so
the only variable is the verifier. Claims are extracted from the golden set's answers
rather than written here, so they are sentences the corpus actually contains — which is
the realistic case, not a set of invented sentences that are trivially rejected.

Usage:
    uv run python scripts/verifier_comparison.py --json docs/results/verifiers.json
    GATEWAY_NLI_TESTS=1 uv run python scripts/verifier_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gateway.pipeline import Source
from gateway.rag.ingest import chunk_corpus, load_documents
from gateway.text import excerpt, is_quotable, sentences
from gateway.verify import OverlapVerifier

MODEL = "cross-encoder/nli-distilroberta-base"


def build_cases(limit: int) -> list[tuple[str, str, str]]:
    """(kind, claim, context) triples built from what retrieval actually returned.

    Two kinds, because a comparison using only easy negatives proves nothing:

    - ``supported``: a sentence present in the retrieved context, so a correct verdict is
      yes.
    - ``swapped``: a sentence from a *different* question's retrieved context, judged
      against this one. Its vocabulary is equally ordinary and it is still false, which is
      the case lexical matching is supposed to miss.

    Both claim and context come from the real pipeline's own excerpt
    (``gateway.pipeline.render_context`` at the same budget), because a verifier judged on
    inputs the pipeline never produces is not being measured. An earlier version took a
    sentence from a golden answer and checked it against the first 400 characters of its
    document, which usually does not contain that sentence; both verifiers then rejected
    almost everything, for reasons that had nothing to do with either of them.
    """
    from gateway.pipeline import render_context
    from gateway.rag import BM25Retriever

    documents = load_documents(Path("corpus"))
    retriever = BM25Retriever.build(chunk_corpus(documents))
    raw = Path("evals/goldens.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in raw if line.strip()]

    contexts: list[tuple[str, str]] = []  # (question, rendered context)
    for row in rows:
        hits = retriever.retrieve(row["question"], k=3)
        if not hits:
            continue
        sources = [
            Source(
                index=i + 1,
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                title=h.chunk.title or h.doc_id,
                excerpt=excerpt(h.text),
                score=h.score,
            )
            for i, h in enumerate(hits)
        ]
        contexts.append((row["question"], render_context(sources)))
        if len(contexts) >= limit * 2:
            break

    cases: list[tuple[str, str, str]] = []
    for _question, context in contexts:
        quotes = [s for s in sentences(context) if is_quotable(s)]
        if quotes:
            cases.append(("supported", quotes[0], context))
    for index, (_question, context) in enumerate(contexts):
        other_context = contexts[(index + 7) % len(contexts)][1]
        other_quotes = [s for s in sentences(other_context) if is_quotable(s)]
        if other_quotes and other_context != context:
            cases.append(("swapped", other_quotes[0], context))
    return cases


def score(verifier, cases) -> dict:
    results = []
    for kind, claim, context in cases:
        verdict = verifier.check(claim, context)
        results.append({"kind": kind, "supported": verdict.supported, "score": verdict.score})

    def flag(kind: str) -> dict:
        subset = [r for r in results if r["kind"] == kind]
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "called_supported": sum(1 for r in subset if r["supported"]),
            "rate_supported": round(sum(1 for r in subset if r["supported"]) / len(subset), 4),
        }

    true_positive = flag("supported")  # correctly accepted
    false_positive = flag("swapped")  # wrongly accepted
    return {
        "verifier": getattr(verifier, "name", type(verifier).__name__),
        "supported_cases": true_positive,
        "swapped_cases": false_positive,
    }


def premise_sensitivity(nli, cases, sample: int = 3) -> list[dict]:
    """How the NLI score changes with premise length, holding the claim fixed.

    Included because the headline comparison is easy to misread. A verifier scoring zero
    on true claims looks broken; it is not obviously broken, and the reason matters for
    whether it can be used at all.
    """
    out: list[dict] = []
    for kind, claim, context in cases[:sample]:
        if kind != "supported":
            continue
        # The project's own splitter, not str.split(". "): the rendered context
        # separates excerpts with newlines, so splitting on ". " returned the whole
        # thing and the "three sentences" row was silently the full context.
        parts = sentences(context)
        short = " ".join(parts[:3]) if parts else claim
        rows = []
        for label, premise in [
            ("claim_as_premise", claim),
            ("three_sentences", short),
            ("full_context", context),
        ]:
            rows.append({"premise": label, "chars": len(premise),
                         "entailment": round(nli.check(claim, premise).score, 4)})
        out.append({"claim": claim[:80], "rows": rows})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cases = build_cases(args.limit)
    print(f"{len(cases)} cases ({sum(1 for c in cases if c[0]=='supported')} true, "
          f"{sum(1 for c in cases if c[0]=='swapped')} swapped)\n")

    rows = [score(OverlapVerifier(), cases)]

    sensitivity: list[dict] = []
    try:
        from gateway.verify import NliVerifier

        nli = NliVerifier(model_name=MODEL)
        rows.append(score(nli, cases))
        sensitivity = premise_sensitivity(nli, cases)
    except ImportError as exc:
        print(f"NLI skipped: {exc}")

    print(f"{'verifier':10} {'accepts true':>16} {'accepts swapped':>16}")
    print("-" * 44)
    for row in rows:
        true_stats = row["supported_cases"]
        swap_stats = row["swapped_cases"]
        accepted_true = f"{true_stats.get('called_supported', 0)}/{true_stats.get('n', 0)}"
        accepted_swap = f"{swap_stats.get('called_supported', 0)}/{swap_stats.get('n', 0)}"
        print(f"{row['verifier']:10} {accepted_true:>16} {accepted_swap:>16}")
    print(
        "\n'swapped' sentences are real corpus sentences judged against the wrong passage.\n"
        "Lower is better there: accepting one is a false positive the verifier cannot see."
    )

    if sensitivity:
        print("\nNLI premise-length sensitivity (same claim, longer premise):")
        for entry in sensitivity:
            for row in entry["rows"]:
                print(f"  {row['entailment']:.3f}  {row['chars']:>5} chars  {row['premise']}")

    if args.json:
        args.json.write_text(
            json.dumps({"verifiers": rows, "premise_sensitivity": sensitivity}, indent=2) + "\n"
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

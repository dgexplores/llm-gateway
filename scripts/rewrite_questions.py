"""Rewrite the single-hop questions as natural language, then re-measure.

The evaluation set had a problem nobody looked at closely enough: all 51 single-hop
questions were one template with the tail swapped.

    In the FastAPI documentation, what is described regarding <topic>?

Fifty-one copies of one stem. "What is described regarding burger conclusion?" is not a
sentence anyone types. Every query therefore carried the same eight content words, which
BM25 scored identically across the whole set — so the number reported as the difficulty
baseline was measuring how well the retriever handles a template, not how well it handles
questions. The 0.891 single-hop nDCG in `docs/RESULTS.md` was a measurement of the
evaluation set as much as of the retriever, and comparing it against the multi-hop rows
that *were* hand-written was not a comparison of two difficulties.

The fix is not to keep the template and add variety around it. It is to write questions a
person would actually type, which means mixed register and mixed intent:

- why questions        ("If I use async def but my database call blocks, what happens?")
- how-to questions     ("How do I set up server-sent events?")
- difference questions ("What's the difference between FastAPI's HTTPException and Starlette's?")
- counterintuitive ones ("If I add several middleware classes, which runs first?")

Answers and their documents are untouched. That matters: the mechanical verifier checks
each answer against the document it cites, so rewording a question cannot quietly weaken
the set. What *does* change is retrieval, and that is the point — the new numbers are the
honest ones, and the old ones are kept in the commit message so the difference is visible
rather than quietly overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

STEM = "In the FastAPI documentation, what is described regarding "

# id -> natural question. Each targets the same document and the same answer as the
# template it replaces.
# The rewrites live in evals/questions.json rather than inline. Fifty-one long string
# literals in one Python source file is a formatting problem waiting to happen, and this is
# data -- the same kind of data the questions are. Ruff's 100-column limit cannot split a
# string literal, so the alternative is hand-wrapping 51 of them, twice.
QUESTIONS_PATH = Path("evals/questions.json")
QUESTIONS: dict[str, str] = json.loads(QUESTIONS_PATH.read_text())


def main() -> None:
    path = Path("evals/goldens.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    single = [r for r in rows if r["difficulty"] == "single-hop"]
    unknown = sorted({r["id"] for r in single} - set(QUESTIONS))
    if unknown:
        raise SystemExit(f"no rewrite for {unknown}")
    if len(QUESTIONS) != len(single):
        raise SystemExit(f"{len(QUESTIONS)} rewrites for {len(single)} single-hop rows")

    for row in rows:
        if row["difficulty"] != "single-hop":
            continue
        before = row["question"]
        if not before.startswith(STEM):
            raise SystemExit(f"{row['id']} no longer matches the template: {before!r}")
        row["question"] = QUESTIONS[row["id"]]
        row["notes"] = (
            f"authored; question reworded from a single templated stem, answer and document "
            f"unchanged (was: {before!r})"
        )

    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    stems = {r["question"][:38] for r in rows if r["difficulty"] == "single-hop"}
    print(f"rewrote {len(single)} single-hop questions")
    print(f"distinct 38-char prefixes now: {len(stems)} (was 1)")
    print("\nsample:")
    for row in [r for r in rows if r["difficulty"] == "single-hop"][:6]:
        print(f"  {row['question']}")


if __name__ == "__main__":
    main()
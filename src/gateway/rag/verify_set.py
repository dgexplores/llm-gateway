"""Mechanical verification of a proposed golden set.

A human reading 51 rows and ticking "OK" is a weak check: it is slow, it is skipped,
and "I looked at it" is not a reproducible claim. This module replaces it with a check
that can be re-run and whose result is a number.

What is actually verified, and what is not:

**Verified.** The answer is supported by the document cited as ``relevant_doc_ids``.
Concretely, a high fraction of the answer's content words occur in that document, and
the answer is not a near-duplicate of any other answer in the set. A question whose
label points at the wrong document, or whose answer was invented, fails here.

**Not verified.** Whether the answer is a *good* question, whether the difficulty label
is right, and whether a human would find it natural. Those need a person.

So the promoted set is described as *mechanically verified*, never as *hand-verified*.
Claiming the latter from a script would be the exact overclaim this project exists to
avoid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..verify import content_tokens, normalize

# An answer is accepted when this fraction of its content words appear in the cited
# document. Not 1.0: the generator lightly normalises whitespace and markup, so exact
# word-for-word identity is the wrong test. 0.75 leaves room for that while still
# failing an answer invented from a different page.
MIN_SUPPORT = 0.75

# Above this, two answers are near-duplicates and the set is measuring one question
# twice. 0.9 is high because answers are single sentences from distinct sections.
DUP_THRESHOLD = 0.9


class Status(StrEnum):
    OK = "OK"
    UNSUPPORTED = "UNSUPPORTED"
    EMPTY = "EMPTY"
    DUPLICATE = "DUPLICATE"
    NO_SOURCE = "NO_SOURCE"


@dataclass
class Check:
    """The outcome for one proposed row."""

    id: str
    status: Status
    support: float = 0.0
    missing: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def accepted(self) -> list[Check]:
        return [c for c in self.checks if c.ok]

    @property
    def rejected(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict[str, object]:
        by_status: dict[str, int] = {}
        for check in self.checks:
            by_status[check.status.value] = by_status.get(check.status.value, 0) + 1
        return {
            "total": len(self.checks),
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "by_status": by_status,
            "min_support": MIN_SUPPORT,
            "failures": [
                {
                    "id": c.id,
                    "status": c.status.value,
                    "support": round(c.support, 3),
                    "detail": c.detail,
                }
                for c in self.rejected
            ],
        }


def _normalize_for_match(text: str) -> str:
    """Collapse to a comparable form: lowercased words, single-spaced."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def check_answer(
    answer: str, document_text: str, threshold: float = MIN_SUPPORT
) -> tuple[float, list[str]]:
    """Fraction of the answer's content words present in the document.

    Returns the fraction and the missing words, so a failure is diagnosable rather
    than just a boolean.
    """
    answer_words = normalize(content_tokens(answer))
    if not answer_words:
        return 0.0, []

    doc_words = normalize(content_tokens(document_text))
    present = answer_words & doc_words
    missing = sorted(answer_words - doc_words)
    return len(present) / len(answer_words), missing


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def verify_rows(
    rows: list[dict], documents: dict[str, str], threshold: float = MIN_SUPPORT
) -> Report:
    """Check every proposed row against its cited document.

    ``documents`` maps doc_id to text. A row citing an absent document fails rather
    than being skipped: a label pointing at a missing document is a broken label, and
    silently dropping it would hide that.
    """
    report = Report()
    seen_tokens: list[tuple[str, set[str]]] = []

    for row in rows:
        qid = row.get("id", "?")
        answer = row.get("answer", "").strip()
        doc_ids = row.get("relevant_doc_ids") or []

        if not doc_ids:
            report.checks.append(
                Check(qid, Status.NO_SOURCE, detail="no relevant_doc_ids")
            )
            continue

        missing_docs = [d for d in doc_ids if d not in documents]
        if missing_docs:
            report.checks.append(
                Check(qid, Status.NO_SOURCE, detail=f"document not found: {missing_docs}")
            )
            continue

        if not answer:
            report.checks.append(Check(qid, Status.EMPTY, detail="empty answer"))
            continue

        answer_norm = _normalize_for_match(answer)
        if not answer_norm:
            report.checks.append(Check(qid, Status.EMPTY, detail="no comparable text"))
            continue

        corpus = "\n".join(documents[d] for d in doc_ids)
        support, missing = check_answer(answer, corpus, threshold)

        if support < threshold:
            report.checks.append(
                Check(
                    qid,
                    Status.UNSUPPORTED,
                    support=support,
                    missing=missing[:8],
                    detail=(
                        f"only {support:.0%} of the answer appears in "
                        f"{', '.join(doc_ids)}; missing {missing[:5]}"
                    ),
                )
            )
            continue

        # Near-duplicate detection. Two identical answers mean the set is measuring one
        # question twice, which inflates every score by weighting it double.
        answer_tokens = normalize(content_tokens(answer))
        duplicate_of = None
        for other_id, other_tokens in seen_tokens:
            if _jaccard(answer_tokens, other_tokens) >= DUP_THRESHOLD:
                duplicate_of = other_id
                break

        if duplicate_of:
            report.checks.append(
                Check(
                    qid,
                    Status.DUPLICATE,
                    support=support,
                    detail=f"near-duplicate of {duplicate_of}",
                )
            )
            continue

        seen_tokens.append((qid, answer_tokens))
        report.checks.append(
            Check(qid, Status.OK, support=support, detail=f"{support:.0%} supported")
        )

    return report


def mark_rows(rows: list[dict], report: Report) -> list[dict]:
    """Return copies with the mechanical verdict written into ``review``.

    A row a human already rejected stays rejected: a machine check cannot overturn a
    human decision, only add to it.
    """
    by_id = {c.id: c for c in report.checks}
    marked: list[dict] = []
    for row in rows:
        copy = dict(row)
        check = by_id.get(row.get("id", ""))
        if copy.get("review") == "REJECT":
            continue
        copy["review"] = check.status.value if check else Status.NO_SOURCE.value
        copy["support"] = round(check.support, 3) if check else 0.0
        marked.append(copy)
    return marked


def main() -> int:
    from .eval import load_goldens  # noqa: F401 - validates the path exists
    from .ingest import load_documents

    corpus_dir = Path("corpus")
    proposed = Path("evals/proposed.jsonl")

    if not proposed.exists():
        print(f"{proposed} not found - run: uv run python -m gateway.rag.label")
        return 1
    if not corpus_dir.exists():
        print(f"{corpus_dir} not found - run ./scripts/fetch_corpus.sh")
        return 1

    rows = [json.loads(line) for line in proposed.read_text().splitlines() if line.strip()]
    documents = {d.doc_id: d.text for d in load_documents(corpus_dir)}

    report = verify_rows(rows, documents)
    summary = report.to_dict()

    print(f"checked {summary['total']} proposed questions against {len(documents)} documents")
    print()
    for status, count in sorted(summary["by_status"].items()):  # type: ignore[union-attr]
        marker = "keep " if status == "OK" else "drop "
        print(f"  {marker}{status:12s} {count}")
    print()
    print(f"accepted {summary['accepted']}, rejected {summary['rejected']}")

    if summary["failures"]:
        print("\nfailures:")
        for failure in summary["failures"][:20]:  # type: ignore[index]
            print(f"  {failure['id']}  {failure['status']:12s} {failure['detail']}")  # type: ignore[index]
        if len(summary["failures"]) > 20:  # type: ignore[arg-type]
            print(f"  ... {len(summary['failures']) - 20} more")  # type: ignore[arg-type]

    marked = mark_rows(rows, report)
    with proposed.open("w", encoding="utf-8") as handle:
        for row in marked:
            handle.write(json.dumps(row) + "\n")

    Path("evals/verification.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nrewrote {proposed} with mechanical verdicts")
    print("wrote evals/verification.json")
    if summary["accepted"]:
        print("\nnext: uv run python -m gateway.rag.promote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

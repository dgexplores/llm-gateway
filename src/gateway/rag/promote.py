"""Promote reviewed questions into the gated golden set.

Deliberately strict. A question enters the gate only when a human has marked it
``OK``; everything else is dropped and reported. The alternative - including pending
rows and hoping they are fine - would let an unreviewed set become the basis of a
published number, which is the exact failure this project exists to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("evals/proposed.jsonl")
TARGET = Path("evals/goldens.jsonl")


def main() -> int:
    if not SOURCE.exists():
        print(f"{SOURCE} not found - run `uv run python -m gateway.rag.label` first")
        return 1

    accepted: list[dict] = []
    pending = 0
    rejected: list[str] = []

    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        status = record.get("review", "PENDING")
        if status == "OK":
            accepted.append(record)
        elif status == "REJECT":
            rejected.append(record["id"])
        else:
            pending += 1

    if not accepted:
        print('no reviewed questions. Set "review": "OK" on the ones you checked.')
        print(f"  {SOURCE}")
        print(f"  {pending} pending, {len(rejected)} rejected")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with TARGET.open("w", encoding="utf-8") as handle:
        for record in accepted:
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "answer": record["answer"],
                        "relevant_doc_ids": record["relevant_doc_ids"],
                        "difficulty": record.get("difficulty", "single-hop"),
                        "notes": record.get("notes", ""),
                    }
                )
                + "\n"
            )

    print(f"promoted {len(accepted)} reviewed questions to {TARGET}")
    if pending:
        print(f"  {pending} still pending, not included in the gate")
    if rejected:
        print(f"  {len(rejected)} rejected: {', '.join(rejected[:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

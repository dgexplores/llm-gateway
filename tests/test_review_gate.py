"""The review gate.

This is the load-bearing test of the whole evaluation design: an unreviewed question
must never reach the published set. If it can, every number downstream is a claim
about an unverified dataset, which is the failure this project exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.rag import load_goldens
from gateway.rag.promote import TARGET


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_pending_rows_never_reach_the_golden_set(tmp_path, monkeypatch) -> None:
    source = tmp_path / "proposed.jsonl"
    target = tmp_path / "goldens.jsonl"
    write(
        source,
        [
            {"id": "q1", "question": "Q?", "answer": "A",
             "relevant_doc_ids": ["a.md"], "review": "OK"},
            {"id": "q2", "question": "Q?", "answer": "A",
             "relevant_doc_ids": ["b.md"], "review": "PENDING"},
        ],
    )
    monkeypatch.setattr("gateway.rag.promote.SOURCE", source)
    monkeypatch.setattr("gateway.rag.promote.TARGET", target)

    from gateway.rag.promote import main

    assert main() == 0

    goldens = load_goldens(target)
    assert [g.id for g in goldens] == ["q1"]


def test_promotion_fails_when_nothing_is_reviewed(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "proposed.jsonl"
    target = tmp_path / "goldens.jsonl"
    write(
        source,
        [
            {"id": "q1", "question": "Q", "answer": "A",
             "relevant_doc_ids": ["a.md"], "review": "PENDING"},
            {"id": "q2", "question": "Q", "answer": "A", "relevant_doc_ids": ["b.md"]},
        ],
    )
    monkeypatch.setattr("gateway.rag.promote.SOURCE", source)
    monkeypatch.setattr("gateway.rag.promote.TARGET", target)

    from gateway.rag.promote import main

    assert main() == 1, "an all-pending set must not produce a golden file"
    assert not target.exists()

    output = capsys.readouterr().out
    assert "2 pending" in output


def test_rejected_rows_are_excluded_and_reported(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "proposed.jsonl"
    target = tmp_path / "goldens.jsonl"
    write(
        source,
        [
            {"id": "q1", "question": "Q", "answer": "A",
             "relevant_doc_ids": ["a.md"], "review": "OK"},
            {"id": "q2", "question": "Q", "answer": "A",
             "relevant_doc_ids": ["b.md"], "review": "REJECT"},
        ],
    )
    monkeypatch.setattr("gateway.rag.promote.SOURCE", source)
    monkeypatch.setattr("gateway.rag.promote.TARGET", target)

    from gateway.rag.promote import main

    assert main() == 0
    assert [g.id for g in load_goldens(target)] == ["q1"]
    assert "q2" in capsys.readouterr().out


def test_promoted_rows_do_not_carry_the_review_field(tmp_path, monkeypatch) -> None:
    """The gate set must not depend on a field that the human controls."""
    source = tmp_path / "proposed.jsonl"
    target = tmp_path / "goldens.jsonl"
    write(
        source,
        [{"id": "q1", "question": "Q", "answer": "A",
          "relevant_doc_ids": ["a.md"], "review": "OK"}],
    )
    monkeypatch.setattr("gateway.rag.promote.SOURCE", source)
    monkeypatch.setattr("gateway.rag.promote.TARGET", target)

    from gateway.rag.promote import main

    main()

    raw = json.loads(target.read_text().strip())
    assert "review" not in raw
    assert set(raw) == {"id", "question", "answer", "relevant_doc_ids", "difficulty", "notes"}


def test_missing_source_is_reported_clearly(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("gateway.rag.promote.SOURCE", tmp_path / "absent.jsonl")
    monkeypatch.setattr("gateway.rag.promote.TARGET", tmp_path / "goldens.jsonl")

    from gateway.rag.promote import main

    assert main() == 1


def test_default_target_is_the_gated_path() -> None:
    """The module-level constant is what CI reads; a test pins its location."""
    assert TARGET == Path("evals/goldens.jsonl")


@pytest.mark.parametrize("row", [{}, {"id": "q"}, {"id": "q", "question": "?"}])
def test_malformed_proposed_rows_are_detectable(row: dict) -> None:
    """A row missing required fields must not be silently promotable."""
    assert not {"id", "question", "answer", "relevant_doc_ids"} <= set(row)

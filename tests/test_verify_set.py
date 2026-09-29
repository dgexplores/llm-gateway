"""Negative controls for the mechanical golden-set verifier.

The verifier accepted 51 of 51 proposed rows on its first run. A check that never
rejects is indistinguishable from a check that does nothing, so the verdict is only
worth publishing once the verifier is shown to fail on input it *should* fail.

These cases are the proof of teeth. If any of them starts passing, the 51/51 result
means nothing and ``docs/RESULTS.md`` must stop quoting it.
"""

from __future__ import annotations

import pytest

from gateway.rag.verify_set import (
    MIN_SUPPORT,
    Status,
    check_answer,
    verify_rows,
)

DOC_A = (
    "FastAPI supports a parameter called status_code which declares the HTTP status "
    "code used for the response. You would pass a valid HTTP status code to it."
)
DOC_B = (
    "A background task is a function that is run after a response has been returned. "
    "The add_task function receives the arguments for that function."
)


def _row(qid: str, answer: str, doc_id: str = "a.md") -> dict:
    return {
        "id": qid,
        "question": "Q?",
        "answer": answer,
        "relevant_doc_ids": [doc_id],
        "review": "PENDING",
    }


# --- the control that must pass


def test_a_genuine_answer_is_accepted() -> None:
    report = verify_rows(
        [_row("q1", "FastAPI supports a parameter called status_code.")],
        {"a.md": DOC_A},
    )

    assert report.accepted
    assert report.checks[0].status is Status.OK


# --- the controls that must fail


def test_a_fabricated_answer_is_rejected() -> None:
    """The whole point. An answer invented from a different page must not pass."""
    report = verify_rows(
        [_row("q1", "FastAPI automatically retries idempotent requests on a 503.")],
        {"a.md": DOC_A},
    )

    assert report.rejected
    assert report.checks[0].status is Status.UNSUPPORTED


def test_an_answer_citing_the_wrong_document_is_rejected() -> None:
    """The label is what the metric is computed against, so a wrong label has to fail
    even when the answer is perfectly true of some *other* document in the corpus."""
    report = verify_rows(
        [_row("q1", "The add_task function receives the arguments for that function.", "b.md")],
        {"a.md": DOC_A, "b.md": DOC_B},
    )

    # Cites b.md, and b.md does support it — so this is the positive case. The negative
    # control is the same answer citing a.md.
    assert report.accepted


def test_the_same_answer_against_the_wrong_document_is_rejected() -> None:
    report = verify_rows(
        [_row("q1", "The add_task function receives the arguments for that function.", "a.md")],
        {"a.md": DOC_A, "b.md": DOC_B},
    )

    assert report.rejected
    assert report.checks[0].status is Status.UNSUPPORTED


def test_a_label_pointing_at_a_missing_document_is_rejected() -> None:
    """A broken label must fail loudly. Skipping it would hide the breakage and quietly
    shrink the evaluated set."""
    report = verify_rows([_row("q1", "Anything at all.", "gone.md")], {"a.md": DOC_A})

    assert report.rejected
    assert report.checks[0].status is Status.NO_SOURCE


def test_an_empty_answer_is_rejected() -> None:
    report = verify_rows([_row("q1", "")], {"a.md": DOC_A})

    assert report.rejected
    assert report.checks[0].status is Status.EMPTY


def test_a_near_duplicate_answer_is_flagged() -> None:
    """A set that measures one question twice inflates the result."""
    report = verify_rows(
        [
            _row("q1", "FastAPI supports a parameter called status_code."),
            _row("q2", "FastAPI supports a parameter called the status_code."),
        ],
        {"a.md": DOC_A},
    )

    assert any(c.status is Status.DUPLICATE for c in report.checks)


def test_two_genuinely_different_answers_are_not_duplicates() -> None:
    report = verify_rows(
        [
            _row("q1", "FastAPI supports a parameter called status_code."),
            _row("q2", "A background task runs after a response has been returned.", "b.md"),
        ],
        {"a.md": DOC_A, "b.md": DOC_B},
    )

    assert len(report.accepted) == 2


# --- the threshold itself


def test_the_threshold_rejects_a_mostly_invented_answer() -> None:
    """Half invented is not a paraphrase. The threshold has to bite at the boundary."""
    answer = "FastAPI supports a parameter called status_code. " + (
        "Redis streams use consumer groups and pending entries lists instead. "
        * 3
    )

    support, _missing = check_answer(answer, DOC_A)

    assert support < MIN_SUPPORT


def test_the_threshold_accepts_a_paraphrase() -> None:
    """The generator normalises whitespace and markup, so exact word identity is the
    wrong test and the threshold has to leave room for it."""
    answer = (
        "FastAPI supports a parameter called `status_code` which declares the "
        "HTTP status code."
    )

    support, _missing = check_answer(answer, DOC_A)

    assert support >= MIN_SUPPORT


def test_a_rejection_explains_itself() -> None:
    """A boolean failure is unactionable. The missing words are the diagnosis."""
    report = verify_rows(
        [_row("q1", "Redis streams use consumer groups and pending entries lists.")],
        {"a.md": DOC_A},
    )

    check = report.checks[0]

    assert not check.ok
    assert check.missing, "a rejection must name the words that were not found"
    assert check.detail


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_the_threshold_is_honoured(threshold: float) -> None:
    report = verify_rows(
        [_row("q1", "FastAPI supports a parameter called status_code.")],
        {"a.md": DOC_A},
        threshold=threshold,
    )

    assert report.checks[0].ok is (check_answer(
        "FastAPI supports a parameter called status_code.", DOC_A
    )[0] >= threshold)


def test_the_report_serialises_for_publication() -> None:
    """The published number is this dict, so its shape is part of the contract."""
    report = verify_rows(
        [_row("q1", "FastAPI supports a parameter called status_code.")],
        {"a.md": DOC_A},
    )

    data = report.to_dict()

    assert data["total"] == 1
    assert data["accepted"] == 1
    assert data["rejected"] == 0
    assert data["by_status"] == {"OK": 1}
    assert data["min_support"] == MIN_SUPPORT
    assert data["failures"] == []

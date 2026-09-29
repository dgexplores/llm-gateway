"""Citation binding.

The invariant: an answer may only cite a source that was actually retrieved. A marker
that resolves to nothing is surfaced, never silently dropped.
"""

from __future__ import annotations

import pytest

from gateway.citations import (
    Citation,
    coverage,
    parse_citations,
    resolve,
    split_claims,
    strip_citations,
)


def cite(index: int, doc_id: str = "a.md") -> Citation:
    return Citation(index=index, chunk_id=f"c{index}", doc_id=doc_id, title="T", excerpt="x")


def test_parses_indices_in_order() -> None:
    assert parse_citations("First [1]. Second [2] and [1] again.") == [1, 2]


def test_strip_removes_all_markers() -> None:
    """Removing a marker must leave no residue: a doubled space, or a space stranded
    before the full stop. Both are visible once rendered."""
    assert strip_citations("Claim [1] and [2].") == "Claim and."
    assert strip_citations("Inline [3] citation.") == "Inline citation."


def test_resolve_binds_a_valid_citation() -> None:
    cited, dangling = resolve("Supported by [1].", [cite(1)])

    assert [c.index for c in cited] == [1]
    assert dangling == []


def test_resolve_reports_a_dangling_citation() -> None:
    """A citation with no matching source must be surfaced, not hidden."""
    cited, dangling = resolve("Invented [7].", [cite(1)])

    assert cited == []
    assert dangling == [7]


def test_resolve_deduplicates_repeated_citations() -> None:
    cited, _ = resolve("Both [1] and again [1].", [cite(1), cite(2)])

    assert [c.index for c in cited] == [1]


def test_resolve_with_no_citations() -> None:
    cited, dangling = resolve("No markers at all.", [cite(1)])

    assert cited == [] and dangling == []


def test_split_claims_attributes_per_sentence() -> None:
    """Per-claim, because an answer can be supported on one point and wrong on another."""
    claims = split_claims("Grounded [1]. Invented claim without support.")

    assert len(claims) == 2
    assert claims[0].citations == (1,)
    assert claims[1].citations == ()


def test_split_claims_strips_markers_from_text() -> None:
    assert split_claims("Text with [3].")[0].text == "Text with."


def test_coverage_measures_cited_fraction() -> None:
    assert coverage("Used [1] and [2].", [cite(1), cite(2), cite(3)]) == pytest.approx(2 / 3)


def test_coverage_is_zero_without_sources() -> None:
    assert coverage("anything", []) == 0.0


def test_coverage_is_one_when_everything_is_cited() -> None:
    assert coverage("[1] [2]", [cite(1), cite(2)]) == 1.0


def test_empty_answer_is_handled() -> None:
    assert split_claims("") == []
    assert parse_citations("") == []


def test_marker_after_the_full_stop_credits_the_previous_claim() -> None:
    """Regression, found in the UI: the claims panel showed "no citation" for a claim
    that clearly carried [1]. A marker written after the sentence belongs to that
    sentence, not to the one that follows it."""
    claims = split_claims(
        "First claim is supported. [1] Second claim is supported. [2] Third. [3]"
    )

    assert [c.citations for c in claims] == [(1,), (2,), (3,)]


def test_each_claim_gets_its_own_marker_in_a_realistic_answer() -> None:
    claims = split_claims(
        "FastAPI has a Dependency Injection system. [1] "
        "Using BackgroundTasks also works with it. [2] "
        "Before diving deeper, let's upgrade the example. [3]"
    )

    assert claims[0].citations == (1,)
    assert claims[1].citations == (2,)
    assert claims[2].citations == (3,)


def test_marker_before_the_full_stop_credits_the_same_claim() -> None:
    assert split_claims("Grounded [1]. Uncited.")[0].citations == (1,)


def test_trailing_marker_with_no_following_claim_is_kept() -> None:
    assert split_claims("Only claim. [7]")[0].citations == (7,)


def test_leading_marker_on_the_first_claim_is_kept() -> None:
    assert split_claims("[1] Opening claim.")[0].citations == (1,)


def test_uncited_claims_are_distinguishable_from_cited_ones() -> None:
    """This is the distinction the claims panel exists to show."""
    claims = split_claims("Supported claim. [1] Invented claim.")

    assert claims[0].citations
    assert not claims[1].citations

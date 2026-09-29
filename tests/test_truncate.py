"""Excerpt truncation.

The excerpt is both shown to the user and sent to the model, so a bad cut at either
edge is visible twice. These cases came from real output, not from theory: an answer
beginning "ou can use..." and a preview ending "...including a respo".
"""

from __future__ import annotations

from gateway.pipeline import truncate_excerpt


def test_short_text_is_untouched() -> None:
    assert truncate_excerpt("A short sentence.", 400) == "A short sentence."


def test_cuts_at_a_sentence_end() -> None:
    text = "First sentence here. Second sentence here. Third sentence here."

    result = truncate_excerpt(text, 45)

    assert result.endswith(".")
    assert result == "First sentence here. Second sentence here."


def test_never_leaves_a_partial_word() -> None:
    """The cut must land on a word boundary, not mid-token."""
    text = "alpha bravo charlie delta echo foxtrot golf hotel"

    result = truncate_excerpt(text, 30)

    words = set(text.split())
    assert result.split()[-1] in words
    assert not result.endswith("foxt")


def test_cuts_at_a_newline_when_there_is_no_sentence_end() -> None:
    text = "A heading without a period\nand then a body line that continues for a while"

    result = truncate_excerpt(text, 20)

    assert len(result) <= 20
    assert result == "A heading without a"


def test_handles_text_with_no_usable_boundary() -> None:
    """One long unbroken token must still be returned, not dropped."""
    result = truncate_excerpt("x" * 500, 100)

    assert result == "x" * 100


def test_strips_surrounding_whitespace() -> None:
    assert truncate_excerpt("   Spaced sentence.   ", 400) == "Spaced sentence."


def test_result_never_exceeds_the_limit() -> None:
    text = f"Sentence number {0}. " * 40
    for limit in (30, 60, 100, 200):
        assert len(truncate_excerpt(text, limit)) <= limit


def test_keeps_most_of_the_text_when_boundaries_are_sparse() -> None:
    """A limit should not discard most of the excerpt just because the sentence end
    sits slightly past it."""
    text = "Short one. " + ("padding words here " * 40) + "End."

    result = truncate_excerpt(text, 120)

    assert len(result) > 60


def test_trims_a_leading_sentence_fragment() -> None:
    """Chunks start mid-sentence by construction when a document is split at a size
    boundary. A chunk starting "e passing it to a response:" is unreadable in the
    preview and nonsensical when quoted back as an answer."""
    # The observed case: a chunk boundary landed mid-word, so the chunk opened with
    # the tail of a sentence rather than a sentence of its own.
    text = (
        "e passing it to a response: you can use the `jsonable_encoder` to convert "
        "your data first. It takes arbitrary objects and converts them to JSON. "
        "That is the whole point of the helper in this situation, and it is why "
        "the response class does not validate anything on your behalf."
    )

    result = truncate_excerpt(text, 400)

    # Cut at the colon, which keeps the clause the query was actually about. Cutting
    # to the next full stop instead would skip a whole sentence of relevant text.
    assert not result.startswith("e passing")
    assert result.startswith("you can use the `jsonable_encoder`")


def test_does_not_drop_a_leading_sentence_that_is_already_complete() -> None:
    """The trim must only fire on a genuine fragment. Otherwise every excerpt would
    silently lose its first sentence, which is worse than the problem it solves."""
    text = "First sentence here. Second sentence here. Third sentence here."

    assert truncate_excerpt(text, 400).startswith("First sentence here.")


def test_keeps_text_when_trimming_the_lead_would_discard_most_of_it() -> None:
    """Skipping to a sentence start must not throw away most of the excerpt."""
    text = "First fragment of a sentence. " + ("detail " * 60) + "End."

    result = truncate_excerpt(text, 120)

    assert len(result) > 40


def test_clean_markup_is_applied_before_display_and_generation() -> None:
    """The excerpt is both shown in the UI and sent to the model, so markup left in it
    is visible twice. Cleaning belongs on the excerpt, with one owner."""
    from gateway.pipeline import clean_markup

    raw = (
        "## Dependencies { #dependencies }\n"
        "**FastAPI** has a <dfn title=\"components\">Dependency Injection</dfn> system.\n"
        "See [the docs](https://example.com/x) and `{* ../../docs_src/a.py *}`.\n"
    )

    cleaned = clean_markup(raw)

    assert "##" not in cleaned
    assert "{ #" not in cleaned
    assert "<dfn" not in cleaned
    assert "https://" not in cleaned
    assert "docs_src" not in cleaned
    assert "Dependency Injection" in cleaned
    assert "the docs" in cleaned


def test_clean_markup_keeps_emphasis_content_but_drops_the_markers() -> None:
    from gateway.pipeline import clean_markup

    assert clean_markup("A **bold** claim.") == "A bold claim."


def test_clean_markup_drops_bullet_lines() -> None:
    from gateway.pipeline import clean_markup

    assert clean_markup("* a list item\nA real sentence here.") == "A real sentence here."

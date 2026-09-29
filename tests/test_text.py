"""The shared text rules.

These cases all came from real output against the FastAPI corpus. The failure mode is
consistent: documentation is written for rendering, so quoted and previewed text arrives
carrying markup that means nothing out of context.

The module exists because the same text is shown to the reader and sent to the model.
When the two cleanups were separate implementations they drifted, and the drift was
visible — the preview showed ``*italics*`` and ``/// tip`` while the quoted sentence did
not. These tests therefore cover the single owner, not each caller.
"""

from __future__ import annotations

import pytest

from gateway.text import (
    clean_markup,
    drop_leading_fragment,
    excerpt,
    is_quotable,
    sentences,
    truncate_excerpt,
)


@pytest.mark.parametrize(
    ("raw", "absent"),
    [
        ("{* ../../docs_src/response_directly/tutorial001_py310.py *}", "docs_src"),
        ("See [the docs](https://starlette.dev/) for more.", "https://"),
        ("```python\nprint(1)\n```\n\nThen prose follows here.", "```"),
        ('A <dfn title="components">Dependency</dfn> system.', "<dfn"),
        ("Response - Change Status Code { #response-change }", "{ #"),
        ("A **bold** and *italic* claim.", "**"),
        ("/// note | Technical Details\nThe response is returned as is.", "///"),
        ("!!! warning\n\nThe value must not be empty at all.", "!!!"),
    ],
)
def test_removes_rendering_residue(raw: str, absent: str) -> None:
    assert absent not in clean_markup(raw)


@pytest.mark.parametrize(
    ("raw", "present"),
    [
        ("[Starlette's docs](https://x.dev/)", "Starlette's docs"),
        ("A **Dependency Injection** system.", "Dependency Injection"),
        ("/// note | Title\nThe response is returned as is.", "returned as is"),
    ],
)
def test_keeps_the_prose_behind_the_markup(raw: str, present: str) -> None:
    """Stripping the marker is only correct if the content survives it."""
    assert present in clean_markup(raw)


def test_drops_heading_lines() -> None:
    assert "Response - Change" not in clean_markup(
        "## Response - Change\nYou probably read before this page existed."
    )


def test_drops_bullet_lines() -> None:
    """Removing the marker and keeping the text would quote a list item as prose."""
    assert "A task function" not in clean_markup("* A task function to run in the background.")


# --- the leading-fragment rule


def test_drops_a_leading_sentence_fragment() -> None:
    """Chunks start mid-sentence by construction when a document is split at a size
    boundary, and a chunk opening "e passing it to a response:" is unreadable."""
    result = drop_leading_fragment(
        "e passing it to a response: you can use the `jsonable_encoder` to convert "
        "your data first. It takes arbitrary objects and converts them to JSON. "
        "That is the whole point of the helper in this situation, and it is why "
        "the response class does not validate anything on your behalf."
    )

    assert not result.startswith("e passing")
    assert result.startswith("you can use the `jsonable_encoder`")


def test_leaves_text_that_starts_properly_alone() -> None:
    """Otherwise every passage would silently lose its first sentence."""
    assert drop_leading_fragment("The status code is set directly.") == (
        "The status code is set directly."
    )


def test_keeps_the_fragment_when_dropping_it_would_discard_everything() -> None:
    result = drop_leading_fragment("ing. Short.")

    assert result == "ing. Short."


# --- truncation


def test_short_text_is_untouched() -> None:
    """A range that is empty for short input would otherwise fall through to the word
    cut and truncate 'A short sentence.' to 'A short'."""
    assert truncate_excerpt("A short sentence.", 400) == "A short sentence."


def test_strips_surrounding_whitespace() -> None:
    assert truncate_excerpt("   Spaced sentence.   ", 400) == "Spaced sentence."


def test_cuts_at_a_sentence_end() -> None:
    result = truncate_excerpt("First sentence here. Second sentence here. Third here.", 45)

    assert result.endswith(".")
    assert result == "First sentence here. Second sentence here."


def test_cuts_at_the_last_word_boundary() -> None:
    result = truncate_excerpt("alpha bravo charlie delta echo foxtrot golf hotel", 30)

    assert result.split()[-1] in {"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"}


def test_handles_one_unbroken_token() -> None:
    """An empty excerpt is worse than a clipped word."""
    assert truncate_excerpt("x" * 500, 100) == "x" * 100


def test_never_exceeds_the_limit() -> None:
    text = "Sentence number zero. " * 40
    for limit in (30, 60, 100, 200):
        assert len(truncate_excerpt(text, limit)) <= limit


def test_keeps_most_of_the_text_when_boundaries_are_sparse() -> None:
    result = truncate_excerpt("Short one. " + ("padding words here " * 40) + "End.", 120)

    assert len(result) > 60


def test_excerpt_cleans_before_it_truncates() -> None:
    """Cleaning after truncation can leave a cut tag such as ``<dfn title="dep``,
    which is how markup ended up mid-sentence in the UI."""
    raw = "## Heading { #h }\n" + ("A sentence about the status_code parameter. " * 30)

    result = excerpt(raw, 200)

    assert "<" not in result
    assert "{" not in result
    assert result.endswith((".", "!", "?", ")"))


# --- sentence selection


def test_splits_into_sentences() -> None:
    found = sentences("First one. Second one. Third one here for length.")

    assert len(found) == 3


def test_a_heading_is_not_returned_as_a_sentence() -> None:
    """Headings are removed before the split, or one is quoted as the answer."""
    found = sentences("## Background Tasks\n\nAdd a task function.")

    assert found == ["Add a task function."]


def test_a_bullet_is_not_returned_as_a_sentence() -> None:
    assert sentences("* A task function to be run in the background.") == []


def test_a_marker_only_source_yields_nothing() -> None:
    assert sentences("{* ../../docs_src/a.py *}") == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Short. Fragment",  # two words
        "the status code can be set with the status_code parameter.",  # lowercase
        "A d",  # too thin to be a claim
        "100 ms",  # no letters
    ],
)
def test_rejects_what_is_not_a_claim(sentence: str) -> None:
    assert not is_quotable(sentence)


def test_accepts_a_short_but_complete_claim() -> None:
    """No length floor: 'Add a task function.' is short and perfectly good."""
    assert is_quotable("Add a task function.")


def test_accepts_a_full_prose_claim() -> None:
    assert is_quotable("The status code can be set with the status_code parameter.")


def test_multiline_source_is_read_whole() -> None:
    """Regression. A line-based parse kept only the first line of each source, so a
    heading was quoted as the entire answer."""
    found = sentences("## Background Tasks\n\nAdd a task function.\n\nIt runs after.")

    assert "Add a task function." in found
    assert "It runs after." in found


def test_handles_empty_input() -> None:
    assert sentences("") == []
    assert clean_markup("") == ""
    assert excerpt("") == ""


# --- integration: the two callers must share one implementation


def test_the_pipeline_cleans_excerpts_it_serves() -> None:
    """The preview and the quoted passage are the same text. If the pipeline grew its
    own cleaner, they would drift — which is how the preview came to show ``*italics*``
    while the quote did not."""
    from gateway.llm.base import Completion
    from gateway.pipeline import Gateway
    from gateway.rag import BM25Retriever, Chunk

    chunk = Chunk(
        "c1",
        "status.md",
        "## Status\n\n**The status code** is set with the [status_code](https://x.dev/) "
        "parameter on the path operation. " * 12,
        "Status",
    )

    class Stub:
        name = "stub"

        def complete(self, messages, *, request_id, temperature=0.0):
            return Completion(text="Answer [1].", model="stub")

    gateway = Gateway(BM25Retriever.build([chunk]), Stub())
    result = gateway.ask("status code", request_id="r1")

    preview = result.sources[0].excerpt
    assert "##" not in preview
    assert "**" not in preview
    assert "https://" not in preview
    assert "status_code" in preview


def test_the_demo_provider_cleans_what_it_quotes() -> None:
    """Same rule, other caller. One owner, so both behave identically."""
    from gateway.llm.base import Message
    from gateway.llm.demo import DemoProvider

    completion = DemoProvider().complete(
        [
            Message.system("s"),
            Message.user(
                "Sources:\n\n[1] **The status code** is set with the "
                "[status_code](https://x.dev/) parameter directly."
            ),
        ]
    )

    assert "**" not in completion.text
    assert "https://" not in completion.text
    assert "status code" in completion.text


# --- titles


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("## Response Status Code { #response-status-code }", "Response Status Code"),
        ("# A **bold** title", "A bold title"),
        ("## Plain Title", "Plain Title"),
    ],
)
def test_heading_drops_its_authoring_anchor(raw: str, expected: str) -> None:
    """The anchor is a MyST authoring device. A reader checking a citation needs the
    human title, not the slug the docs use to link to it."""
    from gateway.text import heading

    assert heading(raw) == expected


def test_source_titles_are_clean_at_ingest() -> None:
    """Fixed where the title is derived, so every reader of it is correct rather than
    each one patching the residue independently."""
    import tempfile
    from pathlib import Path

    from gateway.rag.ingest import load_documents

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.md").write_text(
            "## Response Status Code { #response-status-code }\n\nBody.\n",
            encoding="utf-8",
        )
        assert load_documents(root)[0].title == "Response Status Code"


# --- dangling tails


def test_a_cut_never_ends_on_a_function_word() -> None:
    """"If you " at the end of an excerpt reads as a broken sentence, and a reader
    cannot tell it apart from a genuine trailing clause."""
    from gateway.text import truncate_excerpt

    result = truncate_excerpt(
        "You can declare the HTTP status code used by the path operation. If you want "
        "to return a response body you can declare a response model directly instead.",
        90,
    )

    assert result.split()[-1].lower() not in {"if", "you", "the", "to", "of", "and"}
    assert result.endswith(".")


# --- residue severed by a chunk boundary


def test_drops_a_half_cut_include_directive() -> None:
    """The opening "{* ../../docs_src/back" is in the previous chunk, so what arrives
    is only the tail. It looks like prose to every other rule, which is how a filename
    ended up in a preview."""
    from gateway.text import clean_markup, excerpt

    raw = (
        "round_tasks/tutorial001_py310.py hl[1,13] *}\n\n"
        "**FastAPI** will create the object of type `BackgroundTasks` for you and pass "
        "it as that parameter."
    )

    for result in (clean_markup(raw), excerpt(raw)):
        assert "hl[" not in result
        assert "tutorial001" not in result
        assert result.startswith("FastAPI")


def test_drops_a_half_cut_directive_without_highlights() -> None:
    from gateway.text import clean_markup

    assert clean_markup("../../docs_src/app.py *} The rest is prose.").startswith(
        "The rest is prose."
    )


def test_a_dot_inside_a_filename_is_not_a_sentence_boundary() -> None:
    """The fragment rule was corrupting markup: "tutorial001.py" was cut at its dot,
    leaving "py hl[1,13] }" — strictly worse than the input."""
    result = drop_leading_fragment("tutorial001.py is the module that defines the app.")

    assert result.startswith("tutorial001.py")
    assert not result.startswith("py")


def test_a_real_sentence_end_is_still_a_boundary() -> None:
    result = drop_leading_fragment("ing badly. The status code is returned as declared.")

    assert result.startswith("The status code")


def test_an_excerpt_never_opens_on_punctuation() -> None:
    """Severing "add_task():" leaves a leading ".add_task()" in the preview."""
    from gateway.text import truncate_excerpt

    result = truncate_excerpt(
        "add_task(): .add_task() receives as arguments a task function to be run in "
        "the background of the response.",
        120,
    )

    assert result[0] not in ".,;:)]}"

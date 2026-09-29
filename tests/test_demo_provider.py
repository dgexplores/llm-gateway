"""The demo provider.

Every case here came from real output against the FastAPI corpus, not from a
hypothetical. The failure mode is consistent: documentation is written for rendering,
so quoted text arrives carrying heading markers, source-include directives, anchor
slugs, and fragments left by the chunk boundary. Each becomes visible in an answer.
"""

from __future__ import annotations

import pytest

from gateway.llm.base import Message, ProviderError
from gateway.llm.demo import DemoProvider

# A quote must be substantial and terminate with punctuation. A length check alone
# lets short fragments through, which is how "First source sentence here." became a
# refusal rather than an answer.
MIN_QUOTE_CHARS = 40


def ask(*texts: str) -> str:
    """Send the given strings as numbered sources and return the answer text."""
    blocks = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(texts, start=1))
    return DemoProvider().complete(
        [Message.system("s"), Message.user(f"Question: q\n\nSources:\n{blocks}")]
    ).text


def test_quotes_a_sentence_with_its_citation() -> None:
    text = ask("The status code can be set with the status_code parameter.")

    assert "[1]" in text
    assert "status_code" in text


def test_index_matches_the_source_number() -> None:
    """A citation pointing at the wrong source is worse than none."""
    text = ask(
        "The first source contains enough prose to be quotable here.",
        "The second source contains enough prose to be quotable here.",
    )

    assert "first source contains enough prose" in text and "[1]" in text
    assert "second source contains enough prose" in text and "[2]" in text


def test_skips_a_source_with_no_quotable_sentence() -> None:
    """Source 2 is markup only, so the answer cites just source 1 and stays honest."""
    text = ask(
        "The status code can be set directly on the path.",
        r"{* ../../docs_src/response_directly/tutorial001_py310.py *}",
    )

    assert "[1]" in text
    assert "[2]" not in text


def test_strips_anchor_slugs() -> None:
    text = ask(
        "Response - Change Status Code { #response-change-status-code }\n\n"
        "You can set it directly on the path operation."
    )

    assert "{ #" not in text
    assert "directly on the path operation" in text


def test_drops_heading_lines() -> None:
    """A heading and its paragraph are one newline apart, so flattening them quotes the
    heading as the opening of the answer."""
    text = ask("## Response - Change Status Code\nYou probably read before this page existed.")

    assert "Response - Change Status Code You probably" not in text
    assert "You probably read before this page existed." in text


def test_strips_include_directives() -> None:
    text = ask(
        r"{* ../../docs_src/response_directly/tutorial001_py310.py hl[5:6] *}" "\n\n"
        "The response object is used exactly as it is returned."
    )

    assert "docs_src" not in text
    assert "used exactly as it is returned" in text


def test_strips_html_doc_tags() -> None:
    text = ask(
        'A <dfn title="also known as components">Dependency Injection</dfn> system '
        "is available for routing dependencies."
    )

    assert "<dfn" not in text
    assert "Dependency Injection" in text


def test_keeps_link_text_and_drops_the_url() -> None:
    text = ask("See [the Starlette docs](https://starlette.dev/background/) for details.")

    assert "https://" not in text
    assert "Starlette docs" in text


def test_strips_code_fences() -> None:
    text = ask("```python\nprint('x')\n```\n\nThe response is returned to the client directly.")

    assert "```" not in text
    assert "returned to the client directly" in text


def test_drops_a_leading_dangling_inline_span() -> None:
    """A span opened before the chunk boundary leaves a leading backtick."""
    text = ask("add_task()`:\n\nThe task function receives its arguments as parameters.")

    assert "add_task" not in text
    assert "receives its arguments" in text


def test_never_quotes_a_bullet_as_prose() -> None:
    """A list item is a real claim but not a prose one. A source that is only a list
    has nothing quotable, so the provider refuses rather than dressing the item up as
    a sentence."""
    with pytest.raises(ProviderError, match="no usable sentence"):
        ask("* A task function to be run in the background by the application.")


def test_prefers_prose_over_a_list_item_in_the_same_source() -> None:
    text = ask(
        "* A task function to be run in the background by the application.\n"
        "The task function is a standard function that receives its parameters "
        "directly from the caller."
    )

    assert "standard function" in text
    assert "A task function to be run" not in text


def test_marks_output_as_demo() -> None:
    completion = DemoProvider().complete(
        [
            Message.system("s"),
            Message.user("Sources:\n\n[1] The status code can be set directly here."),
        ]
    )

    assert completion.metadata["demo"] is True


def test_reports_token_counts() -> None:
    completion = DemoProvider().complete(
        [
            Message.system("s"),
            Message.user("Sources:\n\n[1] The status code can be set directly here."),
        ]
    )

    assert completion.prompt_tokens > 0
    assert completion.completion_tokens > 0


def test_refuses_without_context() -> None:
    """No sources means nothing to quote. Fabricating would be the one thing this
    provider must never do."""
    with pytest.raises(ProviderError, match="no context"):
        DemoProvider().complete([Message.user("no sources here")])


def test_refuses_when_context_has_only_markup() -> None:
    with pytest.raises(ProviderError, match="no usable sentence"):
        DemoProvider().complete(
            [Message.user("Sources:\n\n[1] {* ../../docs_src/a.py *}\n[2] ```\n```")]
        )


def test_error_is_not_retryable() -> None:
    """A malformed request will not succeed on a second attempt."""
    with pytest.raises(ProviderError) as caught:
        DemoProvider().complete([Message.user("nothing")])

    assert caught.value.retryable is False


def test_limits_how_many_sources_are_quoted() -> None:
    provider = DemoProvider(max_sentences=2)

    blocks = "\n\n".join(
        f"[{i}] Sentence number {i} has enough length to be quotable prose."
        for i in range(1, 6)
    )
    text = provider.complete(
        [Message.system("s"), Message.user(f"Question: q\n\nSources:\n{blocks}")]
    ).text

    assert text.count("[") == 2


def test_multiline_source_text_is_read_whole() -> None:
    """Regression. A line-based parse kept only the first line of each source, so a
    heading was quoted as the entire answer."""
    text = ask(
        "## Background Tasks\n\nAdd a task function.\n\nIt runs after the response."
    )

    assert "Add a task function." in text

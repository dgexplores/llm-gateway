"""Text normalisation for anything derived from documentation.

Documentation is written for rendering, so extracted text carries markup that means
nothing out of context: heading markers, source-include directives, anchor slugs,
admonition fences, emphasis, link syntax, bare URLs, and sentence fragments left by a
chunk boundary.

This module is the single owner of that cleanup because the same text is used in two
places — the preview shown to the reader and the passage quoted back to the model. Two
implementations meant the two drifted, and the drift was visible: the preview showed
``*italics*`` and ``/// tip`` while the quoted sentence did not.
"""

from __future__ import annotations

import re

# Rendering residue.
_INCLUDE_DIRECTIVE = re.compile(r"\{\*.*?\*\}", re.DOTALL)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_ANCHOR_SLUG = re.compile(r"\s*\{[^{}#]*#[^{}]*\}")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_BARE_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_INLINE_CODE = re.compile(r"`([^`]*)`")
# Admonition fences: "/// note | Title" and "!!! warning". The label and the title are
# authoring scaffolding, never part of the prose.
_ADMONITION = re.compile(r"^\s*(?:///|!!!|\?\?\?)\s*\w*[^|\n]*\|?\s*", re.MULTILINE)
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s")
# A heading marker can land mid-line, not just at the start of one. It happens wherever
# a document is concatenated or a chunk boundary merges two sections, and an inline
# "## Status" is as meaningless in a preview as one on its own line.
_INLINE_HEADING = re.compile(r"\s*#{1,6}\s+[A-Za-z][^\n]{0,60}?(?=\s{2,}|\.\s|\.$|$)")
_BULLET_LINE = re.compile(r"^\s*[-*+]\s+")
# The tail of an include directive severed by a chunk boundary. The opening "{*
# ../../docs_src/x.py" sits in the previous chunk, so what arrives here is
# ".../x.py hl[1,13] *}". It looks like prose to every other rule, which is how a
# filename ended up in a preview.
_DIRECTIVE_TAIL = re.compile(
    r"\A\s*[^\n]{0,300}?[\w/]\.(?:py|md|html|json|toml|ya?ml|css|js|ts)\b[^\n]{0,80}?\*?\}\s*"
)
_SPACES = re.compile(r"[ \t]{2,}")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# A leading backtick is the tail of an inline span opened before the chunk boundary.
_DANGLING_SPAN = re.compile(r"^`+\S*?`?[:;]?\s*")

# A block opening on a lowercase word is the tail of a sentence the chunk boundary cut
# in half. Fragment scoring: how much would be left if the fragment were dropped.
_MIN_KEEP = 40
_KEEP_RATIO = 0.4


# Words that cannot end a thought. An excerpt or quote that stops on one of these is
# visibly truncated mid-clause, which is the exact defect trimming is meant to prevent.
_DANGLING_TAIL = frozenset(
    """a an the of to in on for and or but with that which who whose if when
    while as by at from into onto is are was were be been being this these those
    it its than then also""".split()
)


def heading(text: str) -> str:
    """A heading usable as an evidence label.

    The corpus uses MyST, so a heading arrives carrying its own anchor — "Response
    Status Code { #response-status-code }". The anchor is an authoring device; a reader
    checking a citation needs the human title.
    """
    cleaned = _ANCHOR_SLUG.sub("", text.strip().lstrip("#").strip())
    cleaned = _EMPHASIS.sub(r"\1", _INLINE_CODE.sub(r"\1", cleaned))
    return _SPACES.sub(" ", cleaned).strip()


def _drop_dangling_tail(text: str) -> str:
    """Trim trailing function words left by a cut.

    "If you " at the end of an excerpt reads as a broken sentence, and a reader has no
    way to tell it apart from a genuine trailing clause. Only the words are dropped, so
    a claim never loses its content — but the sentence does lose its ending, so this
    only applies after a cut, never to text that already fits.
    """
    words = text.split()
    while len(words) > 4 and words[-1].lower().strip(".,;:") in _DANGLING_TAIL:
        words.pop()
    return " ".join(words) if len(words) != len(text.split()) else text


def clean_markup(text: str) -> str:
    """Remove documentation markup, keeping the prose.

    Link text and code identifiers survive; the URL and the markers do not. A preview
    showing ``[Starlette's docs](https://...)`` or ``{* ../../docs_src/x.py *}`` tells
    the reader nothing and makes a retrieval score impossible to trust.
    """
    cleaned = _DIRECTIVE_TAIL.sub(" ", text)
    cleaned = _INCLUDE_DIRECTIVE.sub(" ", cleaned)
    cleaned = _CODE_FENCE.sub(" ", cleaned)
    cleaned = _MD_LINK.sub(r"\1", cleaned)
    cleaned = _ANCHOR_SLUG.sub(" ", cleaned)
    cleaned = _HTML_TAG.sub("", cleaned)
    cleaned = _ADMONITION.sub("", cleaned)
    cleaned = _BARE_URL.sub("", cleaned)
    cleaned = _EMPHASIS.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    cleaned = _INLINE_HEADING.sub(" ", cleaned)

    # Headings and list items are dropped line by line. Removing the marker and keeping
    # the text would quote a list item as if it were a sentence.
    cleaned = "\n".join(
        line
        for line in cleaned.splitlines()
        if not _HEADING_LINE.match(line) and not _BULLET_LINE.match(line)
    )
    return _SPACES.sub(" ", cleaned).strip()


def drop_leading_fragment(block: str) -> str:
    """Remove a sentence fragment left at the start of a chunk.

    Chunks begin mid-sentence by construction whenever a document is split at a size
    boundary, so a chunk opening "e passing it to a response: you can use the
    jsonable_encoder" is unreadable in a preview and nonsensical quoted as an answer.
    The cut is only taken when enough survives to be worth keeping, and a colon counts
    as a boundary so the clause the query was about survives rather than a whole
    sentence being skipped to reach the next full stop.
    """
    cleaned = block.strip()
    if not cleaned or not cleaned[0].islower():
        return cleaned

    for index, char in enumerate(cleaned):
        if char == "." and cleaned[index + 1 : index + 2].isalnum():
            continue  # a filename or an abbreviation, not a sentence end
        if char in ".!?:":
            remainder = cleaned[index + 1 :].strip()
            if len(remainder) >= max(_MIN_KEEP, int(len(cleaned) * _KEEP_RATIO)):
                return remainder
            return cleaned

    # No boundary at all: drop the leading partial word.
    space = cleaned.find(" ")
    if 0 < space < 40:
        return cleaned[space + 1 :].strip()
    return cleaned


def _blocks(text: str) -> list[str]:
    """Paragraph-aligned blocks, cleaned and de-fragmented.

    A blank line inside a span means it crosses a paragraph boundary, so the text is
    several things rather than one sentence. Headings carry no prose and are removed
    before the split, or they are quoted as the opening of an answer.
    """
    found: list[str] = []
    for block in re.split(r"\n\s*\n", clean_markup(text)):
        body = " ".join(
            line.strip() for line in block.splitlines() if line.strip()
        ).strip()
        if not body:
            continue
        body = _DANGLING_SPAN.sub("", body).strip()
        if body:
            found.append(drop_leading_fragment(body))
    return found


def sentences(text: str) -> list[str]:
    """Sentence candidates, cleaned and de-fragmented."""
    found: list[str] = []
    for block in _blocks(text):
        for part in _SENTENCE.split(block):
            candidate = _SPACES.sub(" ", part).strip()
            candidate = _DANGLING_SPAN.sub("", candidate).strip()
            if candidate:
                found.append(candidate)
    return found


def is_quotable(sentence: str) -> bool:
    """Whether a sentence reads as a claim rather than a fragment.

    Prose only. A bullet or a heading can be long and capitalised while still being
    unusable as an answer, so length alone is not a sufficient test — and there is no
    length floor, because "Add a task function." is short and perfectly good.
    """
    if not sentence.endswith((".", "!", "?")):
        return False
    if not sentence[0].isupper():
        return False
    return bool(re.search(r"[a-z]{3,}", sentence))


def truncate_excerpt(text: str, limit: int) -> str:
    """Cut to ``limit`` characters at a sentence or word boundary.

    The excerpt is shown to the reader and sent to the model, so a bad cut at either
    edge is visible twice. The leading fragment is trimmed first: an excerpt that opens
    mid-sentence is a defect at the front edge, not the back.
    """
    # Severing a code reference such as "add_task():" leaves a leading ".add_task()".
    # The punctuation is what survived the cut, so it goes first.
    window = drop_leading_fragment(text.strip()).lstrip(".,;:)]} \t\n")

    # Text that already fits needs no cutting. Returning early also avoids the
    # boundary search below, whose range is empty for short input and would otherwise
    # fall through to the word cut and truncate "A short sentence." to "A short".
    if len(window) <= limit:
        return window

    window = window[:limit]

    # Prefer the last sentence or line end, so the excerpt is a whole number of
    # sentences. The 50% floor stops a boundary just past the limit from discarding
    # most of the excerpt to gain one word.
    for boundary in range(len(window) - 1, int(limit * 0.5) - 1, -1):
        if window[boundary] in ".!?\n":
            return _drop_dangling_tail(window[: boundary + 1].strip())

    space = window.rfind(" ")
    if space > 0:
        return _drop_dangling_tail(window[:space].strip())

    # One unbroken token longer than the limit. A clipped word beats an empty excerpt.
    return window.strip()


def excerpt(text: str, limit: int = 400) -> str:
    """Clean, then truncate.

    The order is not interchangeable: cleaning after truncation can leave a cut tag
    such as ``<dfn title="dep``, which is how markup ended up mid-sentence in the UI.
    """
    return truncate_excerpt(clean_markup(text), limit)

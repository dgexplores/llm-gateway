"""Binding answer claims to source spans.

An answer that says ``[2]`` is worthless unless 2 resolves to a specific retrieved
chunk. This module makes that binding explicit and checkable, and refuses to present
an index that does not resolve. A dangling citation in a UI is worse than no
citation: it implies support that was never established.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Removing an inline marker leaves its neighbouring whitespace behind, which turns
# "claim [1]." into "claim ." — visible in rendered output and a giveaway that the
# citation layer did the stripping.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


@dataclass(frozen=True)
class Citation:
    """One resolved citation."""

    index: int
    chunk_id: str
    doc_id: str
    title: str
    excerpt: str
    resolved: bool = True


@dataclass(frozen=True)
class Claim:
    """A sentence of the answer, with whatever it cited."""

    text: str
    citations: tuple[int, ...]


def parse_citations(answer: str) -> list[int]:
    """Citation indices in order of appearance, de-duplicated."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        index = int(match.group(1))
        if index not in seen:
            seen.append(index)
    return seen


def strip_citations(answer: str) -> str:
    """Answer text with markers removed, for display.

    Whitespace is tidied because removing a marker leaves residue: a doubled space
    mid-sentence, or a space stranded before the full stop.
    """
    text = re.sub(r"[ \t]{2,}", " ", _CITATION.sub("", answer))
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()


def split_claims(answer: str) -> list[Claim]:
    """Break the answer into sentences, each carrying its own citations.

    Per-claim rather than per-answer attribution, because the failure this project
    cares about is partial: an answer can be supported on one point and fabricated on
    another, and a single answer-level verdict hides that.

    A marker written *after* the full stop belongs to the sentence before it. Splitting
    on sentence boundaries alone hands it to the following sentence, so
    "supported claim. [1] next claim." would mark the supported claim unsupported and
    credit [1] to the wrong text. Such markers are therefore moved back onto the claim
    they follow.
    """
    leading_markers = re.compile(r"^\s*(?:\[\d+\]\s*)+")

    def clean(segment: str) -> tuple[str, tuple[int, ...]]:
        text = _SPACE_BEFORE_PUNCT.sub(
            r"\1", re.sub(r"[ \t]{2,}", " ", _CITATION.sub("", segment))
        ).strip()
        return text, tuple(int(m.group(1)) for m in _CITATION.finditer(segment))

    claims: list[Claim] = []
    # Markers before the first claim have nothing to attach to yet, so they are held
    # and applied once it exists.
    opening: tuple[int, ...] = ()

    def credit(indices: tuple[int, ...]) -> None:
        """Attach indices to the claim just recorded."""
        nonlocal opening
        if not indices:
            return
        if not claims:
            opening = opening + indices
            return
        previous = claims[-1]
        claims[-1] = Claim(text=previous.text, citations=previous.citations + indices)

    for segment in _SENTENCE_SPLIT.split(answer.strip()):
        # Markers sitting at the start of a segment follow the previous sentence.
        lead = leading_markers.match(segment)
        if lead:
            credit(tuple(int(m.group(1)) for m in _CITATION.finditer(lead.group(0))))
            segment = segment[lead.end() :]

        text, indices = clean(segment)
        if not text:
            # A segment that was only markers has already been credited above.
            continue

        if opening:
            indices = opening + indices
            opening = ()
        claims.append(Claim(text=text, citations=indices))

    if opening and claims:
        # Text was all markers; credit the last claim so the reference is not lost.
        credit(opening)

    return claims


def resolve(
    answer: str,
    sources: list[Citation],
) -> tuple[list[Citation], list[int]]:
    """Bind answer citations to sources.

    Returns the cited sources plus any indices the answer referenced that do not
    resolve. The dangling list is surfaced in the UI rather than silently dropped.
    """
    by_index = {source.index: source for source in sources}
    cited: list[Citation] = []
    dangling: list[int] = []

    for index in parse_citations(answer):
        source = by_index.get(index)
        if source is None:
            if index not in dangling:
                dangling.append(index)
            continue
        if source not in cited:
            cited.append(source)

    return cited, dangling


def coverage(answer: str, sources: list[Citation]) -> float:
    """Fraction of provided sources the answer actually cited.

    A low value means the generator ignored most of what it was given, which is a
    retrieval or prompt problem rather than a generation one. Surfaced so the two are
    not confused when a result is poor.
    """
    if not sources:
        return 0.0
    used = set(parse_citations(answer))
    return len(used & {s.index for s in sources}) / len(sources)

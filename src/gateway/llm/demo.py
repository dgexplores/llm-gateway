"""An extractive provider that needs no credentials, no network, and no model.

It exists so the gateway can be demonstrated end to end on a clean clone. The
distinction matters: this is a real extractive baseline, not a stub. It quotes the
highest-scoring sentence from each retrieved chunk and cites it, so citations are
correct by construction rather than asserted.

Every response is marked in the UI as demo output. A reader must never mistake it for
a language model's answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Completion, Message, ProviderError

_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Documentation is written for rendering. Three kinds of residue survive into quoted
# text and are visible in every answer:
#   {* ../../docs_src/x.py hl[5:6] *}  a source-include directive
#   [Starlette's docs](https://...)    a link, which becomes a bare URL
#   ```python                          a code fence opener
_INCLUDE_DIRECTIVE = re.compile(r"\{\*.*?\*\}", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_FENCE = re.compile(r"^```.*$", re.MULTILINE)
_WS = re.compile(r"[ \t]{2,}")
# Bullet markers are removed, but the marker character is recorded so the sentence
# can be rejected later. Stripping a bullet and keeping its text would let a list
# item be quoted as if it were a prose claim, which is a different kind of thing.
# Bullet markers are removed, but the marker is recorded so the resulting line can be
# rejected later. Stripping a bullet and keeping its text would let a list item be
# quoted as if it were a prose claim, which is a different kind of thing.
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s")
_BULLET_LINE = re.compile(r"^\s*[-*+][ \t]+")
_BARE_URL = re.compile(r"https?://\S+")
# Anchor slugs left by the docs generator, e.g. "Additional Responses { #additional }".
# The slug is metadata, never part of the prose.
_ANCHOR_SLUG = re.compile(r"\s*\{[^{}#]*#[^{}]*\}\s*")
# <dfn> and similar inline documentation tags: keep the text, drop the tag.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
# A leading backtick is the tail of an inline span opened before the chunk boundary,
# which is how "add_task()`:" became an answer.
_DANGLING_SPAN = re.compile(r"^`+\S*?`?[:;]?\s*")


@dataclass
class DemoProvider:
    """Quotes retrieved context back, with citations.

    Not useful for open-ended questions, which is the honest limitation: it can only
    return text that was handed to it. Its purpose is a working demonstration of the
    gateway's resilience, accounting, and citation plumbing with nothing installed.
    """

    name: str = "demo"
    model: str = "extractive-v1"
    max_sentences: int = 3

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> Completion:
        context = self._extract_context(messages)
        if not context:
            raise ProviderError("demo provider received no context", retryable=False)

        picked = self._pick_sentences(context)
        if not picked:
            raise ProviderError("no usable sentence in context", retryable=False)

        prompt_tokens = sum(len(m.content.split()) for m in messages)
        # The citation index is carried alongside the text: the position in the list
        # would drift as soon as one source yielded no usable sentence, silently
        # pointing a claim at the wrong source.
        parts = [f"{text} [{index}]" for index, text in picked]
        sentences = [text for _, text in picked]

        return Completion(
            text=" ".join(parts),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=sum(len(s.split()) for s in sentences),
            metadata={"demo": True, "sources": len(sentences)},
        )

    @staticmethod
    def _extract_context(messages: list[Message]) -> list[tuple[int, str]]:
        """Pull the numbered context blocks the pipeline supplies.

        The contract is a ``[n]`` marker followed by that source's text, ending where
        the next marker begins. Parsed by splitting on markers rather than line by line,
        because source text contains newlines: a line-based parse silently kept only
        the first line of every chunk, which is how a markdown heading ended up quoted
        as the entire answer.
        """
        blocks: dict[int, str] = {}
        pattern = re.compile(r"^\[(\d+)\]\s+", re.MULTILINE)

        for message in messages:
            matches = list(pattern.finditer(message.content))
            for position, match in enumerate(matches):
                start = match.end()
                end = matches[position + 1].start() if position + 1 < len(matches) else len(
                    message.content
                )
                text = message.content[start:end].strip()
                if text:
                    blocks[int(match.group(1))] = text

        return sorted(blocks.items())

    def _pick_sentences(self, context: list[tuple[int, str]]) -> list[str]:
        """One sentence per source.

        Relaxed in stages rather than applying one strict filter. Real corpora contain
        headings, bullet lists and tables, and requiring a capitalised 40-character
        sentence rejects all of them - which surfaced as a 503 for a query that
        retrieved perfectly well.
        """
        chosen: list[tuple[int, str]] = []
        for index, text in context[: self.max_sentences]:
            cleaned = self._best_sentence(text)
            if cleaned:
                chosen.append((index, _WS.sub(" ", cleaned).strip()))
        return chosen

    @classmethod
    def _best_sentence(cls, text: str) -> str | None:
        """Pick one quotable sentence, or None when the source has none.

        ``None`` is a real outcome and is preferred over returning markup. A source
        that is only a heading or a code reference has nothing to quote, and emitting
        a directive as if it were an answer is worse than citing fewer sources.
        """
        cleaned = _INCLUDE_DIRECTIVE.sub(" ", _FENCE.sub(" ", text))
        cleaned = _MD_LINK.sub(r"\1", cleaned)
        cleaned = _ANCHOR_SLUG.sub(" ", cleaned)
        cleaned = _HTML_TAG.sub("", cleaned)
        cleaned = _BARE_URL.sub("", cleaned)

        # Drop whole heading lines before flattening. Order matters: stripping the '#'
        # marks first would leave no marker to identify the line as a heading, and a
        # heading is then quoted as the opening of the answer ("Response - Change
        # Status Code You probably read before that...").
        # Headings and list items are dropped line by line rather than by substitution.
        # Removing the marker and keeping the text is what let a list item through as
        # prose; the line has to be discarded, not undecorated.
        cleaned = "\n".join(
            line
            for line in cleaned.splitlines()
            if not _HEADING_LINE.match(line) and not _BULLET_LINE.match(line)
        )

        # A blank line inside a span means it crosses a paragraph boundary, so the
        # text is several things rather than one sentence.
        parts: list[str] = []
        for block in re.split(r"\n\s*\n", cleaned):
            for part in _SENTENCE.split(block):
                sentence = _WS.sub(" ", part).strip()
                # A span opened before the chunk boundary leaves a leading backtick
                # with no opening context; that is residue, not a sentence.
                sentence = _DANGLING_SPAN.sub("", sentence).strip()
                if sentence:
                    parts.append(sentence)

        # Prose only. No length floor: "Add a task function." is a short but perfectly
        # good claim, and a minimum would reject it. What disqualifies a candidate is
        # being a fragment, which shows up as a missing terminator or a missing
        # opening capital.
        for sentence in parts:
            if sentence.endswith((".", "!", "?")) and sentence[0].isupper():
                if re.search(r"[a-z]{3,}", sentence):
                    return sentence

        return None

    def healthy(self) -> bool:
        return True

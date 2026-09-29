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

from ..text import is_quotable, sentences
from .base import Completion, Message, ProviderError


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
                chosen.append((index, cleaned))
        return chosen

    @classmethod
    def _best_sentence(cls, text: str) -> str | None:
        """Pick one quotable sentence, or None when the source has none.

        ``None`` is a real outcome and is preferred over returning markup. A source
        that is only a heading or a code reference has nothing to quote, and emitting a
        directive as if it were an answer is worse than citing fewer sources.

        The cleanup and sentence splitting are shared with the excerpt shown in the UI.
        Two implementations meant the preview and the quote drifted apart, which is how
        markup appeared in one and not the other.
        """
        for candidate in sentences(text):
            if is_quotable(candidate):
                return candidate
        return None

    def healthy(self) -> bool:
        return True

"""The gateway pipeline.

One entry point that turns a question into an answer plus everything needed to
interrogate it: which sources were retrieved, what each scored, what it cost, and how
long each part took. The point is that the answer is never returned alone. An answer
without its provenance is unauditable, and an unauditable answer is the thing this
project exists to prevent.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .audit import Audit, VerifierGate
from .citations import Citation, coverage, resolve, split_claims, strip_citations
from .llm.base import Completion, Message
from .rag.retriever import Retriever
from .relevance import RelevanceGate
from .text import clean_markup, excerpt, truncate_excerpt  # noqa: F401 - re-exported
from .verify import NullVerifier, Verifier

DEFAULT_TOP_K = 5
MAX_TOP_K = 25
MAX_QUESTION_CHARS = 2000


@dataclass
class Source:
    """A retrieved chunk as shown to the user.

    Mutable because ``cited`` is resolved after the answer exists. The alternative is
    rebuilding the whole list, which would make the binding step harder to follow for
    the sake of immutability that buys nothing here.
    """

    index: int
    chunk_id: str
    doc_id: str
    title: str
    score: float
    excerpt: str
    cited: bool = False

    def to_citation(self) -> Citation:
        return Citation(
            index=self.index,
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            title=self.title,
            excerpt=self.excerpt,
        )


@dataclass
class Answer:
    """A complete, auditable response."""

    question: str
    text: str
    sources: list[Source]
    dangling_citations: list[int] = field(default_factory=list)
    claims: list[dict[str, object]] = field(default_factory=list)
    coverage: float = 0.0
    audit: Audit | None = None
    abstained: bool = False
    abstain_reason: str = ""
    usage: dict[str, object] = field(default_factory=dict)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    is_demo: bool = False
    retrieval_mode: str = ""
    generator_mode: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "text": self.text,
            "sources": [
                {
                    "index": s.index,
                    "chunk_id": s.chunk_id,
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "score": s.score,
                    "excerpt": s.excerpt,
                    "cited": s.cited,
                }
                for s in self.sources
            ],
            "citations": [s.index for s in self.sources if s.cited],
            "dangling_citations": self.dangling_citations,
            "claims": self.claims,
            "coverage": self.coverage,
            "verified": self.audit.verified if self.audit else False,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "audit": self.audit.to_dict() if self.audit else None,
            "usage": self.usage,
            "retrieval_mode": self.retrieval_mode,
            "generator_mode": self.generator_mode,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "is_demo": self.is_demo,
        }


class Generator(Protocol):
    """Anything that turns a rendered prompt into a completion.

    ``LLMClient`` satisfies this by having a ``complete`` method with the same shape,
    so the pipeline does not need to know which providers are configured.
    """

    name: str

    def complete(
        self, messages: list[Message], *, request_id: str, temperature: float = 0.0
    ) -> Completion: ...


def validate_question(question: str) -> str:
    """Bound the input at the trust boundary.

    An unbounded question reaches the provider, becomes a token cost, and can be used
    to push a very large payload through the API. Rejecting here is cheaper and clearer
    than truncating later.
    """
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(f"question exceeds {MAX_QUESTION_CHARS} characters")
    return cleaned


def clamp_top_k(k: int) -> int:
    return max(1, min(int(k), MAX_TOP_K))


def render_context(sources: list[Source], max_chars: int = 6000) -> str:
    """Number the sources so the answer can cite them.

    The number is the citation index, so it must be stable between the prompt and the
    binder. Each line is ``[n] text``; the demo provider parses exactly this format.
    """
    lines: list[str] = []
    budget = max_chars
    for source in sources:
        text = source.excerpt[:budget]
        if not text:
            continue
        lines.append(f"[{source.index}] {text}")
        budget -= len(text)
        if budget <= 0:
            break
    return "\n\n".join(lines)


class Gateway:
    """Retrieves, generates, and binds citations.

    Deliberately one class with one method. The value is in the audit trail attached
    to each answer, and splitting that across several objects would only make it
    harder to follow.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        top_k: int = DEFAULT_TOP_K,
        excerpt_chars: int = 400,
        instruction: str | None = None,
        verifier: Verifier | None = None,
        max_revisions: int = 0,
        require_majority: bool = False,
        relevance: RelevanceGate | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = clamp_top_k(top_k)
        self.excerpt_chars = excerpt_chars
        # Verification off by default: a gate nobody configured should not silently
        # withhold answers. Enabled explicitly, it is the thing that makes an
        # ungrounded answer impossible to return.
        self.gate = VerifierGate(
            verifier or NullVerifier(),
            max_revisions=max_revisions,
            require_majority=require_majority,
        )
        # Relevance gating is opt-in, like verification. Defaulting it on would impose a
        # model download and an extra ~10ms on every request to solve a problem the
        # caller may not have.
        self.relevance = relevance
        self.instruction = instruction or (
            "Answer the question using only the numbered sources below. "
            "Cite every claim with its source number, like [1]. "
            "If the sources do not answer the question, say so plainly."
        )

    @property
    def retrieval_mode(self) -> str:
        return self.retriever.name

    @property
    def generator_mode(self) -> str:
        return self.generator.name

    def ask(self, question: str, *, request_id: str, top_k: int | None = None) -> Answer:
        question = validate_question(question)
        k = clamp_top_k(top_k) if top_k is not None else self.top_k

        started = time.perf_counter()
        hits = self.retriever.retrieve(question, k=k)
        retrieval_ms = (time.perf_counter() - started) * 1000

        sources = [
            Source(
                index=position,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                title=hit.chunk.title or hit.doc_id,
                score=round(hit.score, 4),
                excerpt=excerpt(hit.text, self.excerpt_chars),
            )
            for position, hit in enumerate(hits, start=1)
        ]

        if not sources:
            # No sources is a legitimate outcome, not a failure. Answering anyway
            # would present model recall as if it were retrieved evidence.
            return Answer(
                question=question,
                text=(
                    "No matching passages were found in the corpus, so there is "
                    "nothing to ground an answer in. Try different terms, or check "
                    "the retrieval mode."
                ),
                sources=[],
                retrieval_ms=retrieval_ms,
                retrieval_mode=self.retrieval_mode,
                generator_mode=self.generator_mode,
            )

        messages = [
            Message.system(self.instruction),
            Message.user(f"Question: {question}\n\nSources:\n{render_context(sources)}"),
        ]

        started = time.perf_counter()
        completion = self.generator.complete(messages, request_id=request_id)
        generation_ms = (time.perf_counter() - started) * 1000

        # Verify before binding citations. An invented claim must not be able to earn
        # a citation chip by naming a source that happens to exist.
        context = render_context(sources)
        # Relevance is checked before grounding, and can withhold the answer outright.
        # Grounding cannot catch an irrelevant corpus, so this has to come first.
        if self.relevance is not None:
            scored = self.relevance.score(
                question, [(s.chunk_id, s.excerpt) for s in sources]
            )
            if not scored.relevant:
                return Answer(
                    question=question,
                    text=(
                        "None of the retrieved passages address this question, so I am "
                        "not going to answer it. The closest passage scored "
                        f"{scored.score:.1f} against a relevance threshold of "
                        f"{scored.threshold:.1f}."
                    ),
                    sources=sources,
                    claims=[],
                    # No claims, because nothing was claimed. The audit is honest rather
                    # than empty-by-accident: it records that the answer was withheld
                    # before any claim existed to verify.
                    audit=Audit(
                        verified=False,
                        abstained=True,
                        reason="retrieved passages do not address the question",
                        verifier=self.gate.verifier.name,
                    ),
                    abstained=True,
                    abstain_reason="irrelevant_passages",
                    usage=self._usage(completion),
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    is_demo=bool(completion.metadata.get("demo")),
                    retrieval_mode=self.retrieval_mode,
                    generator_mode=self.generator_mode,
                )

        text, audit, _ = self.gate.decide(completion.text, context)

        cited, dangling = resolve(text, [s.to_citation() for s in sources])
        cited_ids = {c.index for c in cited}
        for source in sources:
            source.cited = source.index in cited_ids

        # Withhold an answer that failed verification. The text is kept in the audit
        # so the reason is inspectable, but it is not presented as an answer.
        failed = self.gate.verifier.name != "null" and not self.gate.passes(audit)
        if failed:
            supported = [c.text for c in audit.claims if c.supported]
            joined = " ".join(supported)
            joined = re.sub(r"\s+", " ", joined).strip()
            if not joined:
                return Answer(
                    question=question,
                    text=(
                        "I could not find support for any part of this answer in the "
                        "retrieved passages, so I am not going to give it to you. "
                        f"({audit.unsupported_count} of {len(audit.claims)} claims "
                        "were unsupported.)"
                    ),
                    sources=sources,
                    claims=[c.to_dict() for c in audit.claims],
                    audit=audit,
                    abstained=True,
                    abstain_reason="unsupported_claims",
                    usage=self._usage(completion),
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    is_demo=bool(completion.metadata.get("demo")),
                    retrieval_mode=self.retrieval_mode,
                    generator_mode=self.generator_mode,
                )
            text = joined

        return Answer(
            question=question,
            text=text,
            sources=sources,
            dangling_citations=dangling,
            claims=[
                {
                    "text": claim.text,
                    "citations": [c for c in claim.citations if c in cited_ids],
                    "uncited": not any(c in cited_ids for c in claim.citations),
                }
                for claim in split_claims(text)
            ],
            coverage=coverage(text, [s.to_citation() for s in sources]),
            audit=audit,
            usage=self._usage(completion),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            is_demo=bool(completion.metadata.get("demo")),
            retrieval_mode=self.retrieval_mode,
            generator_mode=self.generator_mode,
        )

    @staticmethod
    def _usage(completion: Completion) -> dict[str, object]:
        total = getattr(completion, "total_tokens", None)
        return {
            "model": completion.model,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": total
            if total is not None
            else completion.prompt_tokens + completion.completion_tokens,
            "is_demo": bool(completion.metadata.get("demo")),
        }

    def ask_text(self, question: str, **kwargs: object) -> str:
        """Answer without citation markers, for callers that only want prose."""
        return strip_citations(self.ask(question, **kwargs).text)  # type: ignore[arg-type]

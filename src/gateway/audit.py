"""The verification gate.

A grounded answer passes through. An answer with unsupported claims is either revised
or withheld, and the reason is always recorded. The system never returns a confident
answer it cannot support, and never silently drops the check that would have caught
the failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from .citations import split_claims
from .llm.base import Message
from .verify import Verifier

# Why an answer was withheld. Distinct outcomes are recorded separately because they
# need different responses: a bad prompt, a bad retriever, and a corpus gap are three
# different problems.
AbstainReason = Literal["unsupported_claims", "no_context", "revision_exhausted"]


@dataclass
class ClaimAudit:
    """The verification result for one claim."""

    text: str
    supported: bool
    score: float
    reason: str
    citations: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "supported": self.supported,
            "score": round(self.score, 4),
            "reason": self.reason,
            "citations": list(self.citations),
        }


@dataclass
class Audit:
    """Everything the UI needs to show how an answer was checked."""

    verified: bool
    abstained: bool = False
    reason: str = ""
    claims: list[ClaimAudit] = field(default_factory=list)
    verifier: str = "null"
    revisions: int = 0
    elapsed_ms: float = 0.0

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.supported)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for c in self.claims if not c.supported)

    @property
    def support_rate(self) -> float:
        return self.supported_count / len(self.claims) if self.claims else 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "abstained": self.abstained,
            "reason": self.reason,
            "verifier": self.verifier,
            "revisions": self.revisions,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "supported": self.supported_count,
            "unsupported": self.unsupported_count,
            "support_rate": round(self.support_rate, 4),
            "claims": [c.to_dict() for c in self.claims],
        }


class VerifierGate:
    """Audits an answer and decides whether to release it.

    ``revise`` is optional. When supplied and a claim fails, the verifier asks for a
    corrected answer using only the supported material, which recovers an answer that
    was mostly right instead of discarding it over one invented sentence.
    """

    def __init__(
        self,
        verifier: Verifier,
        *,
        max_revisions: int = 1,
        require_majority: bool = False,
    ):
        self.verifier = verifier
        self.max_revisions = max(0, max_revisions)
        # Majority mode is the lenient setting: one bad sentence does not void a
        # correct answer. Strict mode requires every claim, which is the right default
        # when a fabricated sentence is worse than no answer at all.
        self.require_majority = require_majority

    def audit(self, answer: str, context: str) -> Audit:
        started = time.perf_counter()
        claims = split_claims(answer)

        results: list[ClaimAudit] = []
        for claim in claims:
            verdict = self.verifier.check(claim.text, context)
            results.append(
                ClaimAudit(
                    text=claim.text,
                    supported=verdict.supported,
                    score=verdict.score,
                    reason=verdict.reason,
                    citations=claim.citations,
                )
            )

        return Audit(
            verified=bool(results),
            claims=results,
            verifier=self.verifier.name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def passes(self, audit: Audit) -> bool:
        if not audit.claims:
            return False
        if self.require_majority:
            return audit.support_rate > 0.5
        return audit.unsupported_count == 0

    def decide(self, answer: str, context: str) -> tuple[str, Audit, int]:
        """Return the answer to release, its audit, and how many revisions it took.

        The answer is returned unchanged even when it fails. The caller decides
        whether to withhold it; this function's job is to say honestly whether the
        text is grounded, not to edit it.
        """
        attempt = 0
        audit = self.audit(answer, context)

        while not self.passes(audit) and attempt < self.max_revisions:
            attempt += 1
            revised = self._request_revision(answer, audit)
            if revised is None:
                break
            audit = self.audit(revised, context)
            answer = revised

        audit.revisions = attempt
        return answer, audit, attempt

    def _request_revision(self, answer: str, audit: Audit) -> str | None:
        """Ask for a corrected answer. Returns None when no reviser is configured.

        Kept as a separate method so the decision logic above is testable without a
        generator in the loop.
        """
        return None


def revision_prompt(answer: str, audit: Audit, context: str) -> list[Message]:
    """Build the correction request for a reviser.

    Exposed as a function so the prompt is inspectable and testable, rather than being
    assembled inside the decision loop where it cannot be.
    """
    unsupported = [c.text for c in audit.claims if not c.supported]
    listed = "\n".join(f"- {text}" for text in unsupported)
    return [
        Message.system(
            "You are correcting an answer. Remove every claim that the sources do not "
            "support. Do not add new claims. If a claim cannot be supported, drop it."
        ),
        Message.user(
            f"Sources:\n{context}\n\n"
            f"Original answer:\n{answer}\n\n"
            f"Unsupported claims to remove:\n{listed}\n\n"
            "Rewrite the answer using only supported claims, citing sources as [n]."
        ),
    ]

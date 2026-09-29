"""Verification and abstention.

The central claim under test: a system that cannot support an answer should not return
it. These cases are written around that, including the failure modes that make a
verifier worse than useless — a gate that passes everything, or one that fails
everything.
"""

from __future__ import annotations

import pytest

from gateway.audit import VerifierGate, revision_prompt
from gateway.citations import split_claims
from gateway.llm.base import Completion
from gateway.pipeline import Gateway
from gateway.rag import BM25Retriever, Chunk
from gateway.verify import NullVerifier, OverlapVerifier

CONTEXT = (
    "The status code can be set with the status_code parameter on the path operation. "
    "Dependency injection lets a route declare the dependencies it needs."
)


# --- the verifier contract


def test_overlap_accepts_a_grounded_claim() -> None:
    verdict = OverlapVerifier().check(
        "The status code can be set with the status_code parameter.", CONTEXT
    )

    assert verdict.supported
    assert verdict.score == pytest.approx(1.0)


def test_overlap_rejects_an_invented_claim() -> None:
    verdict = OverlapVerifier().check(
        "The status code is cached in Redis with a thirty second TTL.", CONTEXT
    )

    assert not verdict.supported
    assert "missing" in verdict.reason


def test_overlap_tolerates_a_paraphrase() -> None:
    """A verifier that fails every paraphrase is not a verifier, it is a string match."""
    verdict = OverlapVerifier().check(
        "Status codes are set through the status_code parameter.", CONTEXT
    )

    assert verdict.supported


def test_overlap_ignores_function_words() -> None:
    """Matching 'the' and 'is' is not evidence of anything."""
    verifier = OverlapVerifier()
    grounded = verifier.check("The status code parameter is used.", CONTEXT)
    invented = verifier.check("The response is a quantum tunnelling array.", CONTEXT)

    assert grounded.score > invented.score


def test_overlap_handles_empty_context() -> None:
    verdict = OverlapVerifier().check("Any claim at all.", "")

    assert not verdict.supported
    assert verdict.reason == "empty context"


def test_overlap_does_not_refute_a_claim_with_no_checkable_content() -> None:
    """A claim made only of function words cannot be checked, so it must not be
    refuted. Failing these would break every answer that contains "it does that"."""
    verdict = OverlapVerifier().check("It is not.", CONTEXT)

    assert verdict.supported
    assert verdict.reason == "no content words to check"


def test_overlap_rejects_an_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        OverlapVerifier(threshold=0.0)

    with pytest.raises(ValueError, match="threshold"):
        OverlapVerifier(threshold=1.5)


def test_null_verifier_reports_everything_supported() -> None:
    """Reporting an unsupported verdict without measuring one is how a demo becomes
    a false claim."""
    verdict = NullVerifier().check("Wholly invented.", CONTEXT)

    assert verdict.supported
    assert verdict.reason == "verification disabled"


def test_overlap_satisfies_the_verifier_protocol() -> None:
    from gateway.verify import Verifier

    assert isinstance(OverlapVerifier(), Verifier)
    assert isinstance(NullVerifier(), Verifier)


# --- the gate


def test_gate_passes_a_grounded_answer() -> None:
    gate = VerifierGate(OverlapVerifier())

    assert gate.passes(gate.audit("The status code uses the status_code parameter.", CONTEXT))


def test_gate_fails_an_answer_containing_one_fabrication() -> None:
    """Partial failure is the case that matters: an answer can be right on one point
    and invented on another, and a single answer-level verdict hides that."""
    gate = VerifierGate(OverlapVerifier())
    audit = gate.audit(
        "The status code uses the status_code parameter. It is cached in Redis.", CONTEXT
    )

    assert not gate.passes(audit)
    assert audit.unsupported_count == 1
    assert audit.supported_count == 1


def test_gate_reports_per_claim_detail() -> None:
    gate = VerifierGate(OverlapVerifier())
    audit = gate.audit(
        "The status code uses the status_code parameter. It is cached in Redis.", CONTEXT
    )

    assert len(audit.claims) == 2
    assert audit.claims[0].supported
    assert not audit.claims[1].supported
    assert audit.claims[1].score < 1.0


def test_majority_mode_tolerates_one_bad_claim() -> None:
    answer = (
        "The status code uses the status_code parameter. "
        "Dependency injection declares dependencies. "
        "It is cached in Redis."
    )
    strict = VerifierGate(OverlapVerifier(), require_majority=False)
    lenient = VerifierGate(OverlapVerifier(), require_majority=True)

    assert not strict.passes(strict.audit(answer, CONTEXT))
    assert lenient.passes(lenient.audit(answer, CONTEXT))


def test_gate_fails_an_empty_answer() -> None:
    gate = VerifierGate(OverlapVerifier())

    assert not gate.passes(gate.audit("", CONTEXT))


def test_decide_counts_zero_revisions_without_a_reviser() -> None:
    gate = VerifierGate(OverlapVerifier(), max_revisions=2)

    _, audit, revisions = gate.decide("The status code uses status_code.", CONTEXT)

    assert revisions == 0
    assert audit.revisions == 0


def test_audit_serialises_with_every_field_the_ui_needs() -> None:
    import json

    gate = VerifierGate(OverlapVerifier())
    payload = gate.audit("The status code uses status_code. It is in Redis.", CONTEXT).to_dict()

    json.dumps(payload)
    assert {"verified", "verifier", "supported", "unsupported", "claims"} <= set(payload)


def test_revision_prompt_names_the_unsupported_claims() -> None:
    gate = VerifierGate(OverlapVerifier())
    audit = gate.audit("The status code uses status_code. It is cached in Redis.", CONTEXT)

    prompt = revision_prompt("answer text", audit, CONTEXT)

    assert "Redis" in prompt[-1].content
    assert "status_code" not in prompt[-1].content.split("Unsupported claims")[1]


# --- pipeline integration


def build(answer: str, **kwargs) -> Gateway:
    chunks = [
        Chunk(
            "c1",
            "status.md",
            "The status code can be set with the status_code parameter.",
            "Status",
        )
    ]
    generator = _Stub(answer)
    return Gateway(BM25Retriever.build(chunks), generator, **kwargs), generator


class _Stub:
    name = "stub"

    def __init__(self, text: str):
        self.text = text

    def complete(self, messages, *, request_id, temperature=0.0) -> Completion:
        return Completion(text=self.text, model="stub", prompt_tokens=5, completion_tokens=5)


def test_pipeline_drops_a_fabricated_claim_when_verifying() -> None:
    gateway, _ = build(
        "The status code uses the status_code parameter. "
        "It is cached in Redis with a TTL.",
        verifier=OverlapVerifier(),
    )

    result = gateway.ask("status code", request_id="r1")

    assert "Redis" not in result.text
    assert "status_code" in result.text
    assert result.abstained is False
    assert result.audit is not None
    assert result.audit.unsupported_count == 1


def test_pipeline_withholds_an_entirely_unsupported_answer() -> None:
    gateway, _ = build(
        "Kubernetes operators reconcile desired state across a cluster.",
        verifier=OverlapVerifier(),
    )

    result = gateway.ask("status code", request_id="r1")

    assert result.abstained is True
    assert result.abstain_reason == "unsupported_claims"
    assert "could not find support" in result.text.lower()
    assert "Kubernetes" not in result.text


def test_an_unverified_answer_is_not_withheld_by_default() -> None:
    """Verification off must not silently withhold answers: a gate nobody configured
    should not change behaviour."""
    gateway, _ = build("Kubernetes operators reconcile cluster state.")

    result = gateway.ask("status code", request_id="r1")

    assert result.abstained is False
    assert "Kubernetes" in result.text
    assert result.audit.verifier == "null"


def test_a_fabricated_claim_cannot_earn_a_citation_by_naming_a_source() -> None:
    """A claim that cites a real source but says something absent from it must still
    fail. Citation presence is not evidence."""
    gateway, _ = build(
        "The status code is cached in Redis [1].", verifier=OverlapVerifier()
    )

    result = gateway.ask("status code", request_id="r1")

    assert result.abstained is True
    assert all(not c.supported for c in result.audit.claims)


def test_release_text_claims_match_the_verified_text() -> None:
    """The claims panel reads the released text. If it were built from the pre-verification
    answer it would show a claim the user never received."""
    gateway, _ = build(
        "The status code uses the status_code parameter. It is cached in Redis.",
        verifier=OverlapVerifier(),
    )

    result = gateway.ask("status code", request_id="r1")

    assert "Redis" not in " ".join(c["text"] for c in result.claims)


def test_to_dict_reports_the_audit() -> None:
    gateway, _ = build("The status code uses status_code.", verifier=OverlapVerifier())

    payload = gateway.ask("status code", request_id="r1").to_dict()

    assert payload["verified"] is True
    assert payload["abstained"] is False
    assert payload["audit"]["verifier"] == "overlap"


def test_no_context_still_refuses_before_verification() -> None:
    """An answer with no retrieved evidence is refused at retrieval time, not rescued
    by a verifier."""
    from gateway.rag import NullRetriever

    gateway = Gateway(NullRetriever(), _Stub("Anything."), verifier=OverlapVerifier())
    result = gateway.ask("anything", request_id="r1")

    assert result.sources == []
    assert "no matching passages" in result.text.lower()


def test_nli_verifier_is_importable_without_the_ml_stack() -> None:
    """Importing the module must not require transformers, or the whole gateway
    becomes unimportable on a clean clone."""
    import gateway.verify as module

    assert hasattr(module, "NliVerifier")


def test_claims_survive_a_revision_free_path() -> None:
    answer = "The status code uses the status_code parameter."

    assert len(split_claims(answer)) == 1

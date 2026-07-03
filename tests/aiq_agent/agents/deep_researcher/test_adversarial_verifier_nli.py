# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NLI pre-filter cascade in the adversarial verifier.

The NLI model itself is mocked at the ``nli_filter`` module boundary; these
tests cover the cascade routing (auto-support vs LLM fallback), the audit
trail in the verification report, and fail-open behavior.
"""

import json

import pytest

from aiq_agent.agents.deep_researcher.adversarial_verifier import SelectedClaim
from aiq_agent.agents.deep_researcher.adversarial_verifier import build_verification_report
from aiq_agent.agents.deep_researcher.adversarial_verifier import nli_prefilter_claims
from aiq_agent.agents.deep_researcher.adversarial_verifier import nli_support_threshold
from aiq_agent.agents.deep_researcher.adversarial_verifier import run_adversarial_verification
from aiq_agent.common import nli_filter


@pytest.fixture(autouse=True)
def _enable_nli_prefilter(monkeypatch):
    """Opt back in: the directory conftest disables NLI for hermeticity; these
    tests exercise the cascade with the scorer mocked (never a real model)."""
    monkeypatch.setenv("AIQ_NLI_ENABLED", "1")


def _scores(entailment: float) -> dict:
    remainder = (1.0 - entailment) / 2
    return {"entailment": entailment, "neutral": remainder, "contradiction": remainder}


def _claim(claim_id: str, text: str, evidence: list[str] | None = None) -> SelectedClaim:
    return SelectedClaim(
        claim_id=claim_id,
        claim_text=text,
        claim_type="quantitative",
        evidence=evidence if evidence is not None else [f"[https://example.com/{claim_id}] {text}"],
    )


def _patch_scores(monkeypatch, per_pair_scores):
    """Patch nli_filter.score_entailment_batch with queued per-pair scores."""
    queue = list(per_pair_scores)
    captured: list[list[tuple[str, str]]] = []

    def _fake_batch(pairs):
        captured.append(list(pairs))
        return [queue.pop(0) for _ in pairs]

    monkeypatch.setattr(nli_filter, "score_entailment_batch", _fake_batch)
    return captured


class _MockResponse:
    def __init__(self, content):
        self.content = content


class _MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return _MockResponse(self._responses.pop(0))


class TestThresholdKnob:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("AIQ_NLI_SUPPORT_THRESHOLD", raising=False)
        assert nli_support_threshold() == 0.90

    def test_override_and_clamping(self, monkeypatch):
        monkeypatch.setenv("AIQ_NLI_SUPPORT_THRESHOLD", "0.95")
        assert nli_support_threshold() == 0.95
        monkeypatch.setenv("AIQ_NLI_SUPPORT_THRESHOLD", "0.1")
        assert nli_support_threshold() == 0.5
        monkeypatch.setenv("AIQ_NLI_SUPPORT_THRESHOLD", "junk")
        assert nli_support_threshold() == 0.90


class TestNliPrefilterClaims:
    def test_high_entailment_resolves_supported(self, monkeypatch):
        _patch_scores(monkeypatch, [_scores(0.97)])
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B")])
        assert remaining == []
        assert len(resolved) == 1
        assert resolved[0]["verdict"] == "supported"
        assert resolved[0]["verifier_method"] == "nli"
        assert resolved[0]["confidence"] == 0.97
        assert resolved[0]["note"] == "nli_entailment"

    def test_low_entailment_goes_to_llm(self, monkeypatch):
        _patch_scores(monkeypatch, [_scores(0.60)])
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B")])
        assert resolved == []
        assert [c.claim_id for c in remaining] == ["c1"]

    def test_high_contradiction_still_goes_to_llm(self, monkeypatch):
        # NLI contradictions are false-positive-prone on paraphrase: never
        # auto-reject, always let the LLM decide.
        _patch_scores(monkeypatch, [{"entailment": 0.01, "neutral": 0.01, "contradiction": 0.98}])
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B")])
        assert resolved == []
        assert [c.claim_id for c in remaining] == ["c1"]

    def test_max_entailment_across_multiple_extracts(self, monkeypatch):
        _patch_scores(monkeypatch, [_scores(0.30), _scores(0.95)])
        claim = _claim("c1", "Revenue was $35.1B", evidence=["[u1] weak extract", "[u2] strong extract"])
        resolved, remaining = nli_prefilter_claims([claim])
        assert remaining == []
        assert resolved[0]["confidence"] == 0.95

    def test_url_prefix_stripped_from_premise(self, monkeypatch):
        captured = _patch_scores(monkeypatch, [_scores(0.99)])
        claim = _claim("c1", "Revenue was $35.1B", evidence=["[https://x.example.com/a] The extract text."])
        nli_prefilter_claims([claim])
        assert captured[0] == [("The extract text.", "Revenue was $35.1B")]

    def test_none_scores_fall_back_to_llm(self, monkeypatch):
        _patch_scores(monkeypatch, [None])
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B")])
        assert resolved == []
        assert [c.claim_id for c in remaining] == ["c1"]

    def test_claims_without_evidence_pass_through(self, monkeypatch):
        _patch_scores(monkeypatch, [])
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B", evidence=[])])
        assert resolved == []
        assert [c.claim_id for c in remaining] == ["c1"]

    def test_disabled_sends_everything_to_llm(self, monkeypatch):
        monkeypatch.setenv("AIQ_NLI_ENABLED", "0")

        def _boom(pairs):
            raise AssertionError("scoring must not run when disabled")

        monkeypatch.setattr(nli_filter, "score_entailment_batch", _boom)
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B")])
        assert resolved == []
        assert [c.claim_id for c in remaining] == ["c1"]

    def test_scoring_exception_fails_open(self, monkeypatch):
        def _boom(pairs):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(nli_filter, "score_entailment_batch", _boom)
        resolved, remaining = nli_prefilter_claims([_claim("c1", "Revenue was $35.1B")])
        assert resolved == []
        assert [c.claim_id for c in remaining] == ["c1"]

    def test_empty_input(self):
        assert nli_prefilter_claims([]) == ([], [])


class TestReportAuditTrail:
    def test_verifier_method_included_only_for_nli_verdicts(self):
        verdicts = [
            {"claim_id": "c1", "verdict": "supported", "confidence": 0.97, "note": "nli_entailment",
             "verifier_method": "nli"},
            {"claim_id": "c2", "verdict": "contradicted", "confidence": 0.8, "note": "wrong year"},
        ]
        report = build_verification_report(verdicts, tier="deeper")
        assert report["claims"][0]["verifier_method"] == "nli"
        assert "verifier_method" not in report["claims"][1]
        assert report["summary"]["supported"] == 1
        assert report["summary"]["contradicted"] == 1


class TestCascadeEndToEnd:
    async def test_nli_resolved_claims_skip_llm(self, monkeypatch):
        table = json.dumps(
            {
                "atomic_claims": [
                    {
                        "claim_id": "c1",
                        "claim_text": "Revenue was $35.1B in Q3",
                        "claim_type": "quantitative",
                        "status": "verified",
                        "evidence": [{"source_url": "https://e.example.com/1", "extract": "Revenue was $35.1B."}],
                    },
                    {
                        "claim_id": "c2",
                        "claim_text": "Revenue was $99B in Q4",
                        "claim_type": "quantitative",
                        "status": "verified",
                        "evidence": [{"source_url": "https://e.example.com/2", "extract": "Revenue was $35.1B."}],
                    },
                ]
            }
        )
        # c1 clearly entailed; c2 ambiguous → LLM.
        _patch_scores(monkeypatch, [_scores(0.98), _scores(0.20)])
        llm = _MockLLM(['{"verdict": "contradicted", "confidence": 0.85, "note": "extract says $35.1B"}'])

        outcome = await run_adversarial_verification(llm=llm, claim_table_content=table, tier="deeper")

        assert outcome is not None
        assert len(llm.calls) == 1  # only c2 hit the LLM
        assert outcome.summary == {
            "supported": 1,
            "partially_supported": 0,
            "contradicted": 1,
            "not_addressed": 0,
        }
        by_id = {entry["claim_id"]: entry for entry in outcome.report["claims"]}
        assert by_id["c1"]["verifier_method"] == "nli"
        assert "verifier_method" not in by_id["c2"]
        # Claim-table downgrade still applies to the LLM-contradicted claim.
        updated = json.loads(outcome.updated_claim_table_json)
        statuses = {c["claim_id"]: c["status"] for c in updated["atomic_claims"]}
        assert statuses == {"c1": "verified", "c2": "unverified"}

    async def test_all_claims_resolved_by_nli_makes_zero_llm_calls(self, monkeypatch):
        table = json.dumps(
            {
                "atomic_claims": [
                    {
                        "claim_id": "c1",
                        "claim_text": "Revenue was $35.1B in Q3",
                        "claim_type": "quantitative",
                        "status": "verified",
                        "evidence": [{"source_url": "https://e.example.com/1", "extract": "Revenue was $35.1B."}],
                    }
                ]
            }
        )
        _patch_scores(monkeypatch, [_scores(0.99)])
        llm = _MockLLM([])

        outcome = await run_adversarial_verification(llm=llm, claim_table_content=table, tier="deeper")

        assert outcome is not None
        assert llm.calls == []
        assert outcome.summary["supported"] == 1

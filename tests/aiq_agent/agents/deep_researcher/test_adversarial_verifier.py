# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the adversarial claim verifier."""

import asyncio
import json

import pytest

from aiq_agent.agents.deep_researcher.adversarial_verifier import SelectedClaim
from aiq_agent.agents.deep_researcher.adversarial_verifier import apply_verdicts_to_claim_table
from aiq_agent.agents.deep_researcher.adversarial_verifier import build_verification_report
from aiq_agent.agents.deep_researcher.adversarial_verifier import claim_risk_score
from aiq_agent.agents.deep_researcher.adversarial_verifier import is_high_risk_claim
from aiq_agent.agents.deep_researcher.adversarial_verifier import parse_verdict_payload
from aiq_agent.agents.deep_researcher.adversarial_verifier import run_adversarial_verification
from aiq_agent.agents.deep_researcher.adversarial_verifier import select_high_risk_claims
from aiq_agent.agents.deep_researcher.adversarial_verifier import verifier_concurrency
from aiq_agent.agents.deep_researcher.adversarial_verifier import verifier_enabled
from aiq_agent.agents.deep_researcher.adversarial_verifier import verifier_max_claims
from aiq_agent.agents.deep_researcher.adversarial_verifier import verify_claim
from aiq_agent.agents.deep_researcher.adversarial_verifier import verify_claims


def _claim_table(claims: list[dict]) -> str:
    return json.dumps({"atomic_claims": claims})


def _atomic_claim(
    claim_id: str,
    claim_text: str,
    *,
    claim_type: str = "quantitative",
    status: str = "verified",
    extract: str = "NVIDIA reported $35.1B revenue in Q3 FY2025.",
) -> dict:
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_type": claim_type,
        "status": status,
        "evidence": [
            {
                "source_url": "https://example.com/earnings",
                "source_class": "primary_issuer",
                "extract": extract,
            }
        ],
    }


class _MockResponse:
    def __init__(self, content):
        self.content = content


class _MockLLM:
    """Mock LLM whose ainvoke returns queued raw responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self._responses:
            return _MockResponse('{"verdict": "not_addressed", "confidence": 0.0, "note": ""}')
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _MockResponse(response)


class _SlowLLM:
    async def ainvoke(self, messages):
        await asyncio.sleep(5)
        return _MockResponse('{"verdict": "supported", "confidence": 0.9, "note": "late"}')


class TestEnvKnobs:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AIQ_ADVERSARIAL_VERIFIER_ENABLED", raising=False)
        assert verifier_enabled() is True

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AIQ_ADVERSARIAL_VERIFIER_ENABLED", "0")
        assert verifier_enabled() is False

    def test_max_claims_default_and_override(self, monkeypatch):
        monkeypatch.delenv("AIQ_VERIFIER_MAX_CLAIMS", raising=False)
        assert verifier_max_claims() == 24
        monkeypatch.setenv("AIQ_VERIFIER_MAX_CLAIMS", "5")
        assert verifier_max_claims() == 5
        monkeypatch.setenv("AIQ_VERIFIER_MAX_CLAIMS", "junk")
        assert verifier_max_claims() == 24

    def test_concurrency_default(self, monkeypatch):
        monkeypatch.delenv("AIQ_VERIFIER_CONCURRENCY", raising=False)
        assert verifier_concurrency() == 6


class TestClaimSelection:
    def test_high_risk_by_type(self):
        assert is_high_risk_claim("Revenue grew", "quantitative")
        assert is_high_risk_claim("X causes Y", "causal")
        assert not is_high_risk_claim("Some general statement", "other")

    def test_high_risk_by_text_patterns(self):
        assert is_high_risk_claim("Revenue was $35.1 billion", "other")
        assert is_high_risk_claim("Adoption grew 45% year over year", "other")
        assert is_high_risk_claim("Founded in 1993", "other")
        assert is_high_risk_claim("The largest GPU maker", "other")
        assert is_high_risk_claim("It was the first product of its kind", "other")
        assert not is_high_risk_claim("The product focuses on usability", "other")

    def test_selection_filters_and_caps(self):
        claims = [
            _atomic_claim("c1", "Revenue was $35.1B in Q3"),
            _atomic_claim("c2", "The largest AI chip vendor", claim_type="other"),
            _atomic_claim("c3", "Generally well regarded product", claim_type="other"),
            _atomic_claim("c4", "Founded in 1993", claim_type="other"),
        ]
        selected = select_high_risk_claims(_claim_table(claims), max_claims=2)
        assert len(selected) == 2
        ids = {claim.claim_id for claim in selected}
        assert "c3" not in ids
        # Highest risk first: c1 has $ + numbers + risky type
        assert selected[0].claim_id == "c1"

    def test_selection_pulls_packet_evidence(self):
        claims = [_atomic_claim("c1", "Revenue was $35.1B in Q3")]
        packet = json.dumps(
            {
                "sources": [
                    {
                        "url": "https://example.com/press",
                        "extracts": [{"text": "Q3 revenue totaled $35.1 billion.", "claim_ids": ["c1"]}],
                    }
                ]
            }
        )
        selected = select_high_risk_claims(_claim_table(claims), evidence_packet_content=packet)
        assert len(selected) == 1
        assert any("press" in snippet for snippet in selected[0].evidence)

    def test_invalid_claim_table_selects_nothing(self):
        assert select_high_risk_claims("not json") == []
        assert select_high_risk_claims("") == []

    def test_risk_score_ordering(self):
        money = claim_risk_score("Revenue hit $35B", "quantitative")
        plain = claim_risk_score("Includes 3 modes", "other")
        assert money > plain


class TestVerdictParsing:
    def test_parses_strict_json(self):
        parsed = parse_verdict_payload('{"verdict": "supported", "confidence": 0.9, "note": "matches"}')
        assert parsed == {"verdict": "supported", "confidence": 0.9, "note": "matches"}

    def test_parses_json_in_code_fence(self):
        raw = 'Here you go:\n```json\n{"verdict": "contradicted", "confidence": 0.8, "note": "wrong year"}\n```'
        parsed = parse_verdict_payload(raw)
        assert parsed["verdict"] == "contradicted"

    def test_bad_verdict_degrades_to_not_addressed(self):
        parsed = parse_verdict_payload('{"verdict": "definitely_true", "confidence": 1.0}')
        assert parsed["verdict"] == "not_addressed"

    def test_unparseable_degrades_to_not_addressed(self):
        parsed = parse_verdict_payload("no json here at all")
        assert parsed["verdict"] == "not_addressed"
        assert parsed["confidence"] == 0.0

    def test_confidence_clamped(self):
        parsed = parse_verdict_payload('{"verdict": "supported", "confidence": 17}')
        assert parsed["confidence"] == 1.0
        parsed = parse_verdict_payload('{"verdict": "supported", "confidence": "abc"}')
        assert parsed["confidence"] == 0.5


class TestVerifyClaim:
    @pytest.mark.asyncio
    async def test_no_evidence_short_circuits(self):
        llm = _MockLLM([])
        claim = SelectedClaim(claim_id="c1", claim_text="X is 5", evidence=[])
        verdict = await verify_claim(llm, claim)
        assert verdict["verdict"] == "not_addressed"
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_llm_exception_degrades(self):
        llm = _MockLLM([RuntimeError("boom")])
        claim = SelectedClaim(claim_id="c1", claim_text="X is 5", evidence=["[u] X is 5"])
        verdict = await verify_claim(llm, claim)
        assert verdict["verdict"] == "not_addressed"

    @pytest.mark.asyncio
    async def test_timeout_degrades(self):
        claim = SelectedClaim(claim_id="c1", claim_text="X is 5", evidence=["[u] X is 5"])
        verdict = await verify_claim(_SlowLLM(), claim, timeout=0.05)
        assert verdict["verdict"] == "not_addressed"

    @pytest.mark.asyncio
    async def test_bounded_concurrency_runs_all(self):
        llm = _MockLLM(['{"verdict": "supported", "confidence": 0.9, "note": "ok"}'] * 4)
        claims = [
            SelectedClaim(claim_id=f"c{i}", claim_text=f"value is {i}%", evidence=["[u] extract"]) for i in range(4)
        ]
        verdicts = await verify_claims(llm, claims, concurrency=2)
        assert len(verdicts) == 4
        assert all(v["verdict"] == "supported" for v in verdicts)


class TestReportAndClaimTableUpdate:
    def test_report_summary_counts(self):
        verdicts = [
            {"claim_id": "c1", "verdict": "supported", "confidence": 0.9, "note": ""},
            {"claim_id": "c2", "verdict": "contradicted", "confidence": 0.8, "note": "wrong"},
            {"claim_id": "c3", "verdict": "not_addressed", "confidence": 0.0, "note": ""},
        ]
        report = build_verification_report(verdicts, tier="deeper")
        assert report["summary"] == {
            "supported": 1,
            "partially_supported": 0,
            "contradicted": 1,
            "not_addressed": 1,
        }
        assert len(report["claims"]) == 3

    def test_contradicted_claim_downgraded(self):
        table = _claim_table([_atomic_claim("c1", "Revenue was $99B in Q3")])
        verdicts = [{"claim_id": "c1", "verdict": "contradicted", "confidence": 0.9, "note": "actual was $35.1B"}]
        updated = apply_verdicts_to_claim_table(table, verdicts)
        assert updated is not None
        payload = json.loads(updated)
        claim = payload["atomic_claims"][0]
        assert claim["status"] == "unverified"
        assert claim["verification_verdict"] == "contradicted"
        assert "actual was $35.1B" in claim["verification_note"]
        assert claim["hedge_required"] is True

    def test_supported_claim_keeps_status(self):
        table = _claim_table([_atomic_claim("c1", "Revenue was $35.1B in Q3")])
        verdicts = [{"claim_id": "c1", "verdict": "supported", "confidence": 0.95, "note": "matches"}]
        updated = apply_verdicts_to_claim_table(table, verdicts)
        payload = json.loads(updated)
        claim = payload["atomic_claims"][0]
        assert claim["status"] == "verified"
        assert claim["verification_verdict"] == "supported"

    def test_invalid_table_returns_none(self):
        assert apply_verdicts_to_claim_table("not json", [{"claim_id": "c1", "verdict": "supported"}]) is None

    def test_no_matching_claims_returns_none(self):
        table = _claim_table([_atomic_claim("c1", "Revenue was $35.1B")])
        assert apply_verdicts_to_claim_table(table, [{"claim_id": "zz", "verdict": "supported"}]) is None


class TestRunAdversarialVerification:
    @pytest.mark.asyncio
    async def test_full_pass_writes_report_and_updates_table(self):
        table = _claim_table(
            [
                _atomic_claim("c1", "Revenue was $35.1B in Q3"),
                _atomic_claim("c2", "Revenue was $99B in Q4"),
            ]
        )
        llm = _MockLLM(
            [
                '{"verdict": "supported", "confidence": 0.95, "note": "matches extract"}',
                '{"verdict": "contradicted", "confidence": 0.85, "note": "extract says $35.1B"}',
            ]
        )
        outcome = await run_adversarial_verification(llm=llm, claim_table_content=table, tier="deeper")
        assert outcome is not None
        assert outcome.summary["supported"] + outcome.summary["contradicted"] == 2
        report = json.loads(outcome.report_json)
        assert {"summary", "claims"} <= set(report)
        assert outcome.updated_claim_table_json is not None
        updated = json.loads(outcome.updated_claim_table_json)
        statuses = {claim["claim_id"]: claim["status"] for claim in updated["atomic_claims"]}
        contradicted = [
            claim for claim in updated["atomic_claims"] if claim.get("verification_verdict") == "contradicted"
        ]
        assert len(contradicted) == 1
        assert statuses[contradicted[0]["claim_id"]] == "unverified"

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, monkeypatch):
        monkeypatch.setenv("AIQ_ADVERSARIAL_VERIFIER_ENABLED", "0")
        table = _claim_table([_atomic_claim("c1", "Revenue was $35.1B")])
        outcome = await run_adversarial_verification(llm=_MockLLM([]), claim_table_content=table, tier="deeper")
        assert outcome is None

    @pytest.mark.asyncio
    async def test_skips_shallow_tier(self):
        table = _claim_table([_atomic_claim("c1", "Revenue was $35.1B")])
        outcome = await run_adversarial_verification(llm=_MockLLM([]), claim_table_content=table, tier="shallow")
        assert outcome is None

    @pytest.mark.asyncio
    async def test_skips_without_llm(self):
        table = _claim_table([_atomic_claim("c1", "Revenue was $35.1B")])
        outcome = await run_adversarial_verification(llm=None, claim_table_content=table, tier="deeper")
        assert outcome is None

    @pytest.mark.asyncio
    async def test_skips_without_evidence_extracts(self):
        claims = [
            {
                "claim_id": "c1",
                "claim_text": "Revenue was $35.1B",
                "claim_type": "quantitative",
                "status": "unverified",
                "evidence": [],
            }
        ]
        outcome = await run_adversarial_verification(
            llm=_MockLLM([]),
            claim_table_content=_claim_table(claims),
            tier="deeper",
        )
        assert outcome is None

    @pytest.mark.asyncio
    async def test_skips_invalid_claim_table(self):
        outcome = await run_adversarial_verification(llm=_MockLLM([]), claim_table_content="nope", tier="deeper")
        assert outcome is None

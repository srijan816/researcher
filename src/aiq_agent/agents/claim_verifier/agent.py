# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Second-pass claim-to-report alignment verifier.

The verifier is designed to be invoked after a draft report is written. Full
LLM-backed atomic decomposition can be layered on top of these schemas; this
module provides the deterministic alignment and summary logic that should never
block report delivery if the verifier path itself fails.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from aiq_agent.common.claim_table import AtomicClaim
from aiq_agent.common.claim_table import ClaimTable
from aiq_agent.common.claim_table import ClaimType

AlignmentStatus = Literal["supported", "partial_support", "unsupported", "contradicted"]
AlignmentConfidence = Literal["high", "medium", "low"]
CoverageRelevance = Literal["essential", "relevant", "tangential"]
OmissionSeverity = Literal["high", "medium", "low"]

_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]+", re.IGNORECASE)
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "between",
    "claim",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "must",
    "that",
    "their",
    "this",
    "with",
    "would",
}


class ReportAtomicClaim(BaseModel):
    """One factual assertion decomposed from the draft report."""

    report_claim_id: str
    paragraph_index: int
    sentence_excerpt: str
    claim_text: str
    claim_type: ClaimType = "other"


class AlignmentResult(BaseModel):
    """Mapping between a report claim and claim-table support."""

    report_claim_id: str
    supported_by_claim_ids: list[str] = Field(default_factory=list)
    alignment_status: AlignmentStatus
    alignment_confidence: AlignmentConfidence = "medium"
    reasoning: str


class CoverageResult(BaseModel):
    """Whether a verified claim-table entry was used in the report."""

    claim_id: str
    claim_text: str
    in_report: bool
    relevance: CoverageRelevance = "relevant"
    omission_severity: OmissionSeverity | None = None


class VerificationSummary(BaseModel):
    """Compact verifier metrics suitable for SSE/API artifacts."""

    total_report_claims: int
    supported: int
    partial_support: int
    unsupported: int
    contradicted: int
    missing_verified_essential: int
    missing_verified_relevant: int
    support_rate: float
    calibration_failures: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Full verification artifact."""

    job_id: str | None = None
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    report_claims: list[ReportAtomicClaim] = Field(default_factory=list)
    alignments: list[AlignmentResult] = Field(default_factory=list)
    coverage: list[CoverageResult] = Field(default_factory=list)
    summary: VerificationSummary


def verify_report_claims_against_table(
    report_claims: list[ReportAtomicClaim],
    claim_table: ClaimTable,
    *,
    job_id: str | None = None,
) -> VerificationReport:
    """Align report atomic claims to the claim table with deterministic heuristics.

    This is not a replacement for the LLM verifier. It is the safe fallback and
    testable core: if it finds no support, the LLM verifier should not be more
    permissive unless it can explain exactly which claim-table entry supports a
    sentence.
    """
    atomic_claims = _claim_table_atomic_claims(claim_table)
    alignments = [_align_report_claim(report_claim, atomic_claims) for report_claim in report_claims]
    coverage = _coverage_results(atomic_claims, alignments)
    summary = _summary(alignments, coverage)
    return VerificationReport(
        job_id=job_id or claim_table.job_id,
        report_claims=report_claims,
        alignments=alignments,
        coverage=coverage,
        summary=summary,
    )


def _align_report_claim(
    report_claim: ReportAtomicClaim,
    table_claims: list[AtomicClaim],
) -> AlignmentResult:
    candidates = sorted(
        (
            (_overlap_score(report_claim.claim_text, claim.claim_text), claim)
            for claim in table_claims
            if claim.status in {"verified", "partially_verified"}
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    candidates = [item for item in candidates if item[0] >= 0.35]
    if not candidates:
        return AlignmentResult(
            report_claim_id=report_claim.report_claim_id,
            alignment_status="unsupported",
            alignment_confidence="medium",
            reasoning="No verified or partially verified claim-table entry has enough lexical overlap.",
        )

    best_score, best_claim = candidates[0]
    support_ids = [claim.claim_id for score, claim in candidates[:3] if score >= max(0.35, best_score - 0.15)]
    if best_claim.status == "verified":
        return AlignmentResult(
            report_claim_id=report_claim.report_claim_id,
            supported_by_claim_ids=support_ids,
            alignment_status="supported",
            alignment_confidence="high" if best_score >= 0.55 else "medium",
            reasoning=f"Report claim overlaps verified claim-table entry {best_claim.claim_id}.",
        )
    return AlignmentResult(
        report_claim_id=report_claim.report_claim_id,
        supported_by_claim_ids=support_ids,
        alignment_status="partial_support",
        alignment_confidence="medium",
        reasoning=f"Report claim is supported only by partially verified claim-table entry {best_claim.claim_id}.",
    )


def _coverage_results(
    table_claims: list[AtomicClaim],
    alignments: list[AlignmentResult],
) -> list[CoverageResult]:
    used_by_claim_id: dict[str, list[str]] = defaultdict(list)
    for alignment in alignments:
        for claim_id in alignment.supported_by_claim_ids:
            used_by_claim_id[claim_id].append(alignment.report_claim_id)

    results: list[CoverageResult] = []
    for claim in table_claims:
        if claim.status != "verified":
            continue
        in_report = claim.claim_id in used_by_claim_id
        results.append(
            CoverageResult(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                in_report=in_report,
                relevance="essential" if _looks_essential(claim) else "relevant",
                omission_severity=None if in_report else "high" if _looks_essential(claim) else "medium",
            )
        )
    return results


def _summary(alignments: list[AlignmentResult], coverage: list[CoverageResult]) -> VerificationSummary:
    supported = sum(1 for item in alignments if item.alignment_status == "supported")
    partial = sum(1 for item in alignments if item.alignment_status == "partial_support")
    unsupported = sum(1 for item in alignments if item.alignment_status == "unsupported")
    contradicted = sum(1 for item in alignments if item.alignment_status == "contradicted")
    total = len(alignments)
    missing_essential = sum(1 for item in coverage if not item.in_report and item.relevance == "essential")
    missing_relevant = sum(1 for item in coverage if not item.in_report and item.relevance == "relevant")
    calibration_failures = [
        f"{item.report_claim_id}: only partially verified support"
        for item in alignments
        if item.alignment_status == "partial_support"
    ]
    return VerificationSummary(
        total_report_claims=total,
        supported=supported,
        partial_support=partial,
        unsupported=unsupported,
        contradicted=contradicted,
        missing_verified_essential=missing_essential,
        missing_verified_relevant=missing_relevant,
        support_rate=supported / total if total else 1.0,
        calibration_failures=calibration_failures,
    )


def _claim_table_atomic_claims(claim_table: ClaimTable) -> list[AtomicClaim]:
    claims = list(claim_table.atomic_claims)
    for entry in claim_table.entries:
        evidence = []
        if entry.source_url and entry.source_extract:
            from aiq_agent.common.claim_table import ClaimEvidence

            evidence.append(
                ClaimEvidence(
                    source_url=entry.source_url,
                    source_class=entry.source_class or "unknown",
                    extract=entry.source_extract,
                    extract_confidence="high" if entry.confidence == "high" else "medium",
                )
            )
        claims.append(
            AtomicClaim(
                claim_id=entry.claim_id,
                claim_text=entry.claim,
                claim_type="definitional" if entry.claim_type == "definition" else entry.claim_type,
                expected_answer_shape="free_text",
                preferred_source_classes=[
                    entry.required_source_class if entry.required_source_class != "unknown" else "any_credible"
                ],
                status=entry.status,
                resolved_value=entry.value,
                evidence=evidence,
                hedge_required=entry.status == "partially_verified",
                hedge_phrase="according to available secondary evidence"
                if entry.status == "partially_verified"
                else None,
                researcher_notes=entry.notes or entry.downgrade_reason,
                attempted_sources=entry.attempted_sources,
            )
        )
    return claims


def _overlap_score(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) >= 3 and token.lower() not in _STOPWORDS}


def _looks_essential(claim: AtomicClaim) -> bool:
    return claim.claim_type in {"specification", "quantitative", "causal", "comparative", "trend"}

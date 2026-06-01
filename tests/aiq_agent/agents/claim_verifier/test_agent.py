from aiq_agent.agents.claim_verifier import ReportAtomicClaim
from aiq_agent.agents.claim_verifier import verify_report_claims_against_table
from aiq_agent.common.claim_table import AtomicClaim
from aiq_agent.common.claim_table import ClaimEvidence
from aiq_agent.common.claim_table import ClaimTable


def test_verifier_marks_supported_claims_and_coverage():
    table = ClaimTable(
        job_id="job-1",
        atomic_claims=[
            AtomicClaim(
                claim_id="C1",
                claim_text="IDC estimates AI skills shortages could cost $5.5 trillion in 2026",
                claim_type="quantitative",
                expected_answer_shape="number",
                preferred_source_classes=["primary_issuer"],
                status="verified",
                resolved_value="$5.5 trillion",
                evidence=[
                    ClaimEvidence(
                        source_url="https://idc.com/report",
                        source_class="primary_issuer",
                        extract="AI skills shortages could cost $5.5 trillion.",
                    )
                ],
            )
        ],
    )
    report = [
        ReportAtomicClaim(
            report_claim_id="RC1",
            paragraph_index=0,
            sentence_excerpt="IDC estimates AI skills shortages could cost $5.5 trillion in 2026.",
            claim_text="IDC estimates AI skills shortages could cost $5.5 trillion in 2026",
            claim_type="quantitative",
        )
    ]

    verification = verify_report_claims_against_table(report, table)

    assert verification.summary.supported == 1
    assert verification.summary.support_rate == 1.0
    assert verification.coverage[0].in_report


def test_verifier_flags_partial_and_unsupported_claims():
    table = ClaimTable(
        atomic_claims=[
            AtomicClaim(
                claim_id="C1",
                claim_text="Service businesses have stronger survival rates than product businesses",
                claim_type="comparative",
                expected_answer_shape="short_string",
                preferred_source_classes=["primary_issuer", "academic"],
                status="partially_verified",
                resolved_value="Reported but not traced to primary source",
                evidence=[
                    ClaimEvidence(
                        source_url="https://themoneypocket.com/stats",
                        source_class="content_marketing",
                        extract="Service businesses survive more often.",
                    )
                ],
                hedge_required=True,
                hedge_phrase="according to secondary coverage",
            )
        ]
    )
    report = [
        ReportAtomicClaim(
            report_claim_id="RC1",
            paragraph_index=0,
            sentence_excerpt="Service businesses have stronger survival rates than product businesses.",
            claim_text="Service businesses have stronger survival rates than product businesses",
            claim_type="comparative",
        ),
        ReportAtomicClaim(
            report_claim_id="RC2",
            paragraph_index=1,
            sentence_excerpt="AI franchises are guaranteed to succeed.",
            claim_text="AI franchises are guaranteed to succeed",
            claim_type="recommendation",
        ),
    ]

    verification = verify_report_claims_against_table(report, table)

    assert verification.summary.partial_support == 1
    assert verification.summary.unsupported == 1
    assert verification.summary.calibration_failures == ["RC1: only partially verified support"]

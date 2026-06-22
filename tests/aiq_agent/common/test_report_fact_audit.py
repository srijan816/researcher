# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from aiq_agent.common.report_fact_audit import evaluate_report_fact_audit
from aiq_agent.common.report_fact_audit import fact_audit_note


def _packet(*sources: dict):
    return json.dumps({"sources": list(sources)})


def test_fact_audit_flags_duplicate_reference_numbers():
    report = (
        "## Findings\n"
        "The harness supports paired comparisons [1].\n\n"
        "## References\n"
        "[1] Source A: https://example.com/a\n"
        "[1] Source B: https://example.com/b\n"
    )

    audit = evaluate_report_fact_audit(report)

    assert audit.hard_failed
    assert audit.hard_issues[0].code == "duplicate_reference_number"


def test_fact_audit_flags_numeric_value_not_in_cited_extract():
    report = (
        "## Findings\n"
        "G-Eval reported Spearman correlation of 0.81-0.85 with human judgment [1].\n\n"
        "## Sources\n"
        "[1] G-Eval paper: https://arxiv.org/abs/2303.16634\n"
    )
    evidence_packet = _packet(
        {
            "url": "https://arxiv.org/abs/2303.16634",
            "extracts": [
                {"text": "On summarization, G-Eval achieves Spearman correlation of 0.514 with human judgments."}
            ],
        }
    )

    audit = evaluate_report_fact_audit(report, evidence_packet)

    assert audit.hard_failed
    assert audit.hard_issues[0].code == "numeric_value_not_in_cited_evidence"
    assert "0.81" in audit.hard_issues[0].message


def test_fact_audit_flags_arxiv_id_mismatch():
    report = (
        "## Findings\n"
        "AppWorld is described in arXiv 2407.08901 [1].\n\n"
        "## Sources\n"
        "[1] AppWorld: https://arxiv.org/abs/2407.18901\n"
    )

    audit = evaluate_report_fact_audit(report)

    assert audit.hard_failed
    assert audit.hard_issues[0].code == "arxiv_id_citation_mismatch"


def test_fact_audit_warns_when_precise_numeric_claim_lacks_extract():
    report = (
        "## Findings\n"
        "The Notion case study reportedly improved throughput by 10x [1].\n\n"
        "## References\n"
        "[1] Braintrust Notion case study: https://braintrust.dev/customers/notion\n"
    )

    audit = evaluate_report_fact_audit(report, evidence_packet_content=None)

    assert not audit.hard_failed
    assert audit.warnings[0].code == "numeric_claim_missing_extract"


def test_fact_audit_passes_when_cited_extract_contains_reported_value():
    report = (
        "## Findings\n"
        "Lynx/HaluBench contains 15k hallucination-detection samples [1].\n\n"
        "## References\n"
        "[1] Lynx paper: https://arxiv.org/abs/2407.08488\n"
    )
    evidence_packet = _packet(
        {
            "url": "https://arxiv.org/abs/2407.08488",
            "extracts": [{"text": "HaluBench contains 15k samples for hallucination evaluation."}],
        }
    )

    audit = evaluate_report_fact_audit(report, evidence_packet)

    assert not audit.hard_failed
    assert not audit.issues


def test_fact_audit_flags_directional_source_inversion():
    report = (
        "## Findings\n"
        "The AGENTS.md study found that repo instructions improved agent outcomes, with 28.64% lower runtime [1].\n\n"
        "## References\n"
        "[1] AGENTS.md study: https://arxiv.org/abs/2601.20404\n"
    )
    evidence_packet = _packet(
        {
            "url": "https://arxiv.org/abs/2601.20404",
            "extracts": [
                {
                    "text": (
                        "The paper reports that AGENTS.md files reduce success rates and increase costs "
                        "for coding agents in several settings."
                    )
                }
            ],
        }
    )

    audit = evaluate_report_fact_audit(report, evidence_packet)

    assert audit.hard_failed
    codes = {issue.code for issue in audit.hard_issues}
    assert "claim_direction_contradicted_by_evidence" in codes


def test_fact_audit_flags_inflated_verified_source_count():
    report = (
        "## Findings\n"
        "This report is based on 1,283 verified sources [1].\n\n"
        "## References\n"
        "[1] One source: https://example.com/source\n"
    )

    audit = evaluate_report_fact_audit(report)

    assert audit.hard_failed
    assert audit.hard_issues[0].code == "inflated_verified_source_count"


def test_fact_audit_note_is_reader_facing():
    report = "## Findings\nThe result was 10x [1].\n\n## Sources\n[1] Case study: https://example.com/case\n"
    audit = evaluate_report_fact_audit(report)

    note = fact_audit_note(audit)

    assert "Source Accuracy Notes" in note
    assert "captured source extracts" in note

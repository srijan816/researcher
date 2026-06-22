import json

from aiq_agent.common.claim_table import merge_claim_tables_json
from aiq_agent.common.claim_table import summarize_claim_table
from aiq_agent.common.claim_table import validate_claim_table_json


def test_validates_verified_claim_requires_source_support():
    content = json.dumps(
        {
            "entries": [
                {
                    "claim_id": "C1",
                    "claim": "IDC estimates AI skills shortages cost $5.5T.",
                    "claim_type": "quantitative",
                    "status": "verified",
                    "value": "$5.5T",
                    "source_url": "https://idc.com/report",
                    "source_class": "primary_issuer",
                    "required_source_class": "primary_issuer",
                    "source_extract": "IDC estimates...",
                    "confidence": "high",
                }
            ]
        }
    )

    table, errors = validate_claim_table_json(content)

    assert not errors
    assert table is not None
    assert table.entries[0].claim_id == "C1"


def test_rejects_verified_claim_without_extract():
    content = json.dumps(
        {
            "entries": [
                {
                    "claim_id": "C1",
                    "claim": "A specific statistic.",
                    "claim_type": "quantitative",
                    "status": "verified",
                    "source_url": "https://example.com",
                    "source_class": "content_marketing",
                }
            ]
        }
    )

    table, errors = validate_claim_table_json(content)

    assert table is None
    assert "source_extract" in errors[0]


def test_merge_prefers_verified_claim_over_partial():
    partial = json.dumps(
        {
            "entries": [
                {
                    "claim_id": "C1",
                    "claim": "A market-size claim.",
                    "claim_type": "quantitative",
                    "status": "partially_verified",
                    "source_url": "https://blog.example/stats",
                    "source_class": "content_marketing",
                    "source_extract": "A blog reports...",
                    "downgrade_reason": "Primary source not found.",
                }
            ]
        }
    )
    verified = json.dumps(
        {
            "entries": [
                {
                    "claim_id": "C1",
                    "claim": "A market-size claim.",
                    "claim_type": "quantitative",
                    "status": "verified",
                    "source_url": "https://primary.example/report",
                    "source_class": "primary_issuer",
                    "source_extract": "The report states...",
                }
            ]
        }
    )

    table, errors = merge_claim_tables_json([partial, verified])

    assert not errors
    assert table is not None
    assert table.entries[0].status == "verified"
    assert table.entries[0].source_url == "https://primary.example/report"


def test_summarize_claim_table_counts_status_and_types():
    content = json.dumps(
        {
            "entries": [
                {
                    "claim_id": "C1",
                    "claim": "A verified claim.",
                    "claim_type": "trend",
                    "status": "verified",
                    "source_url": "https://example.com/report",
                    "source_class": "authoritative_third_party",
                    "source_extract": "Trend evidence.",
                },
                {
                    "claim_id": "C2",
                    "claim": "An unverified claim.",
                    "claim_type": "causal",
                    "status": "unverified",
                    "notes": "No source found.",
                },
            ]
        }
    )

    summary = summarize_claim_table(content)

    assert summary["valid"]
    assert summary["total"] == 2
    assert summary["by_status"]["verified"] == 1
    assert summary["by_status"]["unverified"] == 1
    assert summary["by_type"]["trend"] == 1


def test_accepts_planner_atomic_claim_profile_without_resolution_notes():
    content = json.dumps(
        {
            "profile": {
                "claim_density": "medium",
                "verifiability": "mixed",
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_text": "IDC estimate of AI skills shortage cost in 2026",
                        "claim_type": "quantitative",
                        "expected_answer_shape": "number",
                        "preferred_source_classes": ["primary_issuer"],
                    }
                ],
            },
            "atomic_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "IDC estimate of AI skills shortage cost in 2026",
                    "claim_type": "quantitative",
                    "expected_answer_shape": "number",
                    "preferred_source_classes": ["primary_issuer"],
                }
            ],
        }
    )

    table, errors = validate_claim_table_json(content)

    assert not errors
    assert table is not None
    assert table.atomic_claims[0].status == "unverified"


def test_merge_prefers_atomic_claim_with_stronger_status_and_evidence():
    weak = json.dumps(
        {
            "atomic_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "A pricing fact",
                    "claim_type": "quantitative",
                    "expected_answer_shape": "number",
                    "preferred_source_classes": ["first_party"],
                    "status": "partially_verified",
                    "resolved_value": "$1",
                    "evidence": [
                        {
                            "source_url": "https://blog.example/pricing",
                            "source_class": "content_marketing",
                            "extract": "A blog says it costs $1.",
                        }
                    ],
                    "hedge_required": True,
                    "hedge_phrase": "according to secondary coverage",
                }
            ]
        }
    )
    strong = json.dumps(
        {
            "atomic_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "A pricing fact",
                    "claim_type": "quantitative",
                    "expected_answer_shape": "number",
                    "preferred_source_classes": ["first_party"],
                    "status": "verified",
                    "resolved_value": "$1",
                    "evidence": [
                        {
                            "source_url": "https://vendor.example/docs/pricing",
                            "source_class": "first_party",
                            "extract": "Pricing is $1.",
                        }
                    ],
                }
            ]
        }
    )

    table, errors = merge_claim_tables_json([weak, strong])

    assert not errors
    assert table is not None
    assert table.atomic_claims[0].status == "verified"
    assert table.atomic_claims[0].evidence[0].source_class == "first_party"

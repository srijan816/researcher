import json

from aiq_agent.common.research_artifacts import build_virtual_research_artifacts
from aiq_agent.common.research_artifacts import mirror_run_artifacts


def _files():
    plan = {
        "report_title": "AI Use Cases",
        "task_analysis": {"user_intent": "Research AI use cases"},
        "report_toc": [{"title": "Executive Summary"}],
        "queries": [{"task_id": "Q1", "query": "AI use cases", "budget_percent": 100, "search_budget": 4}],
    }
    claim_table = {
        "atomic_claims": [
            {
                "claim_id": "C1",
                "claim_text": "AI automation can reduce handling time.",
                "claim_type": "quantitative",
                "expected_answer_shape": "free_text",
                "preferred_source_classes": ["authoritative_third_party"],
                "status": "verified",
                "resolved_value": "reduced handling time",
                "evidence": [
                    {
                        "source_url": "https://mckinsey.com/capabilities/quantum/article",
                        "source_class": "authoritative_third_party",
                        "extract": "Automation reduced handling time in the cited example.",
                        "extract_confidence": "high",
                    }
                ],
            },
            {
                "claim_id": "C2",
                "claim_text": "Unverified market size claim.",
                "claim_type": "quantitative",
                "expected_answer_shape": "number",
                "preferred_source_classes": ["primary_issuer"],
                "status": "unverified",
            },
        ]
    }
    evidence = {
        "job_id": "job-1",
        "source_count": 1,
        "claim_count": 2,
        "sources": [
            {
                "url": "https://mckinsey.com/capabilities/quantum/article",
                "title": "McKinsey Article",
                "source_class": "authoritative_third_party",
                "claim_ids": ["C1"],
                "extracts": [{"text": "Automation reduced handling time.", "claim_ids": ["C1"]}],
            }
        ],
    }
    return {
        "/shared/plan.json": json.dumps(plan),
        "/shared/claim_table.json": json.dumps(claim_table),
        "/shared/evidence_packet.json": json.dumps(evidence),
        "/shared/section_briefs/q1.md": "## Executive Summary\n\nAI automation evidence.",
    }


def test_build_virtual_research_artifacts_creates_compile_files():
    artifacts = build_virtual_research_artifacts(files=_files(), job_id="job-1", request_text="Research AI use cases")

    assert "/shared/research.md" in artifacts
    assert "/shared/sources.json" in artifacts
    assert "/shared/gaps.md" in artifacts
    assert "C1" in artifacts["/shared/research.md"]
    assert "Unverified claim C2" in artifacts["/shared/gaps.md"]


def test_mirror_run_artifacts_writes_memory_folder(tmp_path):
    manifest = mirror_run_artifacts(
        files=_files(),
        job_id="job-1",
        request_text="Research AI use cases",
        final_report="# Final\n\n## Sources\n\n[1] https://example.com\n",
        runs_dir=tmp_path,
    )

    assert manifest.source_count == 1
    assert (tmp_path / manifest.run_dir.split("/")[-1] / "plan.md").exists()
    assert (tmp_path / manifest.run_dir.split("/")[-1] / "research.md").exists()
    assert (tmp_path / manifest.run_dir.split("/")[-1] / "final.md").exists()

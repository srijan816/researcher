# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.report_quality import evaluate_report_source_quality
from aiq_agent.common.report_quality import has_degenerate_repetition
from aiq_agent.common.report_quality import low_diversity_token_loop
from aiq_agent.common.report_quality import repeated_token_run
from aiq_agent.common.report_quality import sanitize_report_structure
from aiq_agent.common.report_quality import strip_degenerate_repetition


def _registry_with_sources(*sources: tuple[str, str]):
    registry = SourceRegistry()
    for url, source_class in sources:
        registry.add(SourceEntry(url=url, title=url, source_class=source_class))
    return registry


def test_repeated_token_run_detects_model_loop():
    text = "Good start. " + ("actors " * 26) + "continue"

    assert repeated_token_run(text, min_run=24) == ("actors", 24)
    assert has_degenerate_repetition(text, min_run=24)


def test_strip_degenerate_repetition_keeps_useful_prefix():
    text = "Good start. " + ("actors " * 14)

    assert strip_degenerate_repetition(text, min_run=10) == "Good start."


def test_low_diversity_token_loop_detects_alternating_model_loop():
    text = "Good evidence. " + ("specialists puppet specialists puppet " * 12)

    assert low_diversity_token_loop(text, min_tokens=40) == ("specialists,puppet", 40)
    assert has_degenerate_repetition(text, min_run=24)


def test_strip_degenerate_repetition_handles_low_diversity_tail():
    text = "Good evidence. " + ("specialists puppet specialists puppet " * 12)

    assert strip_degenerate_repetition(text, min_run=10) == "Good evidence."


def test_strip_degenerate_repetition_leaves_normal_prose():
    text = "Actors include states, firms, banks, civil society, and international organizations."

    assert strip_degenerate_repetition(text, min_run=10) == text


def test_source_quality_flags_single_source_dependency():
    registry = _registry_with_sources(
        ("https://themoneypocket.com/make-money", "blog"),
        ("https://bls.gov/report", "third_party_authoritative"),
        ("https://gemconsortium.org/report", "third_party_authoritative"),
        ("https://weforum.org/report", "third_party_authoritative"),
        ("https://idc.com/report", "third_party_authoritative"),
    )
    body = " ".join(f"Claim {idx} about 2026 income [1]." for idx in range(1, 10))
    report = (
        f"## Findings\n{body}\n\n## References\n"
        "[1] Money Pocket: https://themoneypocket.com/make-money\n"
        "[2] BLS: https://bls.gov/report\n"
        "[3] GEM: https://gemconsortium.org/report\n"
        "[4] WEF: https://weforum.org/report\n"
        "[5] IDC: https://idc.com/report\n"
    )

    result = evaluate_report_source_quality(report, registry)

    assert not result.passed
    assert result.reason.startswith("too_few_distinct_cited_domains") or result.reason.startswith(
        "single_source_dependency"
    )


def test_source_quality_flags_weak_numeric_claims():
    registry = _registry_with_sources(
        ("https://themoneypocket.com/stats", "blog"),
        ("https://ideaproof.io/list", "blog"),
        ("https://packapop.com/report", "blog"),
        ("https://onlinekormo.com/ideas", "blog"),
    )
    report = (
        "## Findings\n"
        "The market is worth $5.5 trillion [1]. "
        "About 93% of workers need reskilling [2]. "
        "Service firms are 2x as likely to survive [3]. "
        "Another 69% struggle with finance [4].\n\n"
        "## References\n"
        "[1] Blog A: https://themoneypocket.com/stats\n"
        "[2] Blog B: https://ideaproof.io/list\n"
        "[3] Blog C: https://packapop.com/report\n"
        "[4] Blog D: https://onlinekormo.com/ideas\n"
    )

    result = evaluate_report_source_quality(report, registry)

    assert not result.passed
    assert "numeric_claims_need_primary_sources" in result.reason or "blog_source_dominance" in result.reason


def test_source_quality_passes_diverse_authoritative_report():
    registry = _registry_with_sources(
        ("https://bls.gov/report", "third_party_authoritative"),
        ("https://gemconsortium.org/report", "third_party_authoritative"),
        ("https://weforum.org/report", "third_party_authoritative"),
        ("https://idc.com/report", "third_party_authoritative"),
    )
    report = (
        "## Findings\n"
        "The labor market signal comes from BLS [1]. "
        "Entrepreneurial activity is tracked by GEM [2]. "
        "Skills changes are covered by WEF [3]. "
        "AI spending estimates are covered by IDC [4]. "
        "The combined evidence supports a diversified strategy [1][2][3][4].\n\n"
        "## References\n"
        "[1] BLS: https://bls.gov/report\n"
        "[2] GEM: https://gemconsortium.org/report\n"
        "[3] WEF: https://weforum.org/report\n"
        "[4] IDC: https://idc.com/report\n"
    )

    result = evaluate_report_source_quality(report, registry)

    assert result.passed


def test_source_quality_accepts_numbered_sources_heading():
    registry = _registry_with_sources(
        ("https://bls.gov/report", "third_party_authoritative"),
        ("https://gemconsortium.org/report", "third_party_authoritative"),
        ("https://weforum.org/report", "third_party_authoritative"),
        ("https://idc.com/report", "third_party_authoritative"),
    )
    report = (
        "## Findings\n"
        "The labor market signal comes from BLS [1]. "
        "Entrepreneurial activity is tracked by GEM [2]. "
        "Skills changes are covered by WEF [3]. "
        "AI spending estimates are covered by IDC [4].\n\n"
        "## 14. Sources\n"
        "[1] BLS: https://bls.gov/report\n"
        "[2] GEM: https://gemconsortium.org/report\n"
        "[3] WEF: https://weforum.org/report\n"
        "[4] IDC: https://idc.com/report\n"
    )

    result = evaluate_report_source_quality(report, registry)

    assert result.passed


def test_source_quality_flags_low_relevance_citations():
    registry = _registry_with_sources(
        ("https://example.edu/qaidam-basin-hydroclimate", "academic"),
        ("https://ssa.gov/employer", "primary_issuer"),
        ("https://kwsp.gov.my/employer", "primary_issuer"),
        ("https://example.com/salt-lake-geoscience", "academic"),
    )
    report = (
        "## Findings\n"
        + " ".join(f"Social movement claim {idx} [1][2][3][4]." for idx in range(1, 4))
        + "\n\n## Sources\n"
        "[1] Qaidam Basin hydroclimate: https://example.edu/qaidam-basin-hydroclimate\n"
        "[2] SSA employer portal: https://ssa.gov/employer\n"
        "[3] KWSP employer portal: https://kwsp.gov.my/employer\n"
        "[4] Salt lake geoscience: https://example.com/salt-lake-geoscience\n"
    )

    result = evaluate_report_source_quality(
        report,
        registry,
        request_text="Research social change, social justice, social movements, activism, protests, and civil rights.",
    )

    assert not result.passed
    assert result.reason.startswith("low_relevance_citation_share")


def test_sanitize_report_structure_removes_duplicate_appendix_sections():
    report = (
        "# Report\n\n"
        "## 1. Core Findings\n\nA strong unique section.\n\n"
        "## K. Recommended Debate Motions\n\n- THW do X\n- THBT Y\n\n"
        "## L. Further Reading\n\n- Real source\n\n"
        "## 23. Recommended Debate Motions\n\n- THW do X\n- THBT Y\n\n"
        "## 24. Further Reading\n\n- Real source\n"
    )

    cleaned = sanitize_report_structure(report)

    assert cleaned.count("Recommended Debate Motions") == 1
    assert cleaned.count("Further Reading") == 1
    assert "Core Findings" in cleaned

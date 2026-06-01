# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Universal source-diversity and source-calibration quality gates."""

from __future__ import annotations

from collections import Counter
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field

from .source_classification import AUTHORITATIVE_CLASSES
from .source_classification import WEAK_DERIVATIVE_CLASSES
from .source_classification import SourceClassification
from .source_classification import classify_url
from .source_classification import normalize_source_class

ResearchTier = Literal["shallow", "medium", "deeper", "deep"]
GateStatus = Literal["pass", "warn", "fail"]


class SourceQualityReport(BaseModel):
    """Structured signal describing citation source quality."""

    total_citations: int
    distinct_domains: int
    class_distribution: dict[str, int] = Field(default_factory=dict)
    dominant_domain: str | None = None
    dominant_domain_share: float = 0.0
    gate_results: dict[str, GateStatus] = Field(default_factory=dict)
    failure_reasons: list[str] = Field(default_factory=list)
    warning_messages: list[str] = Field(default_factory=list)
    overall_status: GateStatus = "pass"


def evaluate_source_quality(
    cited_urls: list[str],
    classifications: dict[str, SourceClassification] | None = None,
    tier: ResearchTier = "deep",
) -> SourceQualityReport:
    """Evaluate universal source quality gates for cited URLs."""
    classifications = classifications or {}
    total = len(cited_urls)
    if total == 0:
        return SourceQualityReport(
            total_citations=0,
            distinct_domains=0,
            gate_results={
                "domain_diversity": "fail",
                "concentration": "pass",
                "authority_floor": "fail",
                "weak_source_concentration": "fail",
            },
            failure_reasons=["no cited URLs"],
            overall_status="fail",
        )

    domains = [_domain(url) for url in cited_urls]
    domain_counts = Counter(domain for domain in domains if domain)
    dominant_domain = None
    dominant_share = 0.0
    if domain_counts:
        dominant_domain, dominant_count = domain_counts.most_common(1)[0]
        dominant_share = dominant_count / total

    source_classes = [_classification_for(url, classifications).source_class.value for url in cited_urls]
    class_counts = Counter(normalize_source_class(source_class) for source_class in source_classes)
    class_distribution = dict(sorted(class_counts.items()))

    gate_results: dict[str, GateStatus] = {}
    failures: list[str] = []
    warnings: list[str] = []

    min_domains = {"shallow": 2, "medium": 3, "deeper": 3, "deep": 4}[tier]
    max_dominant_share = {"shallow": 0.70, "medium": 0.60, "deeper": 0.60, "deep": 0.50}[tier]
    distinct_domains = len(domain_counts)

    if distinct_domains < min_domains:
        gate_results["domain_diversity"] = "fail"
        failures.append(f"too few distinct cited domains ({distinct_domains}/{min_domains})")
    else:
        gate_results["domain_diversity"] = "pass"

    if dominant_share > max_dominant_share:
        gate_results["concentration"] = "warn"
        warnings.append(f"dominant cited domain is {dominant_domain} ({dominant_share:.0%})")
    else:
        gate_results["concentration"] = "pass"

    authoritative_count = sum(class_counts[source_class] for source_class in AUTHORITATIVE_CLASSES)
    authoritative_share = authoritative_count / total
    if tier == "deep" and authoritative_share < 0.10:
        gate_results["authority_floor"] = "fail"
        failures.append(f"authoritative citation share below 10% ({authoritative_share:.0%})")
    elif tier == "deep" and authoritative_share < 0.25:
        gate_results["authority_floor"] = "warn"
        warnings.append(f"authoritative citation share below 25% ({authoritative_share:.0%})")
    else:
        gate_results["authority_floor"] = "pass"

    weak_count = sum(class_counts[source_class] for source_class in WEAK_DERIVATIVE_CLASSES)
    weak_share = weak_count / total
    if tier == "deep" and weak_share > 0.80:
        gate_results["weak_source_concentration"] = "fail"
        failures.append(f"content/vendor marketing citation share above 80% ({weak_share:.0%})")
    elif weak_share > 0.60:
        gate_results["weak_source_concentration"] = "warn"
        warnings.append(f"content/vendor marketing citation share above 60% ({weak_share:.0%})")
    else:
        gate_results["weak_source_concentration"] = "pass"

    overall: GateStatus = "fail" if failures else "warn" if warnings else "pass"
    return SourceQualityReport(
        total_citations=total,
        distinct_domains=distinct_domains,
        class_distribution=class_distribution,
        dominant_domain=dominant_domain,
        dominant_domain_share=dominant_share,
        gate_results=gate_results,
        failure_reasons=failures,
        warning_messages=warnings,
        overall_status=overall,
    )


def _classification_for(
    url: str,
    classifications: dict[str, SourceClassification],
) -> SourceClassification:
    return classifications.get(url) or classifications.get(url.rstrip("/")) or classify_url(url)


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")

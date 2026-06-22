# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fuzzy citation URL matching (canonicalization + host/path fallback)."""

import pytest

from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.citation_verification import _canonical_url
from aiq_agent.common.citation_verification import _registered_host
from aiq_agent.common.citation_verification import verify_citations


@pytest.fixture
def registry():
    return SourceRegistry()


class TestCanonicalUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.example.com/path/", "example.com/path"),
            ("http://example.com/path", "example.com/path"),
            ("HTTPS://WWW.EXAMPLE.COM/Path", "example.com/path"),
            ("https://m.example.com/path", "example.com/path"),
            ("https://mobile.example.com/path", "example.com/path"),
            ("https://amp.example.com/path", "example.com/path"),
            ("https://example.com/path#section-2", "example.com/path"),
            ("https://example.com/path?utm_source=x&utm_medium=y", "example.com/path"),
            ("https://example.com/path?utm_whatever=z", "example.com/path"),
            ("https://example.com/path?fbclid=abc123", "example.com/path"),
            ("https://example.com/path?gclid=abc", "example.com/path"),
            ("https://example.com/path?ref=hn&ref_src=twitter", "example.com/path"),
            ("https://example.com/path/amp", "example.com/path"),
            ("https://example.com/path?amp", "example.com/path"),
            ("https://example.com/path?id=5", "example.com/path?id=5"),
        ],
    )
    def test_canonical_forms(self, url, expected):
        assert _canonical_url(url) == expected

    def test_equivalent_variants_share_canonical_form(self):
        variants = [
            "https://www.example.com/news/story/",
            "http://example.com/news/story",
            "https://m.example.com/news/story?utm_source=newsletter",
            "https://example.com/news/story#top",
        ]
        forms = {_canonical_url(url) for url in variants}
        assert forms == {"example.com/news/story"}


class TestRegisteredHost:
    def test_simple_domain(self):
        assert _registered_host("example.com") == "example.com"

    def test_subdomain_collapses(self):
        assert _registered_host("blog.example.com") == "example.com"

    def test_country_second_level(self):
        assert _registered_host("news.bbc.co.uk") == "bbc.co.uk"


class TestFuzzyResolveUrl:
    @pytest.mark.parametrize(
        ("registered", "cited"),
        [
            # scheme variants
            ("https://example.com/report", "http://example.com/report"),
            # www variants both directions
            ("https://www.example.com/report", "https://example.com/report"),
            ("https://example.com/report", "https://www.example.com/report"),
            # tracking params
            ("https://example.com/report", "https://example.com/report?utm_source=chatgpt.com"),
            ("https://example.com/report?utm_source=feed", "https://example.com/report"),
            ("https://example.com/report", "https://example.com/report?fbclid=IwAR123"),
            # fragments and trailing slash
            ("https://example.com/report", "https://example.com/report/#conclusion"),
            # mobile/amp hosts and /amp suffix
            ("https://example.com/report", "https://m.example.com/report"),
            ("https://example.com/report", "https://example.com/report/amp"),
            # case differences
            ("https://example.com/Report", "https://example.com/report"),
        ],
    )
    def test_trivial_variants_match(self, registry, registered, cited):
        registry.add(SourceEntry(url=registered))
        assert registry.resolve_url(cited) == registered

    def test_subdomain_with_path_prefix_matches(self, registry):
        registry.add(SourceEntry(url="https://www.example.com/research/ai/report-2026"))
        resolved = registry.resolve_url("https://example.com/research/ai/report-2026/summary")
        assert resolved == "https://www.example.com/research/ai/report-2026"

    def test_high_path_similarity_matches(self, registry):
        registry.add(SourceEntry(url="https://example.com/blog/2026/06/ai-research-quality-report"))
        resolved = registry.resolve_url("https://example.com/blog/2026/06/ai-research-quality-reports")
        assert resolved == "https://example.com/blog/2026/06/ai-research-quality-report"

    @pytest.mark.parametrize(
        "cited",
        [
            "https://other.com/report",
            "https://example.com/completely/unrelated/path",
            "https://example.com/report?id=999",
        ],
    )
    def test_real_differences_do_not_match(self, registry, cited):
        registry.add(SourceEntry(url="https://example.com/report?id=123"))
        if cited == "https://example.com/report?id=999":
            assert registry.resolve_url(cited) is None
        else:
            registry.add(SourceEntry(url="https://example.com/report?id=456"))
            assert registry.resolve_url(cited) is None

    def test_ambiguous_canonical_variants_rejected(self, registry):
        registry.add(SourceEntry(url="https://example.com/story/one"))
        registry.add(SourceEntry(url="https://example.com/story/two"))
        # Both are prefix-children of /story — ambiguous, must reject
        assert registry.resolve_url("https://example.com/story/one/two/three/four") is None

    def test_domain_root_does_not_swallow_deep_urls_via_fuzzy(self, registry):
        registry.add(SourceEntry(url="https://example.com/"))
        # The legacy prefix strategy may still match domain-only registry
        # entries; the fuzzy path rule itself requires >= 2 segments. Cited
        # deep path on a *subdomain* defeats the legacy normalized-prefix
        # strategies, so this exercises the fuzzy gate.
        assert registry.resolve_url("https://blog.example.com/us/benefits/healthcare/") is None


class TestVerifyCitationsWithFuzzyMatching:
    def test_trivial_variant_citation_survives(self, registry):
        registry.add(SourceEntry(url="https://www.example.com/research/findings", title="Findings"))
        report = (
            "Key result was confirmed [1].\n\n"
            "## References\n"
            "[1] Findings: https://example.com/research/findings?utm_source=chatgpt.com\n"
        )
        result = verify_citations(report, registry)
        assert result.removed_citations == []
        assert len(result.valid_citations) == 1
        assert "[1]" in result.verified_report

    def test_fabricated_citation_still_removed(self, registry):
        registry.add(SourceEntry(url="https://www.example.com/research/findings"))
        report = (
            "Key result was confirmed [1]. Another claim [2].\n\n"
            "## References\n"
            "[1] Findings: https://example.com/research/findings\n"
            "[2] Fabricated: https://fabricated-source.io/totally/made/up\n"
        )
        result = verify_citations(report, registry)
        assert len(result.valid_citations) == 1
        assert len(result.removed_citations) == 1
        assert result.removed_citations[0]["number"] == 2
        assert "[2]" not in result.verified_report.split("## References")[0]

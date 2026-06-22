# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for source authority classification."""

from aiq_agent.common.source_classification import SourceClass
from aiq_agent.common.source_classification import classify_source
from aiq_agent.common.source_classification import classify_url
from aiq_agent.common.source_classification import normalize_source_class
from aiq_agent.common.source_classification import reload_registry
from aiq_agent.common.source_classification import source_class_counts
from aiq_agent.common.source_classification import source_class_rank


def test_classifies_known_first_party_docs_domain():
    assert classify_source("https://docs.x.ai/docs/grok-imagine") == "first_party"


def test_classifies_known_authoritative_marketplace():
    assert classify_source("https://openrouter.ai/models/anthropic/claude-opus-4.6") == "authoritative_third_party"


def test_classifies_blog_and_forum_sources():
    assert classify_source("https://medium.com/example/post") == "content_marketing"
    assert classify_source("https://www.reddit.com/r/artificial/comments/example") == "forum"


def test_classifies_docs_path_as_first_party_for_unknown_domain():
    assert classify_source("https://vendor.example/docs/api/reference") == "first_party"


def test_source_class_rank_orders_authority():
    assert source_class_rank("first_party") > source_class_rank("authoritative_third_party")
    assert source_class_rank("authoritative_third_party") > source_class_rank("content_marketing")
    assert source_class_rank("forum") > source_class_rank("unknown")


def test_source_class_counts_accepts_dict_entries():
    counts = source_class_counts(
        [
            {"source_class": "first_party"},
            {"source_class": "content_marketing"},
            {"source_class": "unexpected"},
        ]
    )

    assert counts["first_party"] == 1
    assert counts["content_marketing"] == 1
    assert counts["unknown"] == 1


def test_returns_detailed_classification_metadata():
    result = classify_url("https://bls.gov/news.release/example")

    assert result.source_class == SourceClass.PRIMARY_ISSUER
    assert result.normalized_domain == "bls.gov"
    assert result.confidence == 1.0
    assert "registry" in result.classification_reason


def test_classifies_gov_and_edu_suffixes():
    assert classify_source("https://data.example.gov/report") == "primary_issuer"
    assert classify_source("https://lab.example.edu/paper") == "academic"


def test_legacy_classes_are_normalized():
    assert normalize_source_class("blog") == "content_marketing"
    assert normalize_source_class("third_party_authoritative") == "authoritative_third_party"


def test_registry_hot_reload_is_callable():
    registry = reload_registry()

    assert "domains" in registry


def test_classifies_common_analyst_academic_and_trade_sources():
    assert classify_source("https://www.mckinsey.com/capabilities/quantumblack/our-insights/report") == "primary_issuer"
    assert classify_source("https://www.oecd.org/en/publications/example.html") == "primary_issuer"
    assert classify_source("https://pubmed.ncbi.nlm.nih.gov/12345678/") == "academic"
    assert classify_source("https://www.reuters.com/world/example") == "trade_press"

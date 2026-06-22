# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deep-research fact ledger validation."""

from aiq_agent.common.fact_ledger import live_fact_conflicts
from aiq_agent.common.fact_ledger import merge_fact_ledgers_json
from aiq_agent.common.fact_ledger import summarize_fact_ledger
from aiq_agent.common.fact_ledger import validate_fact_ledger_json


def test_validates_verified_fact_with_required_source_fields():
    ledger, errors = validate_fact_ledger_json(
        """
        {
          "entries": [
            {
              "entity": "Grok Imagine",
              "fact": "Maximum clip duration is 6 seconds.",
              "source_url": "https://docs.x.ai/docs/grok-imagine",
              "source_extract": "Clips can be up to 6 seconds long.",
              "source_class": "first_party",
              "confidence": "high",
              "status": "verified"
            }
          ]
        }
        """
    )

    assert errors == []
    assert ledger is not None
    assert ledger.entries[0].entity == "Grok Imagine"


def test_rejects_verified_fact_without_extract():
    ledger, errors = validate_fact_ledger_json(
        """
        {
          "entries": [
            {
              "entity": "Grok Imagine",
              "fact": "Negative prompts are supported.",
              "source_url": "https://example.com",
              "source_class": "blog",
              "status": "verified"
            }
          ]
        }
        """
    )

    assert ledger is None
    assert errors
    assert "source_extract" in errors[0]


def test_accepts_unverified_fact_with_reason():
    summary = summarize_fact_ledger(
        """
        {
          "entries": [
            {
              "entity": "Grok Imagine",
              "fact": "negative prompt support: UNVERIFIED",
              "status": "unverified",
              "confidence": "unverified",
              "reason": "No reviewed first-party documentation addresses this behavior."
            }
          ]
        }
        """
    )

    assert summary["valid"] is True
    assert summary["verified_count"] == 0
    assert summary["unverified_count"] == 1
    assert summary["entities"]["Grok Imagine"]["unverified"] == 1


def test_summarizes_entity_to_entries_map():
    summary = summarize_fact_ledger(
        """
        {
          "Claude Opus 4.6": [
            {
              "fact": "API docs list the model in the current model table.",
              "source_url": "https://docs.anthropic.com/en/docs/about-claude/models",
              "source_extract": "Claude Opus 4.6 appears in the model table.",
              "source_class": "first_party",
              "confidence": "high",
              "status": "verified"
            }
          ]
        }
        """
    )

    assert summary["valid"] is True
    assert summary["entities"]["Claude Opus 4.6"]["verified"] == 1


def test_merge_fact_ledgers_prefers_verified_duplicate():
    unverified = """
    {
      "entries": [
        {
          "entity": "Grok Imagine",
          "fact": "Negative prompt support",
          "status": "unverified",
          "confidence": "unverified",
          "reason": "No source found"
        }
      ]
    }
    """
    verified = """
    {
      "entries": [
        {
          "entity": "Grok Imagine",
          "fact": "Negative prompt support",
          "source_url": "https://docs.x.ai/grok",
          "source_extract": "The docs state the supported prompt behavior.",
          "source_class": "first_party",
          "confidence": "high",
          "status": "verified"
        }
      ]
    }
    """

    ledger, errors = merge_fact_ledgers_json([unverified, verified])

    assert errors == []
    assert ledger is not None
    assert len(ledger.entries) == 1
    assert ledger.entries[0].status == "verified"
    assert ledger.entries[0].source_url == "https://docs.x.ai/grok"


def test_live_fact_conflicts_detect_newer_funding_value():
    ledger, errors = validate_fact_ledger_json(
        """
        {
          "entries": [
            {
              "entity": "Sierra",
              "fact_type": "funding",
              "fact": "Sierra raised $175M at a $4.5B valuation.",
              "value": "$175M / $4.5B",
              "event_date": "2025-02-01",
              "source_url": "https://example.com/old",
              "source_extract": "Sierra raised $175M at a $4.5B valuation.",
              "source_class": "trade_press",
              "confidence": "medium",
              "status": "verified"
            },
            {
              "entity": "Sierra",
              "fact_type": "funding",
              "fact": "Sierra raised $950M at a $15.8B valuation.",
              "value": "$950M / $15.8B",
              "event_date": "2026-05-01",
              "source_url": "https://example.com/new",
              "source_extract": "Sierra raised $950M at a $15.8B valuation.",
              "source_class": "first_party",
              "confidence": "high",
              "status": "verified"
            }
          ]
        }
        """
    )

    assert errors == []
    assert ledger is not None
    conflicts = live_fact_conflicts(ledger)
    assert len(conflicts) == 1
    assert conflicts[0].entity == "Sierra"
    assert "newer verified evidence" in conflicts[0].message


def test_fact_ledger_summary_exposes_live_fact_conflicts():
    summary = summarize_fact_ledger(
        """
        {
          "entries": [
            {
              "entity": "Decagon",
              "fact_type": "valuation",
              "fact": "Decagon was valued at $1.5B.",
              "value": "$1.5B",
              "event_date": "2025-06",
              "source_url": "https://example.com/old",
              "source_extract": "Decagon was valued at $1.5B.",
              "source_class": "trade_press",
              "status": "verified"
            },
            {
              "entity": "Decagon",
              "fact_type": "valuation",
              "fact": "Decagon was valued at $4.5B.",
              "value": "$4.5B",
              "event_date": "2026-01",
              "source_url": "https://example.com/new",
              "source_extract": "Decagon was valued at $4.5B.",
              "source_class": "trade_press",
              "status": "verified"
            }
          ]
        }
        """
    )

    assert summary["valid"] is True
    assert summary["live_fact_conflicts"][0]["entity"] == "Decagon"

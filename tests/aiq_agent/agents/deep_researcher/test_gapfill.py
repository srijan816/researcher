# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded gap-fill round derivation and snapshot reporting."""

from aiq_agent.agents.deep_researcher.gapfill import gapfill_max_rounds
from aiq_agent.agents.deep_researcher.gapfill import gapfill_snapshot


class TestGapfillMaxRounds:
    def test_tier_defaults(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        assert gapfill_max_rounds("shallow") == 0
        assert gapfill_max_rounds("medium") == 1
        assert gapfill_max_rounds("deeper") == 2
        assert gapfill_max_rounds("deep") == 3

    def test_unknown_tier_uses_default_depth(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        # Unknown tiers normalize to the default tier (deeper)
        assert gapfill_max_rounds("bogus") == 2
        assert gapfill_max_rounds(None) == 2

    def test_alias_tiers(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        assert gapfill_max_rounds("quick") == 0
        assert gapfill_max_rounds("comprehensive") == 3

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AIQ_GAPFILL_MAX_ROUNDS", "5")
        assert gapfill_max_rounds("shallow") == 5
        assert gapfill_max_rounds("deep") == 5

    def test_env_override_clamped_and_invalid(self, monkeypatch):
        monkeypatch.setenv("AIQ_GAPFILL_MAX_ROUNDS", "-3")
        assert gapfill_max_rounds("deep") == 0
        monkeypatch.setenv("AIQ_GAPFILL_MAX_ROUNDS", "junk")
        assert gapfill_max_rounds("deep") == 3


class TestGapfillSnapshot:
    def test_reports_rounds_and_reserve(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        snapshot = gapfill_snapshot(
            tier="deeper",
            rounds_used=1,
            budget_snapshot={"budgets": [{"key": "search", "used": 70, "limit": 92, "remaining": 22}]},
            budget_profile={"reserve_search_calls": 15},
        )
        assert snapshot["rounds_used"] == 1
        assert snapshot["rounds_allowed"] == 2
        assert snapshot["rounds_remaining"] == 1
        assert snapshot["reserve_search_calls_total"] == 15
        assert snapshot["reserve_search_calls_remaining"] == 15
        assert snapshot["search_calls_remaining"] == 22
        assert snapshot["may_launch_gap_fill"] is True

    def test_stops_when_rounds_exhausted(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        snapshot = gapfill_snapshot(
            tier="medium",
            rounds_used=1,
            budget_snapshot={"budgets": [{"key": "search", "used": 10, "limit": 62, "remaining": 52}]},
            budget_profile={"reserve_search_calls": 11},
        )
        assert snapshot["rounds_remaining"] == 0
        assert snapshot["may_launch_gap_fill"] is False

    def test_stops_when_reserve_exhausted(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        snapshot = gapfill_snapshot(
            tier="deep",
            rounds_used=0,
            budget_snapshot={"budgets": [{"key": "search", "used": 168, "limit": 168, "remaining": 0}]},
            budget_profile={"reserve_search_calls": 28},
        )
        assert snapshot["reserve_search_calls_remaining"] == 0
        assert snapshot["may_launch_gap_fill"] is False

    def test_missing_budget_data_is_permissive_on_reserve(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        snapshot = gapfill_snapshot(tier="deeper", rounds_used=0)
        assert snapshot["rounds_allowed"] == 2
        assert snapshot["reserve_search_calls_remaining"] is None
        # Rounds still gate the loop even when budget data is unavailable
        assert snapshot["may_launch_gap_fill"] is True

    def test_shallow_never_allows_gap_fill(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        snapshot = gapfill_snapshot(tier="shallow", rounds_used=0)
        assert snapshot["rounds_allowed"] == 0
        assert snapshot["may_launch_gap_fill"] is False

    def test_negative_rounds_used_clamped(self, monkeypatch):
        monkeypatch.delenv("AIQ_GAPFILL_MAX_ROUNDS", raising=False)
        snapshot = gapfill_snapshot(tier="deeper", rounds_used=-4)
        assert snapshot["rounds_used"] == 0

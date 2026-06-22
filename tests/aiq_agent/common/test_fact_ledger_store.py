# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the persistent cross-run fact ledger store."""

import json

import pytest

from aiq_agent.common.fact_ledger_store import build_prior_facts_block
from aiq_agent.common.fact_ledger_store import lookup_facts_for_query
from aiq_agent.common.fact_ledger_store import lookup_prior_facts_block
from aiq_agent.common.fact_ledger_store import persist_fact_ledger_json
from aiq_agent.common.fact_ledger_store import upsert_fact


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "fact_ledger.db")


def _verified_entry(**overrides):
    entry = {
        "entity": "Nvidia",
        "fact": "Nvidia data-center revenue was $35.6B in Q4 FY2025",
        "fact_type": "financials",
        "value": "$35.6B",
        "event_date": "2025-02-26",
        "source_url": "https://nvidianews.nvidia.com/news/q4-fy2025",
        "source_extract": "Data Center revenue was $35.6 billion",
        "source_class": "first_party",
        "status": "verified",
    }
    entry.update(overrides)
    return entry


class TestUpsert:
    def test_roundtrip(self, db_path):
        assert upsert_fact(
            entity="Nvidia",
            fact_type="financials",
            value="$35.6B",
            fact="Q4 FY2025 data-center revenue",
            as_of="2025-02-26",
            source_url="https://example.com/q4",
            job_id="job-1",
            db_path=db_path,
        )
        facts = lookup_facts_for_query("nvidia earnings", db_path=db_path)
        assert len(facts) == 1
        assert facts[0]["value"] == "$35.6B"
        assert facts[0]["source_url"] == "https://example.com/q4"

    def test_upsert_replaces_on_entity_fact_type_key(self, db_path):
        upsert_fact(entity="Nvidia", fact_type="financials", value="old", db_path=db_path)
        upsert_fact(entity="NVIDIA", fact_type="Financials", value="new", db_path=db_path)
        facts = lookup_facts_for_query("nvidia", db_path=db_path)
        assert len(facts) == 1
        assert facts[0]["value"] == "new"

    def test_empty_entity_rejected(self, db_path):
        assert not upsert_fact(entity="  ", fact_type="general", value="x", db_path=db_path)
        assert lookup_facts_for_query("anything", db_path=db_path) == []

    def test_kill_switch(self, db_path, monkeypatch):
        monkeypatch.setenv("AIQ_FACT_LEDGER_ENABLED", "0")
        assert not upsert_fact(entity="Nvidia", fact_type="general", value="x", db_path=db_path)
        assert lookup_facts_for_query("nvidia", db_path=db_path) == []

    def test_unwritable_path_is_noop(self):
        assert not upsert_fact(
            entity="Nvidia",
            fact_type="general",
            value="x",
            db_path="/dev/null/nope/fact_ledger.db",
        )


class TestPersistLedgerJson:
    def test_persists_only_verified_entries(self, db_path):
        content = json.dumps(
            {
                "entries": [
                    _verified_entry(),
                    {
                        "entity": "Acme Corp",
                        "fact": "Series B amount unknown",
                        "status": "unverified",
                        "reason": "no primary source found",
                    },
                ]
            }
        )
        assert persist_fact_ledger_json(content, job_id="job-9", db_path=db_path) == 1
        facts = lookup_facts_for_query("nvidia revenue", db_path=db_path)
        assert len(facts) == 1
        assert facts[0]["job_id"] == "job-9"
        assert facts[0]["as_of"] == "2025-02-26"

    def test_invalid_json_is_noop(self, db_path):
        assert persist_fact_ledger_json("{not json", db_path=db_path) == 0
        assert persist_fact_ledger_json("", db_path=db_path) == 0


class TestLookup:
    def test_entity_token_match_in_query(self, db_path):
        upsert_fact(entity="MiniMax M3", fact_type="release_status", value="GA", db_path=db_path)
        upsert_fact(entity="Totally Unrelated Co", fact_type="general", value="x", db_path=db_path)
        facts = lookup_facts_for_query("How does MiniMax M3 compare to other models?", db_path=db_path)
        assert [fact["entity"] for fact in facts] == ["minimax m3"]

    def test_no_match_returns_empty(self, db_path):
        upsert_fact(entity="Nvidia", fact_type="general", value="x", db_path=db_path)
        assert lookup_facts_for_query("history of the roman empire", db_path=db_path) == []

    def test_limit_respected(self, db_path):
        for index in range(20):
            upsert_fact(entity="Nvidia", fact_type=f"type-{index}", value=str(index), db_path=db_path)
        assert len(lookup_facts_for_query("nvidia", limit=5, db_path=db_path)) == 5

    def test_missing_db_returns_empty(self, tmp_path):
        assert lookup_facts_for_query("nvidia", db_path=str(tmp_path / "missing.db")) == []


class TestPriorFactsBlock:
    def test_block_format(self, db_path):
        upsert_fact(
            entity="Nvidia",
            fact_type="financials",
            value="$35.6B",
            as_of="2025-02-26",
            source_url="https://example.com/q4",
            db_path=db_path,
        )
        block = lookup_prior_facts_block("nvidia outlook", db_path=db_path)
        assert "Previously verified facts" in block
        assert "nvidia — financials: $35.6B" in block
        assert "as of 2025-02-26" in block
        assert "source: https://example.com/q4" in block
        assert "re-verify" in block.lower()

    def test_empty_when_no_matches(self, db_path):
        assert lookup_prior_facts_block("nothing stored", db_path=db_path) == ""

    def test_build_block_skips_valueless_rows(self):
        assert build_prior_facts_block([{"entity": "x", "fact_type": "general", "value": ""}]) == ""
        assert build_prior_facts_block([]) == ""

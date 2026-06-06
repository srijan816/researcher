# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for async job components.

Module under test: frontends/aiq_api/src/aiq_api/jobs/

Test coverage:
    TestIntermediateStepEvent:
        - Event type property generation (category.state)
        - SSE dict serialization
        - Event data handling
        - Artifact category and types

    TestDeepResearchEventCallback:
        - Initialization with/without event store
        - Event emission for workflow/agent chains
        - Tool event emission with URL extraction
        - LLM event emission
        - Graceful handling when no event store is set

    TestDeepResearchEventCallbackAdvanced:
        - URL extraction and cleanup
        - Search tool detection
        - Tool call syntax detection
        - Artifact emission with workflow metadata
        - Input/output extraction

    TestSubmitDeepResearchJob:
        - Raises RuntimeError without NAT_DASK_SCHEDULER_ADDRESS
        - Successful job submission with required env vars
        - Custom job ID handling

    TestEventStore:
        - Event storage and retrieval
        - Cursor-based pagination with after_id
        - Async event retrieval
        - Event cleanup
        - Engine caching and disposal

    TestToolArtifactMapping:
        - Default tool mappings
        - Custom mapping registration

    TestCancellationMonitor:
        - Initialization and state
        - Cancellation check

    TestSQLAlchemyPoolFilter:
        - Error message filtering for CancelledError
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq_agent.auth import Principal
from aiq_api.jobs.callbacks import ArtifactType
from aiq_api.jobs.callbacks import DeepResearchEventCallback
from aiq_api.jobs.callbacks import EventCategory
from aiq_api.jobs.callbacks import EventData
from aiq_api.jobs.callbacks import EventState
from aiq_api.jobs.callbacks import IntermediateStepEvent
from aiq_api.jobs.runner import _build_success_output
from aiq_api.jobs.runner import _emit_quality_audit_artifacts
from aiq_api.jobs.runner import _evaluate_post_run_quality
from aiq_api.jobs.runner import _is_usable_report
from aiq_api.jobs.runner import _recover_report_from_events


@pytest.fixture(name="event_store_cache_guard", autouse=True)
def fixture_event_store_cache_guard():
    """Reset EventStore caches to avoid cross-test leakage."""
    from aiq_api.jobs.event_store import EventStore

    EventStore.dispose_all_engines()
    yield
    EventStore.dispose_all_engines()


def test_post_run_quality_gates_detect_missing_report_and_citations(tmp_path):
    """Post-run gates flag runs that never wrote report.md or cited verified sources."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-quality")
    store.store(
        {
            "type": "artifact.update",
            "name": "https://example.com/source",
            "data": {
                "type": "citation_source",
                "content": "https://example.com/source",
                "url": "https://example.com/source",
            },
        }
    )

    problems = _evaluate_post_run_quality(db_url, "job-quality")

    assert "no /report.md file artifact was produced" in problems
    assert "final report emitted no citation_use events" in problems


def test_quality_warnings_are_preserved_in_success_output():
    """Non-fatal quality problems should be returned with the usable report."""
    output = _build_success_output("# Report", quality_warnings=["source diversity low"])

    assert output["report"] == "# Report"
    assert output["quality_status"] == "warning"
    assert output["quality_warnings"] == ["source diversity low"]


def test_quality_audit_artifact_is_persisted(tmp_path):
    """Quality warnings should be visible through the event/artifact stream."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-quality-warning")

    _emit_quality_audit_artifacts(store, ["source diversity low", "claim table under-resolved"])

    events = EventStore.get_events(db_url, "job-quality-warning", 0, 100)
    event_types = [event.get("type") for event in events]
    artifact = next(event for event in events if event.get("type") == "artifact.update")

    assert "artifact.update" in event_types
    assert "quality.warning" in event_types
    assert artifact["name"] == "quality_audit"
    assert artifact["data"]["type"] == "quality_audit"
    assert artifact["data"]["warnings"] == ["source diversity low", "claim table under-resolved"]


def test_post_run_quality_gates_pass_with_report_and_citation(tmp_path):
    """Post-run gates pass when report file and citation_use artifacts exist."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-quality-ok")
    store.store(
        {
            "type": "artifact.update",
            "name": "/report.md",
            "data": {"type": "file", "content": "# Report\n\nBody"},
        }
    )
    for artifact_type in ("citation_source", "citation_use"):
        store.store(
            {
                "type": "artifact.update",
                "name": "https://example.com/source",
                "data": {
                    "type": artifact_type,
                    "content": "https://example.com/source",
                    "url": "https://example.com/source",
                },
            }
        )

    assert _evaluate_post_run_quality(db_url, "job-quality-ok") == []


def test_post_run_quality_sparse_gate_caps_candidate_source_expectation(tmp_path):
    """Candidate-source fanout should not require citing hundreds of URLs."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-quality-many-candidates")
    store.store(
        {
            "type": "artifact.update",
            "name": "/report.md",
            "data": {"type": "file", "content": "# Report\n\nBody"},
        }
    )
    for index in range(300):
        url = f"https://example.com/source-{index}"
        store.store(
            {
                "type": "artifact.update",
                "name": url,
                "data": {"type": "citation_source", "content": url, "url": url},
            }
        )
    for index in range(25):
        url = f"https://cited{index}.example.com/source"
        store.store(
            {
                "type": "artifact.update",
                "name": url,
                "data": {
                    "type": "citation_use",
                    "content": url,
                    "url": url,
                    "source_class": "authoritative_third_party",
                },
            }
        )

    problems = _evaluate_post_run_quality(db_url, "job-quality-many-candidates")

    assert not any("cited sources too sparse" in problem for problem in problems)


def test_post_run_quality_flags_entity_heavy_plan_without_fact_ledger(tmp_path):
    """Entity-heavy plans must produce a fact ledger, not just a report."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-quality-ledger-missing")
    store.store(
        {
            "type": "artifact.update",
            "name": "/shared/plan.json",
            "data": {
                "type": "file",
                "content": """
                {
                  "task_analysis": {
                    "entities": [
                      {"name": "Grok Imagine", "centrality": "central"},
                      {"name": "Claude Opus 4.6", "centrality": "central"},
                      {"name": "MiniMax M3", "centrality": "central"}
                    ]
                  },
                  "fact_ledger_targets": {
                    "Grok Imagine": [{"question": "Pricing?", "answer_type": "table"}],
                    "Claude Opus 4.6": [{"question": "Context window?", "answer_type": "number"}],
                    "MiniMax M3": [{"question": "Supported tools?", "answer_type": "short_text"}]
                  }
                }
                """,
            },
        }
    )
    store.store({"type": "artifact.update", "name": "/report.md", "data": {"type": "file", "content": "# Report"}})
    store.store(
        {
            "type": "artifact.update",
            "name": "https://docs.x.ai/docs/grok-imagine",
            "data": {
                "type": "citation_use",
                "url": "https://docs.x.ai/docs/grok-imagine",
                "source_class": "first_party",
            },
        }
    )

    problems = _evaluate_post_run_quality(db_url, "job-quality-ledger-missing")

    assert any("missing /shared/fact_ledger.json" in problem for problem in problems)


def test_post_run_quality_flags_central_entity_with_zero_verified_facts(tmp_path):
    """Every central entity in an entity-heavy plan needs at least one verified ledger fact."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-quality-ledger-empty-entity")
    store.store(
        {
            "type": "artifact.update",
            "name": "/shared/plan.json",
            "data": {
                "type": "file",
                "content": """
                {
                  "task_analysis": {
                    "entities": [
                      {"name": "Grok Imagine", "centrality": "central"},
                      {"name": "Claude Opus 4.6", "centrality": "central"},
                      {"name": "MiniMax M3", "centrality": "central"}
                    ]
                  }
                }
                """,
            },
        }
    )
    store.store(
        {
            "type": "artifact.update",
            "name": "/shared/fact_ledger.json",
            "data": {
                "type": "file",
                "content": """
                {
                  "entries": [
                    {
                      "entity": "Grok Imagine",
                      "fact": "Official docs describe Grok Imagine.",
                      "source_url": "https://docs.x.ai/docs/grok-imagine",
                      "source_extract": "Grok Imagine is described here.",
                      "source_class": "first_party",
                      "confidence": "high",
                      "status": "verified"
                    }
                  ]
                }
                """,
            },
        }
    )
    store.store({"type": "artifact.update", "name": "/report.md", "data": {"type": "file", "content": "# Report"}})
    store.store(
        {
            "type": "artifact.update",
            "name": "https://docs.x.ai/docs/grok-imagine",
            "data": {"type": "citation_use", "url": "https://docs.x.ai/docs/grok-imagine"},
        }
    )

    problems = _evaluate_post_run_quality(db_url, "job-quality-ledger-empty-entity")

    assert any("Claude Opus 4.6" in problem and "MiniMax M3" in problem for problem in problems)


def test_recovery_rejects_thinking_payload_and_combines_research_files(tmp_path):
    """Recovery must not promote raw MiniMax thinking blocks as final reports."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    store = EventStore(db_url, "job-recover-notes")
    thinking_payload = (
        "[{'thinking': 'I should now write the final report, but have not done so.', 'type': 'thinking', 'index': 0}]"
    )
    store.store(
        {
            "type": "artifact.update",
            "name": None,
            "data": {"type": "output", "output_category": "final_report", "content": thinking_payload},
        }
    )
    for url in ("https://example.com/source-a", "https://example.com/source-b"):
        store.store(
            {
                "type": "artifact.update",
                "name": url,
                "data": {"type": "citation_source", "content": url, "url": url},
            }
        )
    store.store(
        {
            "type": "artifact.update",
            "name": "/shared/first_research.md",
            "data": {
                "type": "file",
                "content": "# First Research\n\n" + ("Alpha findings with evidence. " * 80),
            },
        }
    )
    store.store(
        {
            "type": "artifact.update",
            "name": "/shared/second_research.txt",
            "data": {
                "type": "file",
                "content": "# Second Research\n\n" + ("Beta findings with trade-offs. " * 80),
            },
        }
    )

    recovered = _recover_report_from_events(db_url, "job-recover-notes")

    assert not _is_usable_report(thinking_payload)
    assert recovered is not None
    assert "first_research.md" in recovered
    assert "second_research.txt" in recovered
    assert "I should now write the final report" not in recovered


class TestIntermediateStepEvent:
    """Tests for the IntermediateStepEvent model."""

    def test_event_type_property(self):
        """Test event_type property generates category.state format."""
        event = IntermediateStepEvent(
            category=EventCategory.LLM,
            state=EventState.START,
            name="test-model",
        )

        assert event.event_type == "llm.start"

    def test_event_type_workflow_end(self):
        """Test event_type for workflow.end."""
        event = IntermediateStepEvent(
            category=EventCategory.WORKFLOW,
            state=EventState.END,
            name="researcher-agent",
        )

        assert event.event_type == "workflow.end"

    def test_event_type_tool_start(self):
        """Test event_type for tool.start."""
        event = IntermediateStepEvent(
            category=EventCategory.TOOL,
            state=EventState.START,
            name="web_search",
        )

        assert event.event_type == "tool.start"

    def test_to_sse_dict_basic(self):
        """Test to_sse_dict generates correct structure."""
        event = IntermediateStepEvent(
            category=EventCategory.LLM,
            state=EventState.START,
            name="test-model",
        )

        result = event.to_sse_dict()

        assert result["type"] == "llm.start"
        assert result["name"] == "test-model"
        assert "id" in result
        assert "timestamp" in result

    def test_to_sse_dict_with_data(self):
        """Test to_sse_dict includes data when present."""
        event = IntermediateStepEvent(
            category=EventCategory.TOOL,
            state=EventState.END,
            name="web_search",
            data=EventData(output="search results"),
        )

        result = event.to_sse_dict()

        assert result["data"] == {"output": "search results"}

    def test_to_sse_dict_with_metadata(self):
        """Test to_sse_dict includes metadata when present."""
        event = IntermediateStepEvent(
            category=EventCategory.LLM,
            state=EventState.END,
            name="test-model",
            metadata={"workflow": "researcher-agent", "thinking": "reasoning..."},
        )

        result = event.to_sse_dict()

        assert result["metadata"]["workflow"] == "researcher-agent"
        assert result["metadata"]["thinking"] == "reasoning..."

    def test_to_sse_dict_excludes_none_values(self):
        """Test to_sse_dict excludes None values."""
        event = IntermediateStepEvent(
            category=EventCategory.WORKFLOW,
            state=EventState.START,
            name=None,
        )

        result = event.to_sse_dict()

        assert "name" not in result

    def test_event_type_artifact_update(self):
        """Test event_type for artifact.update."""
        event = IntermediateStepEvent(
            category=EventCategory.ARTIFACT,
            state=EventState.UPDATE,
            name="researcher-agent",
            data=EventData(type="output", content="# Research findings..."),
        )

        assert event.event_type == "artifact.update"

    def test_artifact_category_exists(self):
        """Test ARTIFACT category is available."""
        assert EventCategory.ARTIFACT.value == "artifact"

    def test_update_state_exists(self):
        """Test UPDATE state is available (present tense)."""
        assert EventState.UPDATE.value == "update"

    def test_artifact_types_exist(self):
        """Test all ArtifactType values are available."""
        assert ArtifactType.FILE.value == "file"
        assert ArtifactType.OUTPUT.value == "output"
        assert ArtifactType.CITATION_SOURCE.value == "citation_source"
        assert ArtifactType.CITATION_USE.value == "citation_use"
        assert ArtifactType.TODO.value == "todo"


class TestDeepResearchEventCallback:
    """Tests for the DeepResearchEventCallback class."""

    def test_init_without_event_store(self):
        """Test initialization without event store."""
        callback = DeepResearchEventCallback()

        assert callback._event_store is None

    def test_init_with_event_store(self):
        """Test initialization with event store."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        assert callback._event_store == mock_store

    def test_emit_final_report_persists_report_md_file_and_output(self):
        """A verified final report should be durable as both /report.md and final output."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.emit_final_report("# Final Report\n\nComplete cited report content.")

        stored_events = [call[0][0] for call in mock_store.store.call_args_list]
        assert [event["type"] for event in stored_events] == ["artifact.update", "artifact.update"]
        assert stored_events[0]["name"] == "/report.md"
        assert stored_events[0]["data"]["type"] == "file"
        assert stored_events[0]["data"]["output_category"] == "final_report"
        assert stored_events[1]["data"]["type"] == "output"
        assert stored_events[1]["data"]["output_category"] == "final_report"

    def test_get_chain_name_from_serialized_name(self):
        """Test _get_chain_name extracts name from serialized dict."""
        callback = DeepResearchEventCallback()

        name = callback._get_chain_name({"name": "test_chain"})

        assert name == "test_chain"

    def test_get_chain_name_from_serialized_id(self):
        """Test _get_chain_name extracts name from id list."""
        callback = DeepResearchEventCallback()

        name = callback._get_chain_name({"id": ["module", "class", "chain_name"]})

        assert name == "chain_name"

    def test_get_chain_name_from_kwargs(self):
        """Test _get_chain_name falls back to kwargs."""
        callback = DeepResearchEventCallback()

        name = callback._get_chain_name(None, name="kwarg_name")

        assert name == "kwarg_name"

    def test_get_chain_name_default(self):
        """Test _get_chain_name returns 'unknown' as default."""
        callback = DeepResearchEventCallback()

        name = callback._get_chain_name(None)

        assert name == "unknown"

    def test_on_chain_start_with_agent_emits_workflow_start(self):
        """Test on_chain_start emits workflow.start event for agent chains."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_chain_start({"name": "planner-agent"}, inputs={})

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "workflow.start"
        assert call_args["name"] == "planner-agent"

    def test_on_chain_start_non_agent_chain_no_event(self):
        """Test on_chain_start does not emit for non-agent chains."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_chain_start({"name": "some_other_chain"}, inputs={})

        mock_store.store.assert_not_called()

    def test_on_chain_start_without_event_store(self):
        """Test on_chain_start does nothing without event store."""
        callback = DeepResearchEventCallback()

        callback.on_chain_start({"name": "planner-agent"}, inputs={})

    def test_on_chain_end_with_agent_emits_workflow_end(self):
        """Test on_chain_end emits workflow.end event for agent chains."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback._run_id_to_name["test-run-id"] = "researcher-agent"
        callback._agent_run_ids["test-run-id"] = ("researcher-agent", "test-run-id")

        callback.on_chain_end({}, run_id="test-run-id", name="researcher-agent")

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "workflow.end"
        assert call_args["name"] == "researcher-agent"

    def test_on_tool_start_emits_event(self):
        """Test on_tool_start emits tool.start event."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_tool_start({"name": "web_search"}, input_str="{'query': 'test'}")

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "tool.start"
        assert call_args["name"] == "web_search"
        assert "data" in call_args

    def test_on_tool_end_emits_event(self):
        """Test on_tool_end emits tool.end event."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback._run_id_to_name["test-run-id"] = "web_search"

        callback.on_tool_end("search results", run_id="test-run-id", name="web_search")

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "tool.end"
        assert call_args["name"] == "web_search"
        assert call_args["data"]["output"] == "search results"

    def test_write_plan_tool_start_does_not_emit_plan_file_artifact(self):
        """The UI should not see /shared/plan.json until write_plan succeeds."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)
        callback._agent_run_ids["planner-run"] = "planner-agent"
        callback._run_id_to_parent["tool-run"] = "planner-run"

        callback.on_tool_start(
            {"name": "write_plan"},
            input_str=str(
                {
                    "report_title": "AI Use Cases 2026",
                    "report_toc": [{"title": "Executive Summary"}, {"title": "Ranked Use Cases"}],
                    "queries": [{"query": "AI use cases 2026 analyst reports"}],
                    "constraints": ["Use authoritative sources for quantified impact."],
                }
            ),
            run_id="tool-run",
            parent_run_id="planner-run",
        )

        stored_events = [call[0][0] for call in mock_store.store.call_args_list]
        assert [event["type"] for event in stored_events] == ["tool.start"]
        assert not any(
            event["type"] == "artifact.update" and event.get("name") == "/shared/plan.json" for event in stored_events
        )

    def test_write_plan_tool_end_success_emits_plan_file_artifact(self):
        """The typed planner tool should surface /shared/plan.json after successful validation."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)
        callback._agent_run_ids["planner-run"] = "planner-agent"
        callback._run_id_to_parent["tool-run"] = "planner-run"

        callback.on_tool_start(
            {"name": "write_plan"},
            input_str=str(
                {
                    "report_title": "AI Use Cases 2026",
                    "report_toc": [{"title": "Executive Summary"}, {"title": "Ranked Use Cases"}],
                    "queries": [{"query": "AI use cases 2026 analyst reports"}],
                    "constraints": ["Use authoritative sources for quantified impact."],
                }
            ),
            run_id="tool-run",
            parent_run_id="planner-run",
        )
        callback.on_tool_end(
            "Plan successfully written and validated at /shared/plan.json. Sections: 2. Researcher queries: 1.",
            run_id="tool-run",
        )

        stored_events = [call[0][0] for call in mock_store.store.call_args_list]
        plan_artifact = next(
            event
            for event in stored_events
            if event["type"] == "artifact.update" and event["name"] == "/shared/plan.json"
        )

        assert plan_artifact["data"]["type"] == "file"
        assert plan_artifact["data"]["file_path"] == "/shared/plan.json"
        assert plan_artifact["data"]["filename"] == "plan.json"
        assert "AI Use Cases 2026" in plan_artifact["data"]["content"]
        assert plan_artifact["metadata"]["workflow"] == "planner-agent"

    def test_on_tool_start_without_event_store(self):
        """Test on_tool_start does nothing without event store."""
        callback = DeepResearchEventCallback()

        callback.on_tool_start({"name": "web_search"}, input_str="query")

    def test_on_tool_start_with_none_serialized(self):
        """Test on_tool_start handles None serialized dict."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_tool_start(None, input_str="query")

        call_args = mock_store.store.call_args[0][0]
        assert call_args["name"] == "unknown"

    def test_on_llm_start_emits_event(self):
        """Test on_llm_start emits llm.start event."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_llm_start({"name": "nemotron-70b"}, prompts=["test prompt"])

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "llm.start"
        assert call_args["name"] == "nemotron-70b"

    def test_on_chat_model_start_emits_event(self):
        """Test on_chat_model_start emits llm.start event."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_chat_model_start({"name": "gpt-4"}, messages=[])

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "llm.start"
        assert call_args["name"] == "gpt-4"

    def test_on_llm_new_token_extracts_minimax_thinking_block(self):
        """MiniMax can stream Anthropic-style thinking blocks instead of strings."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_llm_new_token([{"type": "thinking", "thinking": "checking fresh sources", "index": 0}])

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["type"] == "llm.chunk"
        assert call_args["data"]["chunk"] == "checking fresh sources"

    def test_on_llm_new_token_drops_signature_only_thinking_block(self):
        """Signature-only MiniMax stream blocks should not reach the UI."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_llm_new_token([{"type": "thinking", "signature": "abc123", "index": 0}])

        mock_store.store.assert_not_called()

    def test_on_llm_new_token_drops_degenerate_repetition_chunk(self):
        """Provider repetition loops should not flood persisted SSE events."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback.on_llm_new_token(" puppet" * 30)

        mock_store.store.assert_not_called()

    def test_on_llm_end_normalizes_object_thinking_metadata(self):
        """llm.end metadata.thinking must always be a string for React rendering."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        mock_message = MagicMock()
        mock_message.content = "Response content"
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {
            "thinking": {"type": "thinking", "thinking": "structured reasoning", "index": 0}
        }
        mock_message.response_metadata = {}
        mock_message.usage_metadata = None

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        callback._run_id_to_name["run-1"] = "MiniMax-M3"
        callback.on_llm_end(response=mock_result, run_id="run-1")

        call_args = mock_store.store.call_args_list[0][0][0]
        assert call_args["type"] == "llm.end"
        assert call_args["metadata"]["thinking"] == "structured reasoning"

    def test_on_llm_end_does_not_emit_thinking_only_output_artifact(self):
        """MiniMax thinking-only final turns must not be persisted as report drafts."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        mock_message = MagicMock()
        mock_message.content = [{"type": "thinking", "thinking": "I should write the report now.", "index": 0}]
        mock_message.tool_calls = None
        mock_message.additional_kwargs = {}
        mock_message.response_metadata = {}
        mock_message.usage_metadata = None

        mock_generation = MagicMock()
        mock_generation.message = mock_message

        mock_result = MagicMock()
        mock_result.generations = [[mock_generation]]

        callback._run_id_to_name["run-1"] = "MiniMax-M3"
        callback.on_llm_end(response=mock_result, run_id="run-1")

        stored_events = [call[0][0] for call in mock_store.store.call_args_list]
        assert [event["type"] for event in stored_events] == ["llm.end"]


class TestSubmitDeepResearchJob:
    """Tests for the submit_deep_research_job function."""

    principal = Principal(type="test", sub="user-1", email="test@example.com", name="Test User")

    @pytest.mark.asyncio
    async def test_submit_without_scheduler_raises(self):
        """Test submit_deep_research_job raises without NAT_DASK_SCHEDULER_ADDRESS."""
        from aiq_api.jobs.submit import submit_deep_research_job

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="NAT_DASK_SCHEDULER_ADDRESS"):
                await submit_deep_research_job(
                    input_text="test query",
                    owner="test@example.com",
                )

    @pytest.mark.asyncio
    async def test_submit_with_scheduler(self):
        """Test submit_deep_research_job submits job successfully."""
        from aiq_api.jobs.submit import submit_deep_research_job

        mock_job_store = MagicMock()
        mock_job_store.ensure_job_id.return_value = "test-job-id"
        mock_job_store.submit_job = AsyncMock(return_value=None)

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
                "NAT_JOB_STORE_DB_URL": "sqlite:///./test.db",
                "NAT_CONFIG_PATH": "/path/to/config.yml",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=self.principal):
                    with patch("aiq_api.jobs.submit.create_job_access"):
                        result = await submit_deep_research_job(
                            input_text="test query",
                            owner="test@example.com",
                        )

        assert result == "test-job-id"
        mock_job_store.submit_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_agent_job_passes_data_sources(self):
        """Test submit_agent_job forwards data_sources into worker args."""
        from aiq_api.jobs.submit import submit_agent_job

        mock_job_store = MagicMock()
        mock_job_store.ensure_job_id.return_value = "test-job-id"
        mock_job_store.submit_job = AsyncMock(return_value=None)

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
                "NAT_JOB_STORE_DB_URL": "sqlite:///./test.db",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=self.principal):
                    with patch("aiq_api.jobs.submit.create_job_access"):
                        result = await submit_agent_job(
                            agent_type="deep_researcher",
                            input_text="test query",
                            owner="test@example.com",
                            data_sources=["web_search"],
                        )

        assert result == "test-job-id"
        mock_job_store.submit_job.assert_called_once()
        job_args = mock_job_store.submit_job.call_args.kwargs["job_args"]
        # Tail args are available_documents, data_sources, auth_token, research_depth.
        assert job_args[-3] == ["web_search"]

    @pytest.mark.asyncio
    async def test_submit_agent_job_passes_research_depth(self):
        """Test submit_agent_job forwards research_depth into worker args."""
        from aiq_api.jobs.submit import submit_agent_job

        mock_job_store = MagicMock()
        mock_job_store.ensure_job_id.return_value = "test-job-id"
        mock_job_store.submit_job = AsyncMock(return_value=None)

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
                "NAT_JOB_STORE_DB_URL": "sqlite:///./test.db",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=self.principal):
                    with patch("aiq_api.jobs.submit.create_job_access"):
                        result = await submit_agent_job(
                            agent_type="deep_researcher",
                            input_text="test query",
                            owner="test@example.com",
                            research_depth="deep",
                        )

        assert result == "test-job-id"
        mock_job_store.submit_job.assert_called_once()
        job_args = mock_job_store.submit_job.call_args.kwargs["job_args"]
        assert job_args[-1] == "deep"

    @pytest.mark.asyncio
    async def test_submit_with_custom_job_id(self):
        """Test submit_deep_research_job uses custom job ID."""
        from aiq_api.jobs.submit import submit_deep_research_job

        mock_job_store = MagicMock()
        mock_job_store.ensure_job_id.return_value = "custom-job-id"
        mock_job_store.submit_job = AsyncMock(return_value=None)

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=self.principal):
                    with patch("aiq_api.jobs.submit.create_job_access"):
                        result = await submit_deep_research_job(
                            input_text="test query",
                            owner="test@example.com",
                            job_id="custom-job-id",
                        )

        assert result == "custom-job-id"
        mock_job_store.ensure_job_id.assert_called_with("custom-job-id")

    @pytest.mark.asyncio
    async def test_submit_requires_verified_principal(self):
        """Test submit_agent_job fails closed when no verified principal is available."""
        from aiq_api.jobs.submit import submit_agent_job

        mock_job_store = MagicMock()

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
                "REQUIRE_AUTH": "true",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=None):
                    with pytest.raises(RuntimeError, match="Verified current principal required"):
                        await submit_agent_job(
                            agent_type="deep_researcher",
                            input_text="test query",
                            owner="test@example.com",
                        )

        mock_job_store.submit_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_uses_compatibility_principal_when_auth_disabled(self):
        """Test submit_agent_job still works without verified principal when auth is disabled."""
        from aiq_api.jobs.submit import submit_agent_job

        mock_job_store = MagicMock()
        mock_job_store.ensure_job_id.return_value = "test-job-id"
        mock_job_store.submit_job = AsyncMock(return_value=None)

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
                "NAT_JOB_STORE_DB_URL": "sqlite:///./test.db",
                "REQUIRE_AUTH": "false",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=None):
                    with patch(
                        "aiq_api.jobs.submit.create_job_access",
                    ) as create_job_access:
                        result = await submit_agent_job(
                            agent_type="deep_researcher",
                            input_text="test query",
                            owner="test@example.com",
                        )

        assert result == "test-job-id"
        create_job_access.assert_called_once()
        principal = create_job_access.call_args.args[1]
        assert principal.type == "internal"
        assert principal.sub == "test@example.com"
        assert principal.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_submit_rolls_back_when_job_access_persistence_fails(self):
        """Test submit_agent_job rolls back partial submission on access persistence failure."""
        from aiq_api.jobs.submit import submit_agent_job

        mock_job_store = MagicMock()
        mock_job_store.ensure_job_id.return_value = "test-job-id"
        mock_job_store.submit_job = AsyncMock(return_value=None)

        with patch.dict(
            "os.environ",
            {
                "NAT_DASK_SCHEDULER_ADDRESS": "tcp://localhost:8786",
                "NAT_JOB_STORE_DB_URL": "sqlite:///./test.db",
            },
        ):
            with patch("nat.front_ends.fastapi.async_jobs.job_store.JobStore", return_value=mock_job_store):
                with patch("aiq_api.jobs.submit.get_current_principal", return_value=self.principal):
                    with patch(
                        "aiq_api.jobs.submit.create_job_access",
                        side_effect=RuntimeError("db write failed"),
                    ):
                        with patch("aiq_api.jobs.submit.rollback_job_submission") as rollback_job_submission:
                            with pytest.raises(RuntimeError, match="db write failed"):
                                await submit_agent_job(
                                    agent_type="deep_researcher",
                                    input_text="test query",
                                    owner="test@example.com",
                                )

        mock_job_store.submit_job.assert_called_once()
        rollback_job_submission.assert_called_once_with("test-job-id", "sqlite:///./test.db")


class TestEventStore:
    """Tests for the EventStore class."""

    def test_store_event(self, tmp_path):
        """Test storing an event."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        store = EventStore(db_url, "test-job-1")
        store.store({"type": "test.event", "data": {"key": "value"}})

        events = EventStore.get_events(db_url, "test-job-1")
        assert len(events) == 1
        assert events[0]["type"] == "test.event"

    def test_get_events_empty(self, tmp_path):
        """Test get_events returns empty list for unknown job."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        EventStore._ensure_table_exists(db_url)
        events = EventStore.get_events(db_url, "nonexistent-job")
        assert events == []

    def test_get_events_with_after_id(self, tmp_path):
        """Test get_events with after_id cursor."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        store = EventStore(db_url, "test-job-2")
        store.store({"type": "event.1"})
        store.store({"type": "event.2"})
        store.store({"type": "event.3"})

        all_events = EventStore.get_events(db_url, "test-job-2")
        assert len(all_events) == 3

        after_first = EventStore.get_events(db_url, "test-job-2", after_id=all_events[0]["_id"])
        assert len(after_first) == 2
        assert after_first[0]["type"] == "event.2"

    @pytest.mark.asyncio
    async def test_get_events_async(self, tmp_path):
        """Test async get_events."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        store = EventStore(db_url, "async-job")
        store.store({"type": "async.event"})

        events = await EventStore.get_events_async(db_url, "async-job")
        assert len(events) == 1
        await EventStore.dispose_all_engines_async()

    def test_cleanup_job_events(self, tmp_path):
        """Test cleanup_job_events deletes events."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        store = EventStore(db_url, "cleanup-job")
        store.store({"type": "event.1"})
        store.store({"type": "event.2"})

        deleted = EventStore.cleanup_job_events(db_url, "cleanup-job")
        assert deleted == 2

        events = EventStore.get_events(db_url, "cleanup-job")
        assert len(events) == 0

    def test_engine_caching(self, tmp_path):
        """Test that engines are cached and reused."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        EventStore._sync_engine_cache.clear()

        store1 = EventStore(db_url, "job-1")
        engine1 = store1._sync_engine

        store2 = EventStore(db_url, "job-2")
        engine2 = store2._sync_engine

        assert engine1 is engine2

    def test_dispose_all_engines(self, tmp_path):
        """Test dispose_all_engines clears cache."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"

        EventStore(db_url, "test-job")
        assert len(EventStore._sync_engine_cache) > 0

        EventStore.dispose_all_engines()
        assert len(EventStore._sync_engine_cache) == 0

    @pytest.mark.asyncio
    async def test_dispose_all_engines_async_disposes_all(self):
        """Test dispose_all_engines_async disposes sync and async engines."""
        from aiq_api.jobs.event_store import EventStore

        sync_engine = MagicMock()
        async_engine = MagicMock()
        async_engine.dispose = AsyncMock()

        EventStore._sync_engine_cache = {"sync-db": (sync_engine, 0)}
        EventStore._async_engine_cache = {"async-db": (async_engine, 0)}
        EventStore._tables_initialized.add("sqlite:///test.db")

        await EventStore.dispose_all_engines_async()

        sync_engine.dispose.assert_called_once()
        async_engine.dispose.assert_awaited_once()
        assert EventStore._sync_engine_cache == {}
        assert EventStore._async_engine_cache == {}
        assert EventStore._tables_initialized == set()

    def test_dispose_all_engines_schedules_async_cleanup(self):
        """Test dispose_all_engines schedules async dispose with running loop."""
        from aiq_api.jobs.event_store import EventStore

        sync_engine = MagicMock()
        async_engine = MagicMock()
        async_engine.dispose = AsyncMock()
        loop = MagicMock()

        def run_coroutine(coro):
            import asyncio

            temp_loop = asyncio.new_event_loop()
            try:
                temp_loop.run_until_complete(coro)
            finally:
                temp_loop.close()

        EventStore._sync_engine_cache = {"sync-db": (sync_engine, 0)}
        EventStore._async_engine_cache = {"async-db": (async_engine, 0)}

        with patch("asyncio.get_running_loop", return_value=loop):
            loop.create_task.side_effect = run_coroutine
            EventStore.dispose_all_engines()

        sync_engine.dispose.assert_called_once()
        loop.create_task.assert_called_once()
        async_engine.dispose.assert_called_once()
        assert EventStore._sync_engine_cache == {}
        assert EventStore._async_engine_cache == {}

    def test_cleanup_stale_engines_disposes_async_engine(self):
        """Test stale async engines are disposed with a running loop."""
        from aiq_api.jobs.event_store import ENGINE_CACHE_TTL_SECONDS
        from aiq_api.jobs.event_store import EventStore

        async_engine = MagicMock()
        async_engine.dispose = AsyncMock()
        loop = MagicMock()
        cache = {"async-db": (async_engine, 0)}

        def run_coroutine(coro):
            import asyncio

            temp_loop = asyncio.new_event_loop()
            try:
                temp_loop.run_until_complete(coro)
            finally:
                temp_loop.close()

        with patch("time.monotonic", return_value=ENGINE_CACHE_TTL_SECONDS + 1):
            with patch("asyncio.get_running_loop", return_value=loop):
                loop.create_task.side_effect = run_coroutine
                EventStore._cleanup_stale_engines(cache)

        loop.create_task.assert_called_once()
        async_engine.dispose.assert_called_once()
        assert cache == {}

    def test_cleanup_stale_engines_uses_asyncio_run_without_loop(self):
        """Test stale async engines use asyncio.run without a loop."""
        from aiq_api.jobs.event_store import ENGINE_CACHE_TTL_SECONDS
        from aiq_api.jobs.event_store import EventStore

        async_engine = MagicMock()
        async_engine.dispose = AsyncMock()
        cache = {"async-db": (async_engine, 0)}
        run_calls: list[bool] = []

        def run_coroutine(coro):
            import asyncio

            run_calls.append(True)
            temp_loop = asyncio.new_event_loop()
            try:
                temp_loop.run_until_complete(coro)
            finally:
                temp_loop.close()

        with patch("time.monotonic", return_value=ENGINE_CACHE_TTL_SECONDS + 1):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError):
                with patch("asyncio.run", side_effect=run_coroutine) as run:
                    EventStore._cleanup_stale_engines(cache)

        run.assert_called_once()
        async_engine.dispose.assert_called_once()
        assert run_calls == [True]
        assert cache == {}


class TestToolArtifactMapping:
    """Tests for the ToolArtifactMapping class."""

    def test_default_mappings(self):
        """Test default tool mappings are registered."""
        from aiq_api.jobs.callbacks import ToolArtifactMapping

        mapping = ToolArtifactMapping()

        assert mapping.is_artifact_tool("write_todos")
        assert mapping.is_artifact_tool("write_file")
        assert not mapping.is_artifact_tool("unknown_tool")

    def test_get_mapping(self):
        """Test get_mapping returns correct mapping."""
        from aiq_api.jobs.callbacks import ArtifactType
        from aiq_api.jobs.callbacks import ToolArtifactMapping

        mapping = ToolArtifactMapping()
        todo_mapping = mapping.get_mapping("write_todos")

        assert todo_mapping is not None
        assert todo_mapping["artifact_type"] == ArtifactType.TODO

    def test_register_custom_mapping(self):
        """Test registering a custom tool mapping."""
        from aiq_api.jobs.callbacks import ArtifactType
        from aiq_api.jobs.callbacks import ToolArtifactMapping

        mapping = ToolArtifactMapping()
        mapping.register(
            "custom_tool",
            artifact_type=ArtifactType.OUTPUT,
            content_key="result",
        )

        assert mapping.is_artifact_tool("custom_tool")
        custom = mapping.get_mapping("custom_tool")
        assert custom["artifact_type"] == ArtifactType.OUTPUT


class TestDeepResearchEventCallbackAdvanced:
    """Additional tests for DeepResearchEventCallback."""

    def test_extract_urls(self):
        """Test URL extraction from text."""
        callback = DeepResearchEventCallback()

        text = "Check out https://example.com and http://test.org/page for more info."
        urls = callback._extract_urls(text)

        assert "https://example.com" in urls
        assert "http://test.org/page" in urls

    def test_extract_urls_cleans_trailing_punctuation(self):
        """Test URL extraction removes trailing punctuation."""
        callback = DeepResearchEventCallback()

        text = "Visit https://example.com)."
        urls = callback._extract_urls(text)

        assert urls == ["https://example.com"]

    def test_is_search_tool(self):
        """Test search tool detection."""
        callback = DeepResearchEventCallback()

        assert callback._is_search_tool("tavily_search")
        assert callback._is_search_tool("web_search_tool")
        assert callback._is_search_tool("google_search")
        assert not callback._is_search_tool("write_file")

    def test_contains_tool_call_syntax(self):
        """Test tool call syntax detection."""
        callback = DeepResearchEventCallback()

        # Pattern matches quoted arguments and keyword arguments
        assert callback._contains_tool_call_syntax('Let me call task("query")')
        assert callback._contains_tool_call_syntax("Let me call task(query=value)")
        assert not callback._contains_tool_call_syntax("Normal text without calls")
        # Bare positional arguments don't match to avoid false positives
        assert not callback._contains_tool_call_syntax("Let me call task(query)")

    def test_emit_artifact_adds_workflow_metadata(self):
        """Test that artifact emission includes workflow metadata when provided."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)

        callback._emit_artifact(
            ArtifactType.OUTPUT,
            "test content",
            workflow_source="test-agent",
            agent_id="run-1",
        )

        mock_store.store.assert_called_once()
        call_args = mock_store.store.call_args[0][0]
        assert call_args["metadata"]["workflow"] == "test-agent"
        assert call_args["metadata"]["agent_id"] == "run-1"

    def test_on_chain_end_clears_agent_tracking(self):
        """Test on_chain_end removes agent from tracking when matching."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)
        callback._agent_run_ids["run-1"] = ("researcher-agent", "run-1")
        callback._run_id_to_name["run-1"] = "researcher-agent"

        callback.on_chain_end({}, run_id="run-1", name="researcher-agent")

        assert "run-1" not in callback._agent_run_ids
        assert len(callback._agent_run_ids) == 0

    def test_on_tool_end_extracts_search_urls(self):
        """Test on_tool_end extracts URLs from search tool results."""
        mock_store = MagicMock()
        callback = DeepResearchEventCallback(event_store=mock_store)
        callback._run_id_to_name["run-1"] = "tavily_search"

        callback.on_tool_end("Found: https://example.com/result", run_id="run-1")

        assert "https://example.com/result" in callback._discovered_urls
        assert mock_store.store.call_count >= 2

    def test_parse_tool_input_dict_string(self):
        """Test parsing dict-like string input."""
        callback = DeepResearchEventCallback()

        result = callback._parse_tool_input("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_parse_tool_input_plain_string(self):
        """Test parsing plain string input."""
        callback = DeepResearchEventCallback()

        result = callback._parse_tool_input("plain text query")
        assert result == "plain text query"

    def test_extract_input_with_messages(self):
        """Test extracting input from dict with messages."""
        callback = DeepResearchEventCallback()
        msg = MagicMock()
        msg.content = "message content"

        result = callback._extract_input({"messages": [msg]})
        assert result == "message content"

    def test_extract_output_with_output_key(self):
        """Test extracting output from dict with output key."""
        callback = DeepResearchEventCallback()

        result = callback._extract_output({"output": "the result"})
        assert result == "the result"


class TestCancellationMonitor:
    """Tests for CancellationMonitor."""

    def test_init(self):
        """Test CancellationMonitor initialization."""
        from aiq_api.jobs.runner import CancellationMonitor

        monitor = CancellationMonitor(
            scheduler_address="tcp://localhost:8786",
            db_url="sqlite:///test.db",
            job_id="test-job",
        )

        assert monitor.scheduler_address == "tcp://localhost:8786"
        assert monitor.job_id == "test-job"
        assert not monitor.is_cancelled

    def test_is_cancelled_initially_false(self):
        """Test is_cancelled is initially False."""
        from aiq_api.jobs.runner import CancellationMonitor

        monitor = CancellationMonitor(
            scheduler_address="tcp://localhost:8786",
            db_url="sqlite:///test.db",
            job_id="test-job",
        )

        assert monitor.is_cancelled is False

    def test_check_raises_when_cancelled(self):
        """Test check() raises CancelledError when cancelled."""
        import asyncio

        from aiq_api.jobs.runner import CancellationMonitor

        monitor = CancellationMonitor(
            scheduler_address="tcp://localhost:8786",
            db_url="sqlite:///test.db",
            job_id="test-job",
        )
        monitor._cancelled.set()

        with pytest.raises(asyncio.CancelledError):
            monitor.check()

    def test_stop_cancels_monitor_task(self):
        """Test stop() cancels the monitor task."""
        from aiq_api.jobs.runner import CancellationMonitor

        monitor = CancellationMonitor(
            scheduler_address="tcp://localhost:8786",
            db_url="sqlite:///test.db",
            job_id="test-job",
        )
        mock_task = MagicMock()
        mock_task.done.return_value = False
        monitor._monitor_task = mock_task

        monitor.stop()

        mock_task.cancel.assert_called_once()
        assert monitor._monitor_task is None


class TestDataSourceModel:
    """Tests for the DataSource Pydantic model."""

    def test_data_source_basic_creation(self):
        """Test creating DataSource with required fields."""
        from aiq_api.routes.jobs import DataSource

        source = DataSource(id="web_search", name="Web Search")

        assert source.id == "web_search"
        assert source.name == "Web Search"
        assert source.description is None

    def test_data_source_with_description(self):
        """Test creating DataSource with description."""
        from aiq_api.routes.jobs import DataSource

        source = DataSource(
            id="confluence",
            name="Atlassian Confluence",
            description="Enterprise content from Confluence.",
        )

        assert source.id == "confluence"
        assert source.name == "Atlassian Confluence"
        assert source.description == "Enterprise content from Confluence."

    def test_data_source_serialization(self):
        """Test DataSource serialization to dict."""
        from aiq_api.routes.jobs import DataSource

        source = DataSource(
            id="sharepoint",
            name="Microsoft SharePoint",
            description="Enterprise docs.",
        )

        data = source.model_dump()
        assert data["id"] == "sharepoint"
        assert data["name"] == "Microsoft SharePoint"
        assert data["description"] == "Enterprise docs."


class TestJobErrorEventEmission:
    """Tests for job.error event emission on job failure."""

    @pytest.mark.asyncio
    async def test_exception_emits_job_error_event(self, tmp_path):
        """Test that exceptions emit job.error events to event store."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "error_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "error-test-job"

        event_store = EventStore(db_url, job_id)
        test_error = ValueError("Test error message")

        event_store.store(
            {
                "type": "job.error",
                "data": {
                    "error": str(test_error),
                    "error_type": type(test_error).__name__,
                },
            }
        )

        events = EventStore.get_events(db_url, job_id)
        assert len(events) == 1
        assert events[0]["type"] == "job.error"
        assert events[0]["data"]["error"] == "Test error message"
        assert events[0]["data"]["error_type"] == "ValueError"

    def test_job_error_event_structure(self, tmp_path):
        """Test job.error event has correct structure."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "structure_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "structure-test-job"

        event_store = EventStore(db_url, job_id)

        event_store.store(
            {
                "type": "job.error",
                "data": {
                    "error": "Connection timeout",
                    "error_type": "TimeoutError",
                },
            }
        )

        events = EventStore.get_events(db_url, job_id)

        assert len(events) == 1
        event = events[0]
        assert "type" in event
        assert "data" in event
        assert "error" in event["data"]
        assert "error_type" in event["data"]
        assert "_id" in event

    def test_job_error_preserves_error_type(self, tmp_path):
        """Test that different error types are preserved correctly."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "types_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "types-test-job"

        event_store = EventStore(db_url, job_id)

        error_types = [
            (TimeoutError("timed out"), "TimeoutError"),
            (RuntimeError("runtime issue"), "RuntimeError"),
            (KeyError("missing key"), "KeyError"),
            (ConnectionError("connection lost"), "ConnectionError"),
        ]

        for error, expected_type in error_types:
            event_store.store(
                {
                    "type": "job.error",
                    "data": {
                        "error": str(error),
                        "error_type": type(error).__name__,
                    },
                }
            )

        events = EventStore.get_events(db_url, job_id)
        assert len(events) == 4

        for i, (_, expected_type) in enumerate(error_types):
            assert events[i]["data"]["error_type"] == expected_type

    def test_job_error_with_long_message(self, tmp_path):
        """Test job.error handles long error messages."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "long_error_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "long-error-job"

        event_store = EventStore(db_url, job_id)

        long_error_msg = "Error: " + "A" * 10000

        event_store.store(
            {
                "type": "job.error",
                "data": {
                    "error": long_error_msg,
                    "error_type": "ValueError",
                },
            }
        )

        events = EventStore.get_events(db_url, job_id)
        assert len(events) == 1
        assert events[0]["data"]["error"] == long_error_msg

    @pytest.mark.asyncio
    async def test_job_error_async_retrieval(self, tmp_path):
        """Test job.error events can be retrieved asynchronously."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "async_error_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "async-error-job"

        event_store = EventStore(db_url, job_id)

        event_store.store(
            {
                "type": "job.error",
                "data": {
                    "error": "Async test error",
                    "error_type": "AsyncError",
                },
            }
        )

        events = await EventStore.get_events_async(db_url, job_id)
        assert len(events) == 1
        assert events[0]["type"] == "job.error"
        assert events[0]["data"]["error_type"] == "AsyncError"
        await EventStore.dispose_all_engines_async()


class TestJobCancelledEventComparison:
    """Tests comparing job.cancelled and job.error event patterns."""

    def test_cancelled_and_error_events_coexist(self, tmp_path):
        """Test that cancelled and error events can coexist for same job."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "coexist_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "coexist-job"

        event_store = EventStore(db_url, job_id)

        event_store.store({"type": "job.cancelled", "data": {"reason": "user cancelled"}})
        event_store.store({"type": "job.error", "data": {"error": "cleanup failed", "error_type": "RuntimeError"}})

        events = EventStore.get_events(db_url, job_id)
        assert len(events) == 2
        types = {e["type"] for e in events}
        assert "job.cancelled" in types
        assert "job.error" in types

    def test_job_error_follows_status_event_pattern(self, tmp_path):
        """Test job.error follows same pattern as job.cancelled."""
        from aiq_api.jobs.event_store import EventStore

        db_path = tmp_path / "pattern_test.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "pattern-job"

        event_store = EventStore(db_url, job_id)

        cancelled_event = {"type": "job.cancelled", "data": {"reason": "cancelled by user"}}
        error_event = {
            "type": "job.error",
            "data": {"error": "test error", "error_type": "TestError"},
        }

        event_store.store(cancelled_event)
        event_store.store(error_event)

        events = EventStore.get_events(db_url, job_id)

        for event in events:
            assert "type" in event
            assert event["type"].startswith("job.")
            assert "data" in event


class TestReportRecovery:
    """Tests for async-job report selection and recovery."""

    def test_prefer_report_ignores_model_failure_artifact(self):
        """A model-call error artifact should not override a real report."""
        from aiq_api.jobs.runner import _prefer_report

        report = "# Real Report\n\n## Findings\n\nUseful content.\n\n## Sources\n[1] Example: https://example.com"
        failure = "Model call failed after 2 attempts with APIConnectionError: Connection error."

        assert _prefer_report(report, failure) == report

    def test_extract_report_from_events_skips_model_failure_final_report(self, tmp_path):
        """Report recovery should choose report.md over a later final_report error string."""
        from aiq_api.jobs.event_store import EventStore
        from aiq_api.jobs.runner import _extract_report_from_events

        db_path = tmp_path / "report_recovery.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "report-recovery-job"
        event_store = EventStore(db_url, job_id)
        report = "# Real Report\n\n## Findings\n\nUseful content.\n\n## Sources\n[1] Example: https://example.com"

        event_store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "file",
                    "filename": "/report.md",
                    "content": report,
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "output",
                    "output_category": "final_report",
                    "content": "Model call failed after 2 attempts with APIConnectionError: Connection error.",
                },
            }
        )

        assert _extract_report_from_events(db_url, job_id) == report

    def test_search_result_snapshots_do_not_count_as_final_report(self, tmp_path):
        """Raw search snapshots are resume material, not a polished final report."""
        from aiq_api.jobs.event_store import EventStore
        from aiq_api.jobs.runner import _recover_report_from_events

        db_path = tmp_path / "search_snapshot_recovery.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "search-snapshot-job"
        event_store = EventStore(db_url, job_id)
        search_snapshot = (
            "Result 1: Agent UI dashboards need status, handoff controls, progress timelines, "
            "error recovery states, and transparent reasoning traces. https://example.com/agent-ui\n"
        ) * 20

        event_store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "citation_source",
                    "content": "https://example.com/agent-ui\\nPublished",
                    "url": "https://example.com/agent-ui\\nPublished",
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "output",
                    "output_category": "search_result",
                    "content": search_snapshot,
                },
            }
        )

        assert _recover_report_from_events(db_url, job_id) is None

    def test_recovered_consolidated_file_appends_source_inventory(self, tmp_path):
        """Recovered consolidated notes should preserve collected URLs."""
        from aiq_api.jobs.event_store import EventStore
        from aiq_api.jobs.runner import _recover_report_from_events

        db_path = tmp_path / "consolidated_recovery.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "consolidated-recovery-job"
        event_store = EventStore(db_url, job_id)

        event_store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "citation_source",
                    "content": "https://example.com/source-a",
                    "url": "https://example.com/source-a",
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "data": {
                    "type": "citation_source",
                    "content": "https://example.com/source-b",
                    "url": "https://example.com/source-b",
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "name": "/shared/consolidated_findings.md",
                "data": {
                    "type": "file",
                    "content": ("# Consolidated Findings\n\nImportant evidence from one branch.\n\n" * 30).strip(),
                },
            }
        )

        recovered = _recover_report_from_events(db_url, job_id)

        assert recovered is not None
        assert "## Additional Sources Collected During Run" in recovered
        assert "https://example.com/source-a" in recovered
        assert "https://example.com/source-b" in recovered

    def test_recovery_rejects_off_topic_intermediate_file(self, tmp_path):
        """A university query should not recover a career-pathways artifact as the final report."""
        from aiq_api.jobs.event_store import EventStore
        from aiq_api.jobs.runner import _recover_report_from_events

        db_path = tmp_path / "off_topic_recovery.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "off-topic-recovery-job"
        event_store = EventStore(db_url, job_id)

        event_store.store(
            {
                "type": "job.submitted",
                "data": {
                    "agent_type": "deep_researcher",
                    "input": "University education costs and outcomes for a debate lesson",
                    "owner": "tester@example.com",
                    "data_sources": [],
                    "research_depth": "shallow",
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "name": "/shared/alternative_pathways_primary_debate_lesson.txt",
                "data": {
                    "type": "file",
                    "content": (
                        "# Alternate Ways to Get to Your Career\n\n"
                        "This guide is about apprenticeships, VET, jobs without a degree, and other alternatives."
                    )
                    * 40,
                },
            }
        )

        assert _recover_report_from_events(db_url, job_id) is None

    def test_deep_research_recovery_compiles_intermediate_notes(self, tmp_path):
        """Deep-research jobs should compile notes, not serve a raw recovered bundle."""
        from aiq_api.jobs.event_store import EventStore
        from aiq_api.jobs.runner import _recover_report_from_events

        db_path = tmp_path / "shallow_tier_intermediate_recovery.db"
        db_url = f"sqlite:///{db_path}"
        job_id = "shallow-tier-intermediate-recovery-job"
        event_store = EventStore(db_url, job_id)

        event_store.store(
            {
                "type": "job.submitted",
                "data": {
                    "agent_type": "deep_researcher",
                    "input": "University education value debate for primary students",
                    "owner": "tester@example.com",
                    "data_sources": [],
                    "research_depth": "shallow",
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "name": "https://example.com/university-outcomes",
                "data": {
                    "type": "citation_source",
                    "content": "https://example.com/university-outcomes",
                    "url": "https://example.com/university-outcomes",
                },
            }
        )
        event_store.store(
            {
                "type": "artifact.update",
                "name": "/shared/university_education_debate_research.txt",
                "data": {
                    "type": "file",
                    "content": (
                        "# University Education Value Debate - Primary Student Research Notes\n\n"
                        "Evidence about university education, costs, outcomes, and student debate framing. "
                    )
                    * 60,
                },
            }
        )

        recovered = _recover_report_from_events(db_url, job_id)

        assert recovered is not None
        assert recovered.startswith("# University education value debate")
        assert "## Synthesis Status" in recovered
        assert "Recovered Research Report" not in recovered
        assert "university education" in recovered.lower()
        assert "https://example.com/university-outcomes" in recovered

    def test_recovered_reference_urls_exclude_source_inventory(self):
        """Citation-use recovery should only count report references, not the raw appendix."""
        from aiq_api.jobs.runner import _extract_report_reference_urls

        report = (
            "# Report\n\n"
            "Finding [1].\n\n"
            "## Sources\n\n"
            "[1] Primary: https://example.com/primary\n\n"
            "## Additional Sources Collected During Run\n\n"
            "- https://example.com/raw-discovery\n"
        )

        assert _extract_report_reference_urls(report) == ["https://example.com/primary"]


class TestSQLAlchemyPoolFilter:
    """Tests for SQLAlchemyPoolFilter."""

    def test_filter_passes_normal_errors(self):
        """Test filter passes normal error messages."""
        import logging

        from aiq_api.jobs.event_store import SQLAlchemyPoolFilter

        filter_obj = SQLAlchemyPoolFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Normal error message",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is True

    def test_filter_blocks_cancelled_errors(self):
        """Test filter blocks CancelledError messages."""
        import logging

        from aiq_api.jobs.event_store import SQLAlchemyPoolFilter

        filter_obj = SQLAlchemyPoolFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="CancelledError occurred",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is False

    def test_filter_passes_info_level(self):
        """Test filter passes INFO level messages."""
        import logging

        from aiq_api.jobs.event_store import SQLAlchemyPoolFilter

        filter_obj = SQLAlchemyPoolFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="CancelledError info",
            args=(),
            exc_info=None,
        )

        assert filter_obj.filter(record) is True


class TestAsyncJobRunnerAgentFactory:
    """Tests for async job agent construction."""

    def test_resolve_tool_refs_skips_toolless_agent_configs(self):
        """Claude Code bridge configs should not trigger LangChain tool loading."""
        from aiq_agent.agents.claude_researcher.register import ClaudeResearchAgentConfig
        from aiq_api.jobs.runner import _resolve_tool_refs

        assert _resolve_tool_refs(ClaudeResearchAgentConfig()) == []

    def test_resolve_tool_refs_autoinherits_for_tool_aware_configs(self):
        """Traditional AIQ configs still inherit registered source tools when empty."""
        from aiq_agent.agents.deep_researcher.register import DeepResearchAgentConfig
        from aiq_api.jobs.runner import _resolve_tool_refs

        with patch("aiq_agent.common.get_all_tool_refs", return_value=["tool_a", "tool_b"]):
            assert _resolve_tool_refs(DeepResearchAgentConfig(orchestrator_llm="llm")) == ["tool_a", "tool_b"]

    def test_create_agent_instance_passes_config_and_job_id_when_supported(self):
        """Async workers can receive generic function config without runner-specific agent knowledge."""
        from aiq_agent.agents.deep_researcher.deepagents_runtime import SandboxConfig
        from aiq_agent.agents.deep_researcher.deepagents_runtime import SkillsConfig
        from aiq_agent.agents.deep_researcher.register import DeepResearchAgentConfig
        from aiq_api.jobs.runner import _create_agent_instance

        class FakeDeepResearcherAgent:
            def __init__(
                self,
                *,
                llm_provider,
                tools,
                max_loops,
                verbose,
                callbacks,
                config=None,
                job_id=None,
            ):
                self.llm_provider = llm_provider
                self.tools = tools
                self.max_loops = max_loops
                self.verbose = verbose
                self.callbacks = callbacks
                self.config = config
                self.job_id = job_id

        fn_config = DeepResearchAgentConfig(
            orchestrator_llm="llm",
            max_loops=7,
            skills=SkillsConfig(enabled=True),
            sandbox=SandboxConfig(app_name="async-aiq"),
        )

        agent = _create_agent_instance(
            agent_cls=FakeDeepResearcherAgent,
            llm_provider="provider",
            llm="llm",
            tools=["tool"],
            fn_config=fn_config,
            verbose=True,
            callbacks=["callback"],
            job_id="job-123",
        )

        assert agent.max_loops == 7
        assert agent.job_id == "job-123"
        assert agent.config is fn_config
        assert agent.config.skills.enabled is True
        assert agent.config.skills.sources == ("/skills/",)
        assert agent.config.sandbox is not None
        assert agent.config.sandbox.app_name == "async-aiq"

    def test_async_deep_researcher_constructor_gets_rendered_skill_instructions(self):
        """Async job construction preserves skills/sandbox config through orchestrator prompt creation."""
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent
        from aiq_agent.agents.deep_researcher.deepagents_runtime import SandboxConfig
        from aiq_agent.agents.deep_researcher.deepagents_runtime import SkillsConfig
        from aiq_agent.agents.deep_researcher.models import DeepResearchAgentState
        from aiq_agent.agents.deep_researcher.register import DeepResearchAgentConfig
        from aiq_agent.common import LLMProvider
        from aiq_agent.common import LLMRole
        from aiq_api.jobs.runner import _create_agent_instance

        @tool
        def async_test_search(query: str) -> str:
            """Search test tool."""
            return f"results for {query}"

        mock_llm = MagicMock()
        provider = LLMProvider()
        provider.set_default(mock_llm)
        provider.configure(LLMRole.ORCHESTRATOR, mock_llm)
        provider.configure(LLMRole.PLANNER, mock_llm)
        provider.configure(LLMRole.RESEARCHER, mock_llm)
        fn_config = DeepResearchAgentConfig(
            orchestrator_llm="llm",
            skills=SkillsConfig(enabled=True),
            sandbox=SandboxConfig(app_name="async-aiq"),
        )
        mock_deep_agent = MagicMock()
        mock_deep_agent.with_config.return_value = mock_deep_agent

        with (
            patch(
                "aiq_agent.agents.deep_researcher.deepagents_runtime._create_sandbox_backend",
                return_value=MagicMock(),
            ),
            patch(
                "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
                return_value=mock_deep_agent,
            ) as create,
        ):
            agent = _create_agent_instance(
                agent_cls=DeepResearcherAgent,
                llm_provider=provider,
                llm=mock_llm,
                tools=[async_test_search],
                fn_config=fn_config,
                verbose=False,
                callbacks=[],
                job_id="async-job-123",
            )
            state = DeepResearchAgentState(
                messages=[
                    HumanMessage(
                        content=(
                            "Compare AI infrastructure capex over the last 8 quarters. Include QoQ and YoY growth."
                        )
                    )
                ]
            )
            agent._build_orchestrator_agent(state)

        kwargs = create.call_args.kwargs
        assert kwargs["skills"] == ["/skills/"]
        assert "Available Skills:" in kwargs["system_prompt"]
        assert "Use read_file to load the relevant SKILL.md BEFORE writing any code" in kwargs["system_prompt"]
        assert 'execute("python /workspace/[name].py")' in kwargs["system_prompt"]
        assert "Tell the planner to account for available skills" in kwargs["system_prompt"]
        assert "Include any applicable skill-use requirements from the plan" in kwargs["system_prompt"]
        assert "data-table-analysis" not in kwargs["system_prompt"]
        assert agent.deepagents_runtime.job_id == "async-job-123"

    def test_create_agent_instance_does_not_drop_config_on_internal_type_error(self):
        """Constructor bugs must not silently fall back to no-config construction."""
        from aiq_agent.agents.deep_researcher.deepagents_runtime import SandboxConfig
        from aiq_agent.agents.deep_researcher.deepagents_runtime import SkillsConfig
        from aiq_agent.agents.deep_researcher.register import DeepResearchAgentConfig
        from aiq_api.jobs.runner import _create_agent_instance

        class BrokenDeepResearcherAgent:
            def __init__(
                self,
                *,
                llm_provider,
                tools,
                max_loops,
                verbose,
                callbacks,
                config=None,
                job_id=None,
            ):
                raise TypeError("internal constructor failure")

        fn_config = DeepResearchAgentConfig(
            orchestrator_llm="llm",
            skills=SkillsConfig(enabled=True),
            sandbox=SandboxConfig(app_name="async-aiq"),
        )

        with pytest.raises(TypeError, match="internal constructor failure"):
            _create_agent_instance(
                agent_cls=BrokenDeepResearcherAgent,
                llm_provider="provider",
                llm="llm",
                tools=["tool"],
                fn_config=fn_config,
                verbose=True,
                callbacks=["callback"],
                job_id="job-123",
            )

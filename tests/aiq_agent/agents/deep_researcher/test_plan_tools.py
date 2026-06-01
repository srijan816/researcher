# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed deep-research plan tool."""

import json

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.deep_researcher.custom_middleware import PlanFileValidationMiddleware
from aiq_agent.agents.deep_researcher.plan_tools import create_write_plan_tool
from aiq_agent.agents.deep_researcher.plan_tools import plan_json_from_tool_args


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Fake chat model that accepts tool binding for DeepAgents tests."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _sample_tool_args() -> dict:
    return {
        "report_title": 'AI Use Cases and "Ready" Criteria',
        "report_toc": [
            {"title": "Executive Summary and Ranking Criteria"},
            {"title": "Ranked Top AI Use Cases"},
        ],
        "queries": [
            {
                "query": "2026 AI use cases ROI cost savings analyst reports",
                "tool": "advanced_web_search_tool",
                "target_sections": ["Executive Summary and Ranking Criteria"],
                "target_claims": [
                    {
                        "claim_id": "C1",
                        "claim_type": "quantitative",
                        "claim": 'Authoritative ROI evidence for AI use cases whose definition of "ready" varies',
                        "required_source_class": "primary_issuer",
                    }
                ],
            }
        ],
        "constraints": [
            {
                "category": "source",
                "constraint": "Use authoritative sources for quantitative claims.",
                "rationale": "The report needs trusted numeric evidence.",
                "verification": "Every numeric claim has a source.",
            }
        ],
    }


def test_plan_json_from_tool_args_creates_valid_validator_payload():
    plan_json = plan_json_from_tool_args(**_sample_tool_args())
    errors = PlanFileValidationMiddleware._validate_plan_payload(plan_json)

    assert errors == []
    plan = json.loads(plan_json)
    assert plan["report_title"] == 'AI Use Cases and "Ready" Criteria'
    assert plan["queries"][0]["target_claim_ids"] == ["C1"]


def test_plan_json_from_tool_args_fills_missing_claims_and_sections():
    plan_json = plan_json_from_tool_args(
        report_title="Compact Plan",
        report_toc=[{"title": "Main Section"}],
        queries=[{"query": "compact query evidence"}],
        constraints=["Stay scoped to the request."],
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["queries"][0]["target_sections"] == ["Main Section"]
    assert plan["queries"][0]["target_claims"][0]["claim_id"] == "C1"
    assert plan["constraints"][0]["constraint"] == "Stay scoped to the request."


def test_plan_json_from_tool_args_normalizes_task_budget_percentages():
    plan_json = plan_json_from_tool_args(
        report_title="Allocated Research Plan",
        report_toc=[{"title": "Main Section"}],
        queries=[
            {"query": "primary source evidence", "relevance_weight": 5, "task_category": "primary_data"},
            {"query": "background context evidence", "relevance_weight": 1, "task_category": "foundations"},
            {"query": "risk counterevidence", "relevance_weight": 4, "task_category": "counterevidence"},
        ],
        constraints=["Allocate budget by relevance and evidence difficulty."],
    )

    plan = json.loads(plan_json)
    budget_percents = [query["budget_percent"] for query in plan["queries"]]

    assert sum(budget_percents) == pytest.approx(100.0)
    assert budget_percents == [50.0, 10.0, 40.0]
    assert [query["task_id"] for query in plan["queries"]] == ["Q1", "Q2", "Q3"]
    assert plan["queries"][0]["task_category"] == "primary_data"


def test_plan_json_from_tool_args_accepts_minimax_item_wrappers():
    plan_json = plan_json_from_tool_args(
        report_title="Wrapped Plan",
        report_toc={
            "item": [
                {
                    "title": "Main Section",
                    "subsections": {"item": [{"title": "Nested Section"}]},
                }
            ]
        },
        queries={
            "item": [
                {
                    "query": "wrapped query evidence",
                    "target_sections": {"item": ["Main Section"]},
                    "target_claims": {
                        "item": [
                            {
                                "claim_id": "C1",
                                "claim_type": "discovery",
                                "claim": "Wrapped target claim should normalize correctly",
                                "required_source_class": "mixed",
                            }
                        ]
                    },
                }
            ]
        },
        constraints={"item": ["Stay scoped to the request."]},
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["report_toc"][0]["subsections"][0]["title"] == "Nested Section"
    assert plan["queries"][0]["target_sections"] == ["Main Section"]
    assert plan["queries"][0]["target_claim_ids"] == ["C1"]
    assert plan["task_analysis"]["claim_profile"]["claims"][0]["claim_id"] == "C1"


def test_plan_json_from_tool_args_accepts_single_item_double_wrappers():
    """Regression for M3 serializing one-item arrays as nested item objects."""

    plan_json = plan_json_from_tool_args(
        report_title="Single Wrapped Plan",
        report_toc={
            "item": {
                "item": {
                    "title": "Main Section",
                    "subsections": {"item": {"item": {"title": "Only Subsection"}}},
                }
            }
        },
        queries={
            "item": {
                "item": {
                    "query": "single wrapped query evidence",
                    "target_sections": {"item": {"item": "Main Section"}},
                    "target_claims": {
                        "item": {
                            "item": {
                                "claim_id": "C1",
                                "claim_type": "discovery",
                                "claim": "Single wrapped target claim should normalize correctly",
                                "required_source_class": "mixed",
                            }
                        }
                    },
                }
            }
        },
        constraints={"item": {"item": "Stay scoped to the request."}},
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["report_toc"][0]["title"] == "Main Section"
    assert plan["report_toc"][0]["subsections"][0]["title"] == "Only Subsection"
    assert plan["queries"][0]["target_sections"] == ["Main Section"]
    assert plan["queries"][0]["target_claim_ids"] == ["C1"]
    assert plan["task_analysis"]["claim_profile"]["claims"] == [
        {
            "claim_id": "C1",
            "claim_text": "Single wrapped target claim should normalize correctly",
            "claim_type": "discovery",
            "expected_answer_shape": "free_text",
            "preferred_source_classes": ["mixed"],
            "target_task_id": "Q1",
        }
    ]


def test_plan_json_from_tool_args_accepts_minimax_text_wrappers():
    plan_json = plan_json_from_tool_args(
        report_title="Middle East Conflict Debates",
        report_toc=[{"title": "Topic Landscape"}],
        queries=[
            {
                "query": "Middle East conflict debates broad foundations authoritative sources",
                "target_claims": {
                    "item": [
                        {"text": "Recurring structural themes across Middle East conflicts"},
                        {"$text": "Institutional architecture including UN and ICJ"},
                    ]
                },
            }
        ],
        constraints={
            "item": [
                {"$text": "Topic-first breadth: do not let the motion sub-area dominate."},
                {"text": "Strict citation: named events and years need inline citations."},
            ]
        },
        fact_ledger_targets={
            "item": [
                {"entity": "Israel", "key_facts": "government and named operations"},
                {"entity": "United Nations", "key_facts": "UNSC resolutions and humanitarian data"},
            ]
        },
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["queries"][0]["target_claims"][0]["claim"] == (
        "Recurring structural themes across Middle East conflicts"
    )
    assert plan["queries"][0]["target_claims"][1]["claim"] == "Institutional architecture including UN and ICJ"
    assert plan["queries"][0]["target_claim_ids"] == ["C1.1", "C1.2"]
    assert plan["constraints"][0]["constraint"].startswith("Topic-first breadth")
    assert plan["fact_ledger_targets"]["entities"][0]["entity"] == "Israel"


def test_write_plan_tool_schema_does_not_expose_runtime_argument():
    schema_fields = create_write_plan_tool().args_schema.model_fields

    assert "runtime" not in schema_fields
    assert {"report_title", "report_toc", "queries", "constraints"}.issubset(schema_fields)


def test_plan_json_from_tool_args_rejects_missing_queries():
    with pytest.raises(ValueError, match="queries"):
        plan_json_from_tool_args(
            report_title="Bad Plan",
            report_toc=[{"title": "Main Section"}],
            queries=[],
            constraints=["Stay scoped."],
        )


@pytest.mark.asyncio
async def test_write_plan_state_is_visible_to_later_read_file_tool():
    """Regression: custom StateBackend writes must land in the same graph state."""

    write_args = _sample_tool_args()
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "write_plan", "args": write_args, "id": "wp1"}]),
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"file_path": "/shared/plan.json"}, "id": "rf1"}],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_deep_agent(model=model, tools=[create_write_plan_tool()], system_prompt="test")

    result = await agent.ainvoke({"messages": [HumanMessage(content="write and read plan")]})

    files = result["files"]
    assert "/shared/plan.json" in files
    assert "/plan.json" in files
    plan = json.loads(files["/shared/plan.json"]["content"])
    assert plan["report_title"] == write_args["report_title"]

    read_tool_messages = [message for message in result["messages"] if getattr(message, "name", None) == "read_file"]
    assert read_tool_messages
    assert "AI Use Cases" in str(read_tool_messages[-1].content)

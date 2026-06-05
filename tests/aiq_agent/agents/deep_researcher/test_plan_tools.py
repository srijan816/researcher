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
    assert plan["task_analysis"]["scope_profile"]["primary_subject"] == 'AI Use Cases and "Ready" Criteria'


def test_plan_json_from_tool_args_preserves_topic_first_scope_profile():
    plan_json = plan_json_from_tool_args(
        report_title="Social Movements Training Research Dossier",
        report_toc=[{"title": "Foundations"}, {"title": "Cases"}],
        queries=[
            {
                "query": "social movements academic foundations and cases",
                "target_sections": ["Foundations", "Cases"],
                "target_claims": [
                    {
                        "claim_id": "C1",
                        "claim_type": "definition",
                        "claim": "Social movements need a topic-first training dossier",
                        "required_source_class": "academic",
                    }
                ],
            }
        ],
        constraints=["Keep debate training as application context."],
        output_style={
            "mode": "lesson_first",
            "topic_anchor": "Social change, social justice, and social movements",
            "target": "Slide-deck-ready training dossier",
            "avoid": ["A report about WSDC/BP itself"],
        },
        task_analysis={
            "scope_profile": {
                "primary_subject": "Social change, social justice, and social movements",
                "deliverable_type": "slide-deck-ready training dossier",
                "application_context": "WSDC/BP competitive debate training",
                "scope_mode": "topic_first",
                "budget_mode": "lesson_first",
            }
        },
    )

    plan = json.loads(plan_json)
    profile = plan["task_analysis"]["scope_profile"]
    assert profile["primary_subject"] == "Social change, social justice, and social movements"
    assert profile["application_context"] == "WSDC/BP competitive debate training"
    assert profile["scope_mode"] == "topic_first"


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


def test_plan_json_from_tool_args_expands_single_query_for_multi_section_reports():
    plan_json = plan_json_from_tool_args(
        report_title="Multi Section Plan",
        report_toc=[
            {"title": "Technical Foundations"},
            {"title": "Market Evidence"},
            {"title": "Implementation Risks"},
        ],
        queries=[
            {
                "query": "Research the complete topic using authoritative evidence",
                "tool": "advanced_web_search_tool",
                "target_sections": ["Technical Foundations", "Market Evidence", "Implementation Risks"],
                "target_claims": [
                    {
                        "claim_id": "C0",
                        "claim_type": "discovery",
                        "claim": "Broad evidence needed for the full report",
                        "required_source_class": "mixed",
                    }
                ],
            }
        ],
        constraints=["Cover each section with source-backed evidence."],
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert len(plan["queries"]) == 3
    assert [query["target_sections"] for query in plan["queries"]] == [
        ["Technical Foundations"],
        ["Market Evidence"],
        ["Implementation Risks"],
    ]
    assert sum(query["budget_percent"] for query in plan["queries"]) == pytest.approx(100.0)
    assert plan["task_analysis"]["claim_profile"]["claims"][0]["target_task_id"] == "Q1"


def test_plan_json_from_tool_args_preserves_single_query_for_focused_screen():
    plan_json = plan_json_from_tool_args(
        report_title="Focused Screen",
        report_toc=[
            {"title": "Candidate Table"},
            {"title": "Evidence Notes"},
            {"title": "Caveats"},
        ],
        queries=[{"query": "single focused screen evidence task"}],
        constraints=["Stay compact."],
        output_style={"mode": "focused_screen"},
    )

    plan = json.loads(plan_json)
    assert len(plan["queries"]) == 1
    assert plan["queries"][0]["target_sections"] == ["Candidate Table", "Evidence Notes", "Caveats"]


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


def test_plan_json_from_tool_args_coerces_numeric_strings_and_mode_aliases():
    plan_json = plan_json_from_tool_args(
        report_title="Numeric Coercion Plan",
        report_toc=[
            {"title": "Foundations"},
            {"title": "Evidence"},
        ],
        queries=[
            {
                "query": "first query evidence",
                "tool": "advanced_web_search_tool",
                "relevance_weight": "5",
                "budget_percent": "55.6%",
                "search_budget": "12 calls",
                "target_sections": {"item": ["Foundations"]},
                "target_claims": {
                    "item": [
                        {
                            "claim_id": "C1",
                            "claim_type": "quantitative",
                            "claim": "Quantitative evidence for the first query",
                            "required_source_class": "primary_issuer",
                        }
                    ]
                },
            },
            {
                "query": "second query evidence",
                "tool": "advanced_web_search_tool",
                "relevance_weight": "1",
                "budget_percent": "44.4%",
                "search_budget": "6",
                "target_sections": {"item": "Evidence"},
                "target_claims": {
                    "item": {
                        "item": {
                            "claim_id": "C2",
                            "claim_type": "discovery",
                            "claim": "Discovery evidence for the second query",
                            "required_source_class": "mixed",
                        }
                    }
                },
            },
        ],
        constraints=["Keep it compact."],
        output_style={"mode": "standard report", "avoid": {"item": "verbose repair loops"}},
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["output_style"]["mode"] == "standard_report"
    assert plan["queries"][0]["relevance_weight"] == 5
    assert plan["queries"][0]["budget_percent"] == pytest.approx(55.6)
    assert plan["queries"][0]["search_budget"] == 12
    assert plan["queries"][1]["relevance_weight"] == 1
    assert plan["queries"][1]["budget_percent"] == pytest.approx(44.4)
    assert plan["queries"][1]["search_budget"] == 6


def test_plan_json_from_tool_args_accepts_prose_source_strategy_and_compacts_payload():
    long_text = " ".join(["detailed planner prose"] * 80)
    plan_json = plan_json_from_tool_args(
        report_title="Planner Robustness Plan",
        report_toc=[{"title": "Main Section"}],
        queries=[
            {
                "query": long_text,
                "rationale": long_text,
                "target_sections": [f"Section {index} {long_text}" for index in range(12)],
                "target_claims": [
                    {
                        "claim_id": f"C{index}",
                        "claim_type": "discovery",
                        "claim": f"Claim {index}: {long_text}",
                        "required_source_class": "mixed",
                    }
                    for index in range(6)
                ],
            }
        ],
        constraints=[long_text for _ in range(20)],
        task_analysis={
            "user_intent": long_text,
            "source_strategy": long_text,
            "claim_profile": long_text,
            "entities": {"central": ["MiniMax M3"], "peripheral": [{"name": "SearXNG"}]},
        },
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["task_analysis"]["source_strategy"]["summary"].endswith("…")
    assert plan["task_analysis"]["claim_profile"]["summary"].endswith("…")
    assert plan["task_analysis"]["entities"] == [
        {"name": "MiniMax M3", "centrality": "central"},
        {"name": "SearXNG", "centrality": "peripheral"},
    ]
    assert len(plan["queries"][0]["target_claims"]) == 3
    assert len(plan["queries"][0]["target_sections"]) == 8
    assert len(plan["constraints"]) == 12
    assert len(plan["queries"][0]["query"]) <= 700
    assert len(plan["queries"][0]["rationale"]) <= 360


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


def test_plan_json_from_tool_args_flattens_nested_claim_lists():
    """Regression for M3 putting a list of claims into one target_claims slot."""

    plan_json = plan_json_from_tool_args(
        report_title="CRO Video Strategy Plan",
        report_toc=[
            {
                "title": "Mute-to-Unmute UX",
                "subsections": {"item": [{"title": "Browser Policy"}, {"title": "Case Studies"}]},
            }
        ],
        queries=[
            {
                "query": "Research mute autoplay UX case studies and browser policy evidence",
                "seed_queries": {
                    "item": {
                        "item": [
                            "Chrome autoplay policy muted video",
                            "iOS Safari autoplay muted playsinline policy",
                        ]
                    }
                },
                "target_sections": {"item": {"item": "Mute-to-Unmute UX"}},
                "target_claims": {
                    "item": {
                        "item": [
                            {
                                "claim_id": "C1",
                                "claim_type": "specification",
                                "claim": "Chrome and Safari autoplay policies require muted or user-initiated media",
                                "required_source_class": "first_party",
                            },
                            {
                                "item": [
                                    {
                                        "claim_id": "C2",
                                        "claim_type": "existence",
                                        "claim": "Brands use kinetic typography to preserve meaning in muted video",
                                        "required_source_class": "authoritative_third_party",
                                    },
                                    {
                                        "item": {
                                            "claim_id": "C3",
                                            "claim_type": "recommendation",
                                            "claim": "Unmute UX should minimize friction and preserve context",
                                            "required_source_class": "primary_issuer",
                                        }
                                    },
                                ]
                            },
                        ]
                    }
                },
            }
        ],
        constraints={"item": {"item": "Stay scoped to mute-first homepage video UX."}},
    )

    plan = json.loads(plan_json)
    assert PlanFileValidationMiddleware._validate_plan_payload(plan_json) == []
    assert plan["queries"][0]["seed_queries"] == [
        "Chrome autoplay policy muted video",
        "iOS Safari autoplay muted playsinline policy",
    ]
    assert plan["queries"][0]["target_sections"] == ["Mute-to-Unmute UX"]
    assert plan["queries"][0]["target_claim_ids"] == ["C1", "C2", "C3"]
    assert [claim["claim"] for claim in plan["queries"][0]["target_claims"]] == [
        "Chrome and Safari autoplay policies require muted or user-initiated media",
        "Brands use kinetic typography to preserve meaning in muted video",
        "Unmute UX should minimize friction and preserve context",
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

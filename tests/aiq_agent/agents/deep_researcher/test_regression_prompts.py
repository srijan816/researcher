# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression set for prompts that previously exposed planning failures."""

import json
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.clarifier.agent import ClarifierAgent
from aiq_agent.agents.deep_researcher.models import DeepResearchAgentState
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "research_regression_prompts.json"


@tool
def regression_web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@pytest.fixture
def regression_cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture
def clarifier_agent() -> ClarifierAgent:
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    provider = MagicMock(spec=LLMProvider)
    provider.get = MagicMock(return_value=llm)
    return ClarifierAgent(llm_provider=provider, user_prompt_callback=AsyncMock())


@pytest.fixture
def deep_agent():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    provider = LLMProvider()
    provider.set_default(llm)
    provider.configure(LLMRole.ORCHESTRATOR, llm)
    provider.configure(LLMRole.PLANNER, llm)
    provider.configure(LLMRole.RESEARCHER, llm)
    with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent"):
        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

        return DeepResearcherAgent(llm_provider=provider, tools=[regression_web_search_tool])


def test_regression_prompt_set_has_expected_user_facing_plans(regression_cases, clarifier_agent):
    for case in regression_cases:
        title = clarifier_agent._fallback_plan_title(case["prompt"])
        sections = clarifier_agent._fallback_plan_sections(case["prompt"])
        combined = " ".join(sections)

        assert title == case["expected_plan_title"], case["id"]
        for term in case["expected_section_terms"]:
            assert term.lower() in combined.lower(), case["id"]


def test_regression_prompt_set_preloads_internal_plans_when_deterministic(deep_agent, regression_cases):
    for case in regression_cases:
        if case["id"] == "ai_life_strategy":
            continue
        state = DeepResearchAgentState(messages=[HumanMessage(content=case["prompt"])])
        updated = deep_agent._inject_approved_plan_if_available(state)

        if "/shared/plan.json" not in updated.files:
            continue

        plan = json.loads("\n".join(updated.files["/shared/plan.json"]["content"]))
        assert plan["report_title"] == case["expected_plan_title"], case["id"]
        assert plan["queries"], case["id"]
        assert sum(query.get("budget_percent", 0) for query in plan["queries"]) == pytest.approx(100.0), case["id"]
        assert all(query.get("search_budget") for query in plan["queries"]), case["id"]
        assert "Landscape" not in " ".join(section["title"] for section in plan["report_toc"]), case["id"]


def test_planner_prompt_uses_typed_write_plan_not_raw_json():
    prompt = Path("src/aiq_agent/agents/deep_researcher/prompts/planner.j2").read_text()

    assert "write_plan" in prompt
    assert "Never call\n`write_file` for `/shared/plan.json`" in prompt
    assert "```json" not in prompt


def test_prompts_require_generic_task_budget_allocation():
    planner_prompt = Path("src/aiq_agent/agents/deep_researcher/prompts/planner.j2").read_text()
    orchestrator_prompt = Path("src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2").read_text()
    researcher_prompt = Path("src/aiq_agent/agents/deep_researcher/prompts/researcher.j2").read_text()

    assert "Research Task Decomposition" in planner_prompt
    assert "budget_percent" in planner_prompt
    assert "sum to exactly 100" in planner_prompt
    assert "Search budget: N search calls for this task" in orchestrator_prompt
    assert "Treat that per-task search budget as a hard cap" in researcher_prompt

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

"""Tests for the DeepResearcherAgent."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from aiq_agent.agents.deep_researcher.custom_middleware import SequentialSearchMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import TaskBatchLimitMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolResultPruningMiddleware
from aiq_agent.agents.deep_researcher.models import DeepResearchAgentState
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common.citation_verification import SourceEntry


def _valid_deep_report(title: str = "Deep research report") -> str:
    body = (
        "This report compares the current evidence, explains the relevant tradeoffs, "
        "and keeps the conclusion grounded in the cited source. "
    ) * 18
    return (
        f"# {title}\n\n"
        "## Executive Summary\n\n"
        f"{body}\n\n"
        "## Analysis\n\n"
        f"{body}\n\n"
        "## Caveats\n\n"
        f"{body}\n\n"
        "## Sources\n\n"
        "[1] https://example.com\n"
    )


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


class TestDeepResearcherAgent:
    """Tests for the DeepResearcherAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = LLMProvider()
        provider.set_default(mock_llm)
        provider.configure(LLMRole.ORCHESTRATOR, mock_llm)
        provider.configure(LLMRole.PLANNER, mock_llm)
        provider.configure(LLMRole.RESEARCHER, mock_llm)
        provider.get = MagicMock(wraps=provider.get)
        return provider

    @pytest.fixture
    def real_tool(self):
        """Create a real LangChain tool."""
        return web_search_tool

    @pytest.fixture
    def mock_create_deep_agent(self):
        """Create a mock for create_deep_agent (deepagents)."""
        mock_agent = MagicMock()
        mock_agent.with_config = MagicMock(return_value=mock_agent)
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content=_valid_deep_report())]})
        return mock_agent

    def test_init_with_defaults(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test DeepResearcherAgent initialization with defaults."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=mock_create_deep_agent,
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            assert agent.llm_provider == mock_llm_provider
            assert len(agent.tools) == 1
            assert agent.max_loops == 2
            assert agent.verbose is True
            assert agent.callbacks == []

    def test_planner_tools_commit_without_external_think_or_delegation(
        self, mock_llm_provider, real_tool, mock_create_deep_agent
    ):
        """Planner uses native model thinking, then commits without external think/delegation loops."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=mock_create_deep_agent,
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            planner_tool_names = {tool.name for tool in agent.planner_tools}
            assert "think" not in planner_tool_names
            assert "write_plan" in planner_tool_names
            assert real_tool.name in planner_tool_names
            assert "defer_task" not in planner_tool_names

    def test_init_with_custom_settings(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test DeepResearcherAgent initialization with custom settings."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            callbacks = [MagicMock()]
            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
                max_loops=5,
                verbose=False,
                callbacks=callbacks,
            )

            assert agent.max_loops == 5
            assert agent.verbose is False
            assert agent.callbacks == callbacks

    def test_init_without_tools(self, mock_llm_provider, mock_create_deep_agent):
        """Test DeepResearcherAgent initialization without tools."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=None,
            )

            assert agent.tools == []

    def test_load_prompts(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test _load_prompts loads all required prompts."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            # Should have planner, researcher, and orchestrator prompts
            assert "planner" in agent._prompts
            assert "researcher" in agent._prompts
            assert "orchestrator" in agent._prompts

    def test_load_prompts_fallback(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test _load_prompts uses inline defaults when files not found."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            with patch(
                "aiq_agent.agents.deep_researcher.agent.load_prompt",
                side_effect=FileNotFoundError(),
            ):
                from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

                agent = DeepResearcherAgent(
                    llm_provider=mock_llm_provider,
                    tools=[real_tool],
                )

                assert "planner" in agent._prompts
                assert "research" in agent._prompts["planner"].lower() or "plan" in agent._prompts["planner"].lower()

    def test_get_inline_default(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test _get_inline_default returns correct defaults."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            planner_default = agent._get_inline_default("planner")
            assert "research" in planner_default.lower() or "plan" in planner_default.lower()

            researcher_default = agent._get_inline_default("researcher")
            assert "research" in researcher_default.lower()

            orchestrator_default = agent._get_inline_default("orchestrator")
            assert "orchestrat" in orchestrator_default.lower() or "research" in orchestrator_default.lower()

            unknown_default = agent._get_inline_default("unknown")
            assert "unknown" in unknown_default.lower()

    def test_extract_approved_plan(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Approved plan context should be parsed into title and sections."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            parsed = agent._extract_approved_plan(
                "**Approved Research Plan**\n\n"
                "Title: Current 40-50% Fair-Value Discount Candidates\n\n"
                "Sections:\n"
                "- Candidate Price/Fair Value Table\n"
                "- Source Quality and Caveats\n"
            )

            assert parsed == (
                "Current 40-50% Fair-Value Discount Candidates",
                ["Candidate Price/Fair Value Table", "Source Quality and Caveats"],
            )

    def test_inject_approved_plan_seeds_current_request_plan_floor(
        self, mock_llm_provider, real_tool, mock_create_deep_agent
    ):
        """Approved previews should seed a fresh executable floor, not stale files."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(
                messages=[
                    HumanMessage(
                        content="Which stocks are currently trading at 40–50% below their estimated fair value"
                    )
                ],
                clarifier_result=(
                    "**Approved Research Plan**\n\n"
                    "Title: Current 40-50% Fair-Value Discount Candidates\n\n"
                    "Sections:\n"
                    "- Candidate Price/Fair Value Table\n"
                    "- Source Quality and Caveats\n"
                ),
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "/plan.json" in updated.files
            assert "/shared/plan.json" in updated.files
            assert updated.clarifier_result == state.clarifier_result
            plan = json.loads("\n".join(updated.files["/shared/plan.json"]["content"]))
            assert plan["report_title"] == "Current 40-50% Fair-Value Discount Candidates"
            assert plan["queries"]

    def test_generic_approved_plan_does_not_preload_plan_file(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Placeholder approved plans should fall back to planner-agent."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(
                messages=[HumanMessage(content="How should teams design AI-augmented workflows in 2026?")],
                clarifier_result=(
                    "**Approved Research Plan**\n\n"
                    "Title: Research Report\n\n"
                    "Sections:\n"
                    "- Introduction\n"
                    "- Background\n"
                    "- Analysis\n"
                    "- Findings\n"
                    "- Conclusion\n"
                ),
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "/plan.json" not in updated.files
            assert "/shared/plan.json" not in updated.files
            assert "generic placeholder" in (updated.clarifier_result or "")

    def test_fallback_like_approved_plan_for_rich_query_passes_specific_context_only(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Rich prompts should replace generic context, while planner-agent still writes the plan."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            query = (
                "Conduct a comprehensive deep research report on the top 10 highest-value use cases of AI in 2026. "
                "For each use case cover ROI, real-world examples, maturity, enabling technologies, barriers, "
                "and who benefits most. Rank the 10 use cases by overall business value."
            )
            state = DeepResearchAgentState(
                messages=[HumanMessage(content=query)],
                clarifier_result=(
                    "**Approved Research Plan**\n\n"
                    "Title: Conduct a comprehensive deep research report on the top 10 highest-value use "
                    "cases of AI i\n\n"
                    "Sections:\n"
                    "- Conduct comprehensive deep research report Landscape\n"
                    "- Recent Evidence and Signals\n"
                    "- Capability Gaps\n"
                    "- Adoption Risks and Recommendations\n"
                ),
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "/plan.json" in updated.files
            assert "/shared/plan.json" in updated.files
            assert "Top 10 Highest-Value AI Use Cases in 2026" in (updated.clarifier_result or "")
            assert "Executive Summary and Ranking Criteria" in (updated.clarifier_result or "")
            assert "generic placeholder" not in (updated.clarifier_result or "")
            plan = json.loads("\n".join(updated.files["/shared/plan.json"]["content"]))
            assert plan["report_title"] == "Top 10 Highest-Value AI Use Cases in 2026"
            assert len(plan["queries"]) >= 2

    def test_structured_lesson_prompt_passes_topic_first_context_only(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Lesson prompts with a broad topic should not preload stale plan files."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(
                messages=[
                    HumanMessage(
                        content=(
                            "Content research dossier filename hint: doctors-patients-content-focused-research.md\n\n"
                            "Generate a structured content research dossier, not a lesson plan and not a "
                            "debate case file.\n"
                            "Report type: content_research_dossier.\n"
                            "Exact lesson topic: Doctors & Patients.\n"
                            "Final debate motion: This house would allow doctors to refuse to perform treatments "
                            "that are against their own ethical principles.\n"
                            "Audience: G5-6 PSD II Grade 5-6 primary debate students.\n\n"
                            "The Markdown must include these sections:\n"
                            "1. Research scope and final motion anchor.\n"
                            "2. Age and audience assumptions.\n"
                            "3. Topic essentials: definitions, key terms, and background concepts.\n"
                            "4. Factual findings with citations: only claims that are well-supported.\n"
                            "5. Key examples and case studies: concrete examples with what each teaches.\n"
                            "6. Motion relevance notes: raw material that helps explain the final motion.\n"
                        )
                    )
                ],
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "/plan.json" in updated.files
            assert "/shared/plan.json" in updated.files
            assert "Doctors & Patients Content Research Dossier" in (updated.clarifier_result or "")
            assert "Topic essentials" in (updated.clarifier_result or "")
            assert "Approved Research Plan" in (updated.clarifier_result or "")
            plan = json.loads("\n".join(updated.files["/shared/plan.json"]["content"]))
            assert plan["task_analysis"]["exact_lesson_topic"] == "Doctors & Patients"
            assert "refuse to perform treatments" in plan["task_analysis"]["final_debate_motion"]

            state_with_existing_plan_context = state.model_copy(
                update={
                    "clarifier_result": (
                        "**Approved Research Plan**\n\n"
                        "Title: Motion-only brief\n\n"
                        "Sections:\n"
                        "- Conscientious objection only\n"
                    )
                }
            )
            updated_with_existing_plan_context = agent._inject_approved_plan_if_available(
                state_with_existing_plan_context
            )

            assert "/plan.json" in updated_with_existing_plan_context.files
            assert "Doctors & Patients Content Research Dossier" in (
                updated_with_existing_plan_context.clarifier_result or ""
            )
            assert "Conscientious objection only" not in (updated_with_existing_plan_context.clarifier_result or "")

    def test_broad_topic_markdown_lesson_prompt_passes_topic_first_context_only(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Oracle lesson prompts use Broad topic / markdown-bold labels, not Exact lesson topic."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(
                messages=[
                    HumanMessage(
                        content=(
                            "**Broad topic**: Education and Tech\n"
                            "**Final debate motion**: This house would mandate teachers to use AI in teaching "
                            "their students\n"
                            "**Student tier**: PSD II (Grade 5-6 debate students)\n\n"
                            "The final report must have EXACTLY these 14 sections, in this order:\n"
                            "1. Topic Landscape (8-12 sub-areas; mark the motion's sub-area with [MOTION])\n"
                            "2. Core Concept\n"
                            "3. How It Works\n"
                            "4. Stakeholders and Power\n"
                            "5. Landmark Cases\n"
                            "6. Bridge to the Motion\n"
                            "7. Motion-Specific Context\n"
                            "8. Arguments FOR the motion\n"
                            "9. Arguments AGAINST the motion\n"
                            "10. Tensions and Limits\n"
                            "11. Practice Motions\n"
                            "12. Hook Question\n"
                            "13. Teacher Notes\n"
                            "14. Sources\n"
                        )
                    )
                ],
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "/plan.json" in updated.files
            assert "/shared/plan.json" in updated.files
            assert "Education and Tech Content Research Dossier" in (updated.clarifier_result or "")
            assert "Topic Landscape" in (updated.clarifier_result or "")
            assert "Approved Research Plan" in (updated.clarifier_result or "")

    def test_training_deck_prompt_keeps_primary_subject_ahead_of_debate_frame(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Slide-deck/debate deliverables should not become WSDC/BP-only research."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(
                messages=[
                    HumanMessage(
                        content=(
                            "You are an elite research specialist. Your mission is to conduct deep research on "
                            "social change, social justice, and social movements to create a world-class WSDC/BP "
                            "competitive debate training session. Produce a complete slide-deck-ready research "
                            "document with slide titles, core bullets, and rich speaker notes."
                        )
                    )
                ],
                research_depth="deeper",
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "/shared/plan.json" in updated.files
            assert "social change, social justice, and social movements Training Research Dossier" in (
                updated.clarifier_result or ""
            )
            plan = json.loads("\n".join(updated.files["/shared/plan.json"]["content"]))
            profile = plan["task_analysis"]["scope_profile"]
            assert profile["primary_subject"] == "social change, social justice, and social movements"
            assert profile["application_context"] == "WSDC/BP competitive debate training"
            assert profile["scope_mode"] == "topic_first"
            assert "WSDC/BP" not in plan["report_title"]
            assert len(plan["report_toc"]) == 6
            assert sum(query["budget_percent"] for query in plan["queries"]) == pytest.approx(100.0)

    def test_lesson_prompt_passes_abbreviation_glossary_to_planner(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Curriculum shorthand like OT should not be expanded into unrelated product terms."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(
                messages=[
                    HumanMessage(
                        content=(
                            "**Broad topic**: Middle East Conflict Debates\n"
                            "**Final debate motion**: latest Israeli attacks for OT\n"
                            "**Student tier**: OT\n\n"
                            "The final report must have EXACTLY these 14 sections, in this order:\n"
                            "1. Topic Landscape\n"
                            "2. Core Concept\n"
                            "3. How It Works\n"
                        )
                    )
                ],
            )

            updated = agent._inject_approved_plan_if_available(state)

            assert "Abbreviation glossary" in (updated.clarifier_result or "")
            assert "OT = Official Teams" in (updated.clarifier_result or "")
            assert "do not invent alternate meanings" in (updated.clarifier_result or "")

    def test_normalize_files_state_adds_missing_metadata(
        self,
        mock_llm_provider,
        real_tool,
        mock_create_deep_agent,
    ):
        """Legacy/preloaded virtual files should not crash glob/grep metadata reads."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            normalized = agent._normalize_files_state({"/shared/plan.json": {"content": ["{}"]}})

            assert normalized["/shared/plan.json"]["content"] == ["{}"]
            assert normalized["/shared/plan.json"]["created_at"]
            assert normalized["/shared/plan.json"]["modified_at"]

    def test_research_depth_controls_tool_limits(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Research-depth tiers should scale deep research tool budgets."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            shallow = DeepResearchAgentState(
                messages=[HumanMessage(content="Compare models")],
                research_depth="shallow",
            )
            medium = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="medium")
            deeper = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deeper")
            deep = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deep")

            assert agent._tool_limits_for_state(shallow)["advanced_web_search_tool"] == 20
            assert agent._tool_limits_for_state(shallow)["planner:advanced_web_search_tool"] == 1
            assert agent._tool_limits_for_state(medium)["advanced_web_search_tool"] == 77
            assert agent._tool_limits_for_state(medium)["planner:advanced_web_search_tool"] == 1
            assert agent._tool_limits_for_state(deeper)["advanced_web_search_tool"] == 92
            assert agent._tool_limits_for_state(deeper)["planner:advanced_web_search_tool"] == 1
            assert agent._tool_limits_for_state(deep)["advanced_web_search_tool"] == 168
            assert agent._tool_limits_for_state(deep)["planner:advanced_web_search_tool"] == 2
            assert agent._parallel_tool_limits_for_state(shallow)["task"] == 1
            assert agent._parallel_tool_limits_for_state(medium)["task"] == 3
            assert agent._parallel_tool_limits_for_state(deeper)["task"] == 4
            assert agent._parallel_tool_limits_for_state(deep)["task"] == 3

    def test_medium_tier_uses_medium_orchestrator_when_configured(self, real_tool):
        """Medium should be able to run synthesis with the thinking-off orchestrator alias."""
        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

        default_llm = MagicMock(name="default_orchestrator")
        medium_llm = MagicMock(name="medium_orchestrator")
        provider = LLMProvider()
        provider.set_default(default_llm)
        provider.configure(LLMRole.ORCHESTRATOR, default_llm)
        provider.configure(LLMRole.MEDIUM_ORCHESTRATOR, medium_llm)

        agent = DeepResearcherAgent(llm_provider=provider, tools=[real_tool])

        medium = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="medium")
        deep = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deep")

        assert agent._orchestrator_llm_for_state(medium) is medium_llm
        assert agent._orchestrator_llm_for_state(deep) is default_llm

    def test_deeper_tier_uses_dedicated_model_roles_when_configured(self, real_tool):
        """Deeper should be able to run on dedicated fast role aliases."""
        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

        default_llm = MagicMock(name="thinking_orchestrator")
        deeper_orchestrator = MagicMock(name="deeper_orchestrator")
        deeper_planner = MagicMock(name="deeper_planner")
        deeper_researcher = MagicMock(name="deeper_researcher")
        provider = LLMProvider()
        provider.set_default(default_llm)
        provider.configure(LLMRole.ORCHESTRATOR, default_llm)
        provider.configure(LLMRole.PLANNER, default_llm)
        provider.configure(LLMRole.RESEARCHER, default_llm)
        provider.configure(LLMRole.DEEPER_ORCHESTRATOR, deeper_orchestrator)
        provider.configure(LLMRole.DEEPER_PLANNER, deeper_planner)
        provider.configure(LLMRole.DEEPER_RESEARCHER, deeper_researcher)

        agent = DeepResearcherAgent(llm_provider=provider, tools=[real_tool])

        deeper = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deeper")
        deep = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deep")

        assert agent._orchestrator_llm_for_state(deeper) is deeper_orchestrator
        assert agent._planner_llm_for_state(deeper) is deeper_planner
        assert agent._researcher_llm_for_state(deeper) is deeper_researcher
        assert agent._orchestrator_llm_for_state(deep) is default_llm
        assert agent._planner_llm_for_state(deep) is default_llm
        assert agent._researcher_llm_for_state(deep) is default_llm

    def test_deep_high_stakes_use_thinking_but_deeper_stays_fast(self, real_tool):
        """Deep may use the thinking orchestrator; deeper should stay on the fast path."""
        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

        thinking_orchestrator = MagicMock(name="thinking_orchestrator")
        fast_orchestrator = MagicMock(name="fast_orchestrator")
        provider = LLMProvider()
        provider.set_default(thinking_orchestrator)
        provider.configure(LLMRole.ORCHESTRATOR, thinking_orchestrator)
        provider.configure(LLMRole.DEEPER_ORCHESTRATOR, fast_orchestrator)
        provider.configure(LLMRole.MEDIUM_ORCHESTRATOR, fast_orchestrator)

        agent = DeepResearcherAgent(llm_provider=provider, tools=[real_tool])

        ordinary_deeper = DeepResearchAgentState(
            messages=[HumanMessage(content="Research best AI coding workflows")],
            research_depth="deeper",
        )
        legal_deeper = DeepResearchAgentState(
            messages=[HumanMessage(content="Research regulatory and legal risks in AI compliance")],
            research_depth="deeper",
        )
        large_doc_deeper = DeepResearchAgentState(
            messages=[HumanMessage(content="Analyze these uploaded documents")],
            research_depth="deeper",
            available_documents=[
                {"file_name": "a.pdf", "summary": "one"},
                {"file_name": "b.pdf", "summary": "two"},
                {"file_name": "c.pdf", "summary": "three"},
            ],
        )
        deep = DeepResearchAgentState(
            messages=[HumanMessage(content="Research AI evals")],
            research_depth="deep",
        )

        assert agent._orchestrator_llm_for_state(ordinary_deeper) is fast_orchestrator
        assert agent._orchestrator_llm_for_state(legal_deeper) is fast_orchestrator
        assert agent._orchestrator_llm_for_state(large_doc_deeper) is fast_orchestrator
        assert agent._orchestrator_llm_for_state(deep) is thinking_orchestrator

    def test_deep_tier_uses_dedicated_thinking_model_roles_when_configured(self, real_tool):
        """Deep should be able to run planner/researcher on dedicated thinking-enabled aliases."""
        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

        default_llm = MagicMock(name="default")
        deep_planner = MagicMock(name="deep_planner")
        deep_researcher = MagicMock(name="deep_researcher")
        provider = LLMProvider()
        provider.set_default(default_llm)
        provider.configure(LLMRole.PLANNER, default_llm)
        provider.configure(LLMRole.RESEARCHER, default_llm)
        provider.configure(LLMRole.DEEP_PLANNER, deep_planner)
        provider.configure(LLMRole.DEEP_RESEARCHER, deep_researcher)

        agent = DeepResearcherAgent(llm_provider=provider, tools=[real_tool])

        deeper = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deeper")
        deep = DeepResearchAgentState(messages=[HumanMessage(content="Compare models")], research_depth="deep")

        assert agent._planner_llm_for_state(deep) is deep_planner
        assert agent._researcher_llm_for_state(deep) is deep_researcher
        assert agent._planner_llm_for_state(deeper) is default_llm
        assert agent._researcher_llm_for_state(deeper) is default_llm

    def test_researcher_context_pruning_is_tighter_than_orchestrator(self, mock_llm_provider, real_tool):
        """Subagents should keep compact context; final synthesis can use more."""
        from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

        agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])

        planner_pruning = agent._tool_result_pruning_middleware_for_scope("planner")
        researcher_pruning = agent._tool_result_pruning_middleware_for_scope("researcher")
        orchestrator_pruning = agent._tool_result_pruning_middleware_for_scope("orchestrator")

        assert isinstance(researcher_pruning, ToolResultPruningMiddleware)
        assert planner_pruning.recent_max_chars < orchestrator_pruning.recent_max_chars
        assert researcher_pruning.recent_max_chars < orchestrator_pruning.recent_max_chars
        assert researcher_pruning.keep_last_n < orchestrator_pruning.keep_last_n
        assert researcher_pruning.max_tool_call_arg_chars < orchestrator_pruning.max_tool_call_arg_chars

    @pytest.mark.asyncio
    async def test_provider_roles_used_on_init(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test LLM roles (planner, researcher, orchestrator) are requested when run() is invoked."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            mock_create_deep_agent.ainvoke.return_value = {
                "messages": [AIMessage(content=_valid_deep_report("Quick query research report"))]
            }
            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(messages=[HumanMessage(content="Quick query")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com"))
            await agent.run(state)

            mock_llm_provider.get.assert_any_call(LLMRole.PLANNER)
            mock_llm_provider.get.assert_any_call(LLMRole.RESEARCHER)
            mock_llm_provider.get.assert_any_call(LLMRole.ORCHESTRATOR)

    @pytest.mark.asyncio
    async def test_run_basic_query(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test run() with a basic query."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            mock_create_deep_agent.ainvoke.return_value = {
                "messages": [AIMessage(content=_valid_deep_report("CUDA vs OpenCL comparison report"))]
            }
            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            state = DeepResearchAgentState(messages=[HumanMessage(content="Compare CUDA vs OpenCL in depth")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com"))

            result = await agent.run(state)

            assert result is not None
            assert result.messages is not None
            assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_run_empty_messages(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test run() with empty messages."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            state = DeepResearchAgentState(messages=[])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com"))

            result = await agent.run(state)

            assert result is not None

    @pytest.mark.asyncio
    async def test_run_with_callbacks(self, mock_llm_provider, real_tool, mock_create_deep_agent):
        """Test run() uses callbacks."""
        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_create_deep_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            mock_create_deep_agent.ainvoke.return_value = {
                "messages": [AIMessage(content=_valid_deep_report("Test query research report"))]
            }
            mock_callback = MagicMock()
            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
                callbacks=[mock_callback],
            )

            state = DeepResearchAgentState(messages=[HumanMessage(content="Test query")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com"))

            await agent.run(state)

            # Callbacks should have been passed to ainvoke
            call_kwargs = mock_create_deep_agent.ainvoke.call_args
            assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_run_handles_error(self, mock_llm_provider, real_tool):
        """Test run() handles errors gracefully."""
        mock_agent = MagicMock()
        mock_agent.with_config = MagicMock(return_value=mock_agent)
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("Agent error"))

        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            state = DeepResearchAgentState(messages=[HumanMessage(content="Test query")])

            with pytest.raises(Exception, match="Agent error"):
                await agent.run(state)

    @pytest.mark.asyncio
    async def test_run_empty_result_messages(self, mock_llm_provider, real_tool):
        """Test run() handles empty result messages."""
        mock_agent = MagicMock()
        mock_agent.with_config = MagicMock(return_value=mock_agent)
        mock_agent.ainvoke = AsyncMock(return_value={"messages": []})

        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            state = DeepResearchAgentState(messages=[HumanMessage(content="Test")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com"))

            with pytest.raises(RuntimeError, match="empty final report"):
                await agent.run(state)

    @pytest.mark.asyncio
    async def test_run_preserves_valid_message_content(self, mock_llm_provider, real_tool):
        """Test run() preserves valid body content while finalizing references."""
        final_report = _valid_deep_report("Original query final analysis")
        result_messages = [
            HumanMessage(content="Original query"),
            AIMessage(content="I'll help with that."),
            ToolMessage(content="Search results here", tool_call_id="123"),
            AIMessage(content=final_report),
        ]

        mock_agent = MagicMock()
        mock_agent.with_config = MagicMock(return_value=mock_agent)
        mock_agent.ainvoke = AsyncMock(return_value={"messages": result_messages})

        with patch("aiq_agent.agents.deep_researcher.agent.create_deep_agent", return_value=mock_agent):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )

            state = DeepResearchAgentState(messages=[HumanMessage(content="Original query")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://example.com"))

            result = await agent.run(state)

            # Non-final messages and report body content should be preserved.
            assert result.messages[0].content == "Original query"
            assert result.messages[1].content == "I'll help with that."
            assert result.messages[2].content == "Search results here"
            final_content = result.messages[3].content
            assert "# Original query final analysis" in final_content
            assert "## Executive Summary" in final_content
            assert "## Analysis" in final_content
            assert "## Caveats" in final_content
            assert "https://example.com" not in final_content


class TestRunRetryStatePreservation:
    """Tests that run() retry on incomplete report preserves full state (files, todos)."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = LLMProvider()
        provider.set_default(mock_llm)
        provider.configure(LLMRole.ORCHESTRATOR, mock_llm)
        provider.configure(LLMRole.PLANNER, mock_llm)
        provider.configure(LLMRole.RESEARCHER, mock_llm)
        return provider

    @pytest.fixture
    def real_tool(self):
        """Create a real LangChain tool."""
        return web_search_tool

    @pytest.mark.asyncio
    async def test_run_incomplete_report_retry_passes_full_state(self, mock_llm_provider, real_tool):
        """Second ainvoke on retry must receive full state (files, todos), not only messages."""
        incomplete_content = "Short report.\n## Section One\nText."
        complete_content = "A" * 1600 + "\n## Intro\n\n## Methods\n\n## Results\n\n## Sources\n[1] http://example.com"

        first_result = {
            "messages": [
                HumanMessage(content="Compare X and Y"),
                AIMessage(content=incomplete_content),
            ],
            "files": {"research_notes.txt": "Findings from search..."},
            "todos": [{"id": "1", "status": "completed", "title": "Planning"}],
        }
        second_result = {
            "messages": [
                HumanMessage(content="Compare X and Y"),
                AIMessage(content=incomplete_content),
                HumanMessage(content="Your report is not yet complete..."),
                AIMessage(content=complete_content),
            ],
            "files": first_result["files"],
            "todos": first_result["todos"],
        }

        # Return incomplete then complete; repeat complete so any extra ainvoke calls succeed
        mock_agent = MagicMock()
        mock_agent.with_config = MagicMock(return_value=mock_agent)
        mock_agent.ainvoke = AsyncMock(side_effect=[first_result, second_result] + [second_result] * 10)

        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=mock_agent,
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(messages=[HumanMessage(content="Compare X and Y")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="http://example.com"))

            await agent.run(state)

            # Find the retry call: state has "files" and last message is feedback
            call_list = mock_agent.ainvoke.call_args_list
            retry_calls = [
                c[0][0]
                for c in call_list
                if isinstance(c[0][0], dict)
                and c[0][0].get("files") == {"research_notes.txt": "Findings from search..."}
                and c[0][0].get("todos")
                and c[0][0]["messages"]
                and "not yet complete" in str(c[0][0]["messages"][-1].content)
            ]
            assert retry_calls, "At least one retry must pass full state (files, todos) and feedback message"
            second_call_state = retry_calls[0]
            assert second_call_state["files"] == {"research_notes.txt": "Findings from search..."}
            assert second_call_state["todos"] == [{"id": "1", "status": "completed", "title": "Planning"}]
            assert len(second_call_state["messages"]) == 3
            assert "not yet complete" in str(second_call_state["messages"][-1].content)

    @pytest.mark.asyncio
    async def test_run_incomplete_report_retry_appends_feedback_message(self, mock_llm_provider, real_tool):
        """Retry must append a HumanMessage with feedback; previous messages preserved."""
        short_content = "Brief."
        full_content = "X" * 1600 + "\n## A\n\n## B\n\n## Sources\n[1] https://a.com"

        first_result = {"messages": [HumanMessage(content="Q"), AIMessage(content=short_content)]}
        second_result = {
            "messages": [
                first_result["messages"][0],
                first_result["messages"][1],
                HumanMessage(content="Your report is not yet complete. Reason: too_short..."),
                AIMessage(content=full_content),
            ],
        }

        mock_agent = MagicMock()
        mock_agent.with_config = MagicMock(return_value=mock_agent)
        mock_agent.ainvoke = AsyncMock(side_effect=[first_result, second_result] + [second_result] * 10)

        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=mock_agent,
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(
                llm_provider=mock_llm_provider,
                tools=[real_tool],
            )
            state = DeepResearchAgentState(messages=[HumanMessage(content="Q")])
            agent.source_registry_middleware.registry.add(SourceEntry(url="https://a.com"))

            await agent.run(state)

            # Find the retry call: last message is feedback about too_short
            call_list = mock_agent.ainvoke.call_args_list
            retry_calls = [
                c[0][0]
                for c in call_list
                if isinstance(c[0][0], dict)
                and c[0][0].get("messages")
                and len(c[0][0]["messages"]) == 3
                and "too_short" in str(c[0][0]["messages"][-1].content)
            ]
            assert retry_calls, "Retry must append feedback message to messages"
            second_call_state = retry_calls[0]
            assert second_call_state["messages"][0].content == "Q"
            assert second_call_state["messages"][1].content == short_content
            assert "too_short" in str(second_call_state["messages"][2].content)
            assert "Expand" in str(second_call_state["messages"][2].content)


class TestIsReportComplete:
    """Tests for _is_report_complete heuristic."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        provider = LLMProvider()
        provider.set_default(mock_llm)
        provider.configure(LLMRole.ORCHESTRATOR, mock_llm)
        provider.configure(LLMRole.PLANNER, mock_llm)
        provider.configure(LLMRole.RESEARCHER, mock_llm)
        return provider

    @pytest.fixture
    def real_tool(self):
        return web_search_tool

    def test_complete_report_returns_true(self, mock_llm_provider, real_tool):
        """Report with length, headers, and Sources section is complete."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            content = "A" * 1600 + "\n## Introduction\n\n## Methods\n\n## Sources\n[1] http://x.com"
            result = {"messages": [AIMessage(content=content)]}
            is_complete, reason = agent._is_report_complete(result)
            assert is_complete is True
            assert "complete" in reason.lower()

    def test_too_short_returns_false(self, mock_llm_provider, real_tool):
        """Report under length threshold is incomplete."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            result = {"messages": [AIMessage(content="Short.")]}
            is_complete, reason = agent._is_report_complete(result)
            assert is_complete is False
            assert "too_short" in reason

    def test_missing_sources_returns_false(self, mock_llm_provider, real_tool):
        """Report without Sources section is incomplete."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            content = "A" * 1600 + "\n## Intro\n\n## Body\n\nNo sources here."
            result = {"messages": [AIMessage(content=content)]}
            is_complete, reason = agent._is_report_complete(result)
            assert is_complete is False
            assert "missing_sources" in reason or "sources" in reason.lower()

    def test_empty_messages_returns_false(self, mock_llm_provider, real_tool):
        """Empty messages is incomplete."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            result = {"messages": []}
            is_complete, reason = agent._is_report_complete(result)
            assert is_complete is False
            assert "no_messages" in reason or "message" in reason.lower()

    def test_write_file_tool_call_extracts_content(self, mock_llm_provider, real_tool):
        """Report written via write_file tool call should be detected as complete."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            report_content = "A" * 1600 + "\n## Introduction\n\n## Methods\n\n## Sources\n[1] http://x.com"
            # AIMessage with empty text but report in write_file tool call
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "write_file", "args": {"file_path": "/report.md", "content": report_content}, "id": "tc1"}
                ],
            )
            result = {"messages": [msg]}
            is_complete, reason = agent._is_report_complete(result)
            assert is_complete is True
            assert "complete" in reason.lower()

    def test_research_notes_fallback_builds_report(self, mock_llm_provider, real_tool):
        """Research notes should be usable when the final synthesis message is too short."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            notes = (
                "**Query Topic**\n\n"
                "**Research Notes**\n"
                "| Ticker | Price | Fair Value | Discount |\n"
                "| --- | --- | --- | --- |\n"
                "| ABC | $10 | $18 | 44% |\n\n"
                "Detailed evidence. " + "A" * 800 + "\n\n**Sources**\n[1] Example: https://example.com"
            )
            result = {
                "messages": [AIMessage(content="Model call failed after retries.")],
                "files": {
                    "/shared/stock_screen.txt": {"content": notes.splitlines()},
                    "/shared/plan.json": {"content": ["{}"]},
                },
            }

            report = agent._build_report_from_research_notes(result)

            assert report.startswith("# Research Findings")
            assert "stock_screen.txt" in report
            assert "ABC" in report
            assert "Model call failed" not in report

    def test_model_failure_text_is_not_complete_report(self, mock_llm_provider, real_tool):
        """Provider connection errors must not be accepted as successful reports."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            result = {
                "messages": [
                    AIMessage(content="Model call failed after 2 attempts with APIConnectionError: Connection error.")
                ]
            }

            is_complete, reason = agent._is_report_complete(result)

            assert is_complete is False
            assert reason == "model_call_failed"

    def test_provider_thinking_payload_is_not_complete_report(self, mock_llm_provider, real_tool):
        """MiniMax reasoning-only provider blocks must trigger repair instead of success."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            thinking_payload = (
                "[{'thinking': 'The user wants a comprehensive report and I should write one now.', "
                "'type': 'thinking', 'index': 0}]"
            )
            result = {"messages": [AIMessage(content=thinking_payload)]}

            is_complete, reason = agent._is_report_complete(result)
            report = agent._extract_report_content_from_result(result)

            assert is_complete is False
            assert "too_short" in reason
            assert report == ""

    def test_report_file_wins_over_model_failure_message(self, mock_llm_provider, real_tool):
        """If report.md exists, a later terminal model error should not hide it."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            report_content = "A" * 1600 + "\n## Findings\n\n## Sources\n[1] Example: https://example.com"
            result = {
                "messages": [
                    AIMessage(content="Model call failed after 2 attempts with APIConnectionError: Connection error.")
                ],
                "files": {"/report.md": {"content": report_content.splitlines()}},
            }

            report = agent._extract_report_content_from_result(result)

            assert report == report_content

    def test_claim_fragments_merge_into_canonical_claim_table(self, mock_llm_provider, real_tool):
        """Per-researcher claim fragments should be merged deterministically."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool], job_id="job-claims")
            result = {
                "messages": [AIMessage(content="")],
                "files": {
                    "/shared/claims/claims_market.json": {
                        "content": json.dumps(
                            {
                                "atomic_claims": [
                                    {
                                        "claim_id": "C1",
                                        "claim_text": "The market has a verified demand signal.",
                                        "claim_type": "trend",
                                        "expected_answer_shape": "free_text",
                                        "preferred_source_classes": ["authoritative_third_party"],
                                        "status": "verified",
                                        "resolved_value": "Verified demand signal",
                                        "evidence": [
                                            {
                                                "source_url": "https://example.com/report",
                                                "source_class": "authoritative_third_party",
                                                "extract": "The market has a verified demand signal.",
                                                "extract_confidence": "high",
                                            }
                                        ],
                                    }
                                ]
                            }
                        ).splitlines()
                    }
                },
            }

            merged = agent._merge_claim_fragments_into_result(result, job_id="job-claims")

            assert merged is True
            assert "/shared/claim_table.json" in result["files"]
            assert "shared/claim_table.json" in result["files"]
            assert agent._claim_table_quality_reason(result) is None

    def test_claim_table_quality_counts_atomic_claims(self, mock_llm_provider, real_tool):
        """Atomic claim tables should not be treated as empty just because entries is empty."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            result = {
                "files": {
                    "/shared/claim_table.json": {
                        "content": json.dumps(
                            {
                                "atomic_claims": [
                                    {
                                        "claim_id": "C1",
                                        "claim_text": "The report has a verified evidence-backed claim.",
                                        "claim_type": "trend",
                                        "expected_answer_shape": "free_text",
                                        "preferred_source_classes": ["any_credible"],
                                        "status": "verified",
                                        "resolved_value": "Evidence-backed claim",
                                        "evidence": [
                                            {
                                                "source_url": "https://example.com/source",
                                                "source_class": "authoritative_third_party",
                                                "extract": "The report has a verified evidence-backed claim.",
                                                "extract_confidence": "high",
                                            }
                                        ],
                                    }
                                ]
                            }
                        ).splitlines()
                    }
                }
            }

            assert agent._claim_table_quality_reason(result) is None

    def test_structured_artifact_merge_builds_evidence_packet(self, mock_llm_provider, real_tool):
        """Merged claim/extract artifacts should produce the M3 evidence packet."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool], job_id="job-evidence")
            result = {
                "messages": [AIMessage(content="")],
                "files": {
                    "/shared/claims/claims_market.json": {
                        "content": json.dumps(
                            {
                                "atomic_claims": [
                                    {
                                        "claim_id": "C1",
                                        "claim_text": "The source supports this claim.",
                                        "claim_type": "trend",
                                        "expected_answer_shape": "free_text",
                                        "preferred_source_classes": ["authoritative_third_party"],
                                        "status": "verified",
                                        "resolved_value": "Supported claim",
                                        "evidence": [
                                            {
                                                "source_url": "https://example.com/report",
                                                "source_class": "authoritative_third_party",
                                                "extract": "The source supports this claim.",
                                                "extract_confidence": "high",
                                            }
                                        ],
                                    }
                                ]
                            }
                        ).splitlines()
                    },
                    "/shared/extracts/market.json": {
                        "content": json.dumps(
                            {
                                "extracts": [
                                    {
                                        "claim_ids": ["C1"],
                                        "url": "https://example.com/report",
                                        "title": "Example Report",
                                        "source_class": "authoritative_third_party",
                                        "extract": "Another source-close extract.",
                                    }
                                ]
                            }
                        ).splitlines()
                    },
                },
            }

            agent._merge_structured_research_artifacts_into_result(result)

            assert "/shared/claim_table.json" in result["files"]
            assert "/shared/evidence_packet.json" in result["files"]
            packet = json.loads("\n".join(result["files"]["/shared/evidence_packet.json"]["content"]))
            assert packet["job_id"] == "job-evidence"
            assert packet["source_count"] == 1
            assert packet["sources"][0]["claim_ids"] == ["C1"]

    def test_structured_artifact_merge_backfills_section_brief_from_notes(self, mock_llm_provider, real_tool):
        """Final synthesis should get a clean section brief even when researchers only wrote notes."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool], job_id="job-brief")
            result = {
                "messages": [AIMessage(content="")],
                "files": {
                    "/shared/topic_notes.txt": {
                        "content": (
                            "This researcher note contains sourced findings about the topic. "
                            "It is long enough to be useful as synthesis input and should be "
                            "compiled into an intermediate section brief for the final writer. "
                            "The note includes several concrete findings, caveats, and URLs like "
                            "https://example.com/report that the source registry can later verify."
                        ).splitlines()
                    }
                },
            }

            agent._merge_structured_research_artifacts_into_result(result)

            assert "/shared/section_briefs/compiled_from_notes.md" in result["files"]
            brief = "\n".join(result["files"]["/shared/section_briefs/compiled_from_notes.md"]["content"])
            assert "Section Briefs Compiled From Research Notes" in brief
            assert "topic notes.txt" in brief

    def test_source_quality_warning_does_not_make_report_incomplete(self, mock_llm_provider, real_tool):
        """M3 path should deliver real reports and let post-run gates carry soft warnings."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            agent.source_registry_middleware._get_registry().add(
                SourceEntry(url="https://example.com/source-1", source_class="content_marketing")
            )
            content = _valid_deep_report("Weak but usable report").replace(
                "[1] https://example.com",
                "[1] https://example.com/source-1",
            )
            result = {
                "messages": [AIMessage(content=content)],
                "files": {"/report.md": {"content": content.splitlines()}},
            }

            assert agent._is_report_complete(result) == (True, "complete_via_heuristic")

    def test_report_with_many_available_sources_must_use_tier_source_floor(self, mock_llm_provider, real_tool):
        """A rich deeper run should not pass synthesis with only a handful of referenced sources."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            registry = agent.source_registry_middleware._get_registry()
            for index in range(1, 16):
                registry.add(SourceEntry(url=f"https://source{index}.example.com/report"))

            body = (
                "This report has enough length and structure, but it cites too few of the "
                "verified sources collected during the research run. "
            ) * 18
            refs = "\n".join(f"[{index}] https://source{index}.example.com/report" for index in range(1, 8))
            result = {
                "messages": [
                    AIMessage(
                        content=(
                            f"# Report\n\n## Findings\n\n{body}\n\n## Implications\n\n{body}\n\n## Sources\n{refs}"
                        )
                    )
                ]
            }
            state = DeepResearchAgentState(messages=[HumanMessage(content="Q")], research_depth="deeper")

            is_complete, reason = agent._is_report_complete(result, state)

            assert is_complete is False
            assert "too_few_sources_used" in reason

    def test_report_source_floor_waits_until_enough_sources_are_available(self, mock_llm_provider, real_tool):
        """A genuinely thin run should not fail the source floor merely because the tier is deeper."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            agent.source_registry_middleware._get_registry().add(SourceEntry(url="https://example.com/source-1"))
            content = _valid_deep_report("Thin but complete report").replace(
                "[1] https://example.com",
                "[1] https://example.com/source-1",
            )
            result = {"messages": [AIMessage(content=content)]}
            state = DeepResearchAgentState(messages=[HumanMessage(content="Q")], research_depth="deeper")

            assert agent._is_report_complete(result, state) == (True, "complete_via_heuristic")

    def test_fact_ledger_fragments_merge_into_canonical_ledger(self, mock_llm_provider, real_tool):
        """Per-researcher fact-ledger fragments should be merged deterministically."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            result = {
                "files": {
                    "/shared/fact_ledger_grok.json": {
                        "content": json.dumps(
                            {
                                "entries": [
                                    {
                                        "entity": "Grok Imagine",
                                        "fact": "Official docs describe prompt behavior.",
                                        "source_url": "https://docs.x.ai/grok",
                                        "source_extract": "Official docs describe prompt behavior.",
                                        "source_class": "first_party",
                                        "confidence": "high",
                                        "status": "verified",
                                    }
                                ]
                            }
                        ).splitlines()
                    }
                }
            }

            merged = agent._merge_fact_ledger_fragments_into_result(result)

            assert merged is True
            assert "/shared/fact_ledger.json" in result["files"]
            content = "\n".join(result["files"]["/shared/fact_ledger.json"]["content"])
            assert "Grok Imagine" in content
            assert "https://docs.x.ai/grok" in content

    @pytest.mark.asyncio
    async def test_artifact_compiler_builds_report_when_finalizer_is_thinking_only(
        self, mock_llm_provider, mock_llm, real_tool
    ):
        """Persisted artifacts should produce a usable report when MiniMax emits only thinking."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            mock_llm.ainvoke.side_effect = RuntimeError("compiler model unavailable")
            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            evidence = (
                "# Consolidated Findings\n\n"
                "The sabbatical planning market has premium coaching analogs and clear customer pain. "
                "A strong offer should combine planning software, financial runway tools, and expert templates. "
                "Evidence source: https://example.com/sabbatical-planning\n\n"
                "## Market\n\n" + "Detailed evidence. " * 140
            )
            result = {
                "messages": [
                    AIMessage(
                        content=(
                            "[{'thinking': 'Now I need to write the final report.', 'type': 'thinking', 'index': 0}]"
                        )
                    )
                ],
                "files": {
                    "/shared/consolidated_findings.md": {"content": evidence.splitlines()},
                    "/shared/plan.json": {"content": ["{}"]},
                },
            }
            state = DeepResearchAgentState(messages=[HumanMessage(content="Research takeabreak.life monetization")])

            report = await agent._compile_report_from_artifacts(state, result)

            assert report.startswith("#")
            assert "## Sources" in report
            assert "https://example.com/sabbatical-planning" in report
            assert "Recovered Research Report" not in report
            assert agent.source_registry_middleware.registry.has_url("https://example.com/sabbatical-planning")

    @pytest.mark.asyncio
    async def test_artifact_compiler_reads_route_stripped_shared_resume_files(
        self, mock_llm_provider, mock_llm, real_tool
    ):
        """Resume files preloaded for the /shared route are stored without a /shared prefix."""
        with patch(
            "aiq_agent.agents.deep_researcher.agent.create_deep_agent",
            return_value=MagicMock(),
        ):
            from aiq_agent.agents.deep_researcher.agent import DeepResearcherAgent

            mock_llm.ainvoke.side_effect = RuntimeError("compiler model unavailable")
            agent = DeepResearcherAgent(llm_provider=mock_llm_provider, tools=[real_tool])
            evidence = (
                "# Foreign Policy Research Notes\n\n"
                "The research covers sovereignty, non-intervention, targeted killing, and human-rights abuses. "
                "Evidence source: https://example.com/foreign-policy\n\n"
                "## Findings\n\n" + "Detailed evidence about sovereignty and targeted killing. " * 140
            )
            state = DeepResearchAgentState(
                messages=[HumanMessage(content="Foreign Policy & Sovereignty lesson research")],
                files={
                    "/researcher_task1.md": {"content": evidence.splitlines()},
                    "/plan.json": {"content": ["{}"]},
                },
            )
            result = {
                "messages": [AIMessage(content=[{"type": "thinking", "thinking": "writing now"}])],
                "files": {},
            }

            report = await agent._compile_report_from_artifacts(state, result)

            assert "foreign-policy" in report
            assert "Foreign Policy Research Notes" in report
            assert "## researcher task1.md" in report
            assert "plan.json" not in report


class TestSequentialSearchMiddleware:
    """Tests for search-call trimming middleware."""

    @pytest.mark.asyncio
    async def test_allows_two_parallel_search_calls(self):
        middleware = SequentialSearchMiddleware({"advanced_web_search_tool", "web_search_tool"})
        request = MagicMock()
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "searching"},
                {"type": "tool_use", "id": "1", "name": "advanced_web_search_tool", "input": {"question": "a"}},
                {"type": "tool_use", "id": "2", "name": "advanced_web_search_tool", "input": {"question": "b"}},
                {"type": "tool_use", "id": "3", "name": "think", "input": {"thought": "x"}},
            ],
            tool_calls=[
                {"name": "advanced_web_search_tool", "args": {"question": "a"}, "id": "1"},
                {"name": "advanced_web_search_tool", "args": {"question": "b"}, "id": "2"},
                {"name": "think", "args": {"thought": "x"}, "id": "3"},
            ],
        )

        async def handler(_request):
            return MagicMock(result=[message], structured_response=None)

        response = await middleware.awrap_model_call(request, handler)

        assert len(response.result[0].tool_calls) == 3
        assert response.result[0].tool_calls[0]["id"] == "1"
        assert response.result[0].tool_calls[1]["id"] == "2"
        assert response.result[0].tool_calls[2]["name"] == "think"
        content_blocks = response.result[0].content
        assert isinstance(content_blocks, list)
        assert [block.get("id") for block in content_blocks if block.get("type") == "tool_use"] == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_trims_beyond_two_parallel_search_calls(self):
        middleware = SequentialSearchMiddleware({"advanced_web_search_tool", "web_search_tool"})
        request = MagicMock()
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "searching"},
                {"type": "tool_use", "id": "1", "name": "advanced_web_search_tool", "input": {"question": "a"}},
                {"type": "tool_use", "id": "2", "name": "web_search_tool", "input": {"query": "b"}},
                {"type": "tool_use", "id": "3", "name": "advanced_web_search_tool", "input": {"question": "c"}},
                {"type": "tool_use", "id": "4", "name": "think", "input": {"thought": "x"}},
            ],
            tool_calls=[
                {"name": "advanced_web_search_tool", "args": {"question": "a"}, "id": "1"},
                {"name": "web_search_tool", "args": {"query": "b"}, "id": "2"},
                {"name": "advanced_web_search_tool", "args": {"question": "c"}, "id": "3"},
                {"name": "think", "args": {"thought": "x"}, "id": "4"},
            ],
        )

        async def handler(_request):
            return MagicMock(result=[message], structured_response=None)

        response = await middleware.awrap_model_call(request, handler)

        assert [tc["id"] for tc in response.result[0].tool_calls] == ["1", "2", "4"]
        content_blocks = response.result[0].content
        assert isinstance(content_blocks, list)
        assert [block.get("id") for block in content_blocks if block.get("type") == "tool_use"] == ["1", "2", "4"]


class TestTaskBatchLimitMiddleware:
    """Tests for researcher task burst protection."""

    @pytest.mark.asyncio
    async def test_defers_extra_parallel_task_calls(self):
        middleware = TaskBatchLimitMiddleware(default_limit=1)
        request = MagicMock()
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "launching researchers"},
                {"type": "tool_use", "id": "1", "name": "task", "input": {"description": "section a"}},
                {"type": "tool_use", "id": "2", "name": "task", "input": {"description": "section b"}},
            ],
            tool_calls=[
                {"name": "task", "args": {"description": "section a"}, "id": "1"},
                {"name": "task", "args": {"description": "section b"}, "id": "2"},
            ],
        )

        async def handler(_request):
            return MagicMock(result=[message], structured_response=None)

        response = await middleware.awrap_model_call(request, handler)

        assert response.result[0].tool_calls[0]["name"] == "task"
        assert response.result[0].tool_calls[1]["name"] == "defer_task"
        assert response.result[0].tool_calls[1]["args"]["description"] == "section b"
        content_blocks = response.result[0].content
        assert isinstance(content_blocks, list)
        tool_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]
        assert tool_blocks[1]["name"] == "defer_task"
        assert tool_blocks[1]["input"]["description"] == "section b"

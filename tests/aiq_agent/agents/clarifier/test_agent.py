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

"""Tests for the ClarifierAgent."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.clarifier.agent import DEFAULT_CLARIFICATION_PROMPT
from aiq_agent.agents.clarifier.agent import ClarifierAgent
from aiq_agent.agents.clarifier.models import ClarificationResponse
from aiq_agent.agents.clarifier.models import ClarifierAgentState
from aiq_agent.agents.clarifier.models import ClarifierResult
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole


@tool
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


class TestClarifierAgentInit:
    """Tests for ClarifierAgent initialization."""

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
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.fixture
    def mock_user_callback(self):
        """Create a mock user prompt callback."""
        return AsyncMock(return_value="User response")

    def test_init_with_defaults(self, mock_llm_provider, mock_user_callback):
        """Test initialization with default values."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        assert agent.llm_provider == mock_llm_provider
        assert agent.tools == []
        assert agent.user_prompt_callback == mock_user_callback
        assert agent.max_turns == 3
        assert agent.log_response_max_chars == 2000
        assert agent.verbose is False
        assert agent.callbacks == []
        assert agent.system_prompt is not None

    def test_init_with_tools(self, mock_llm_provider, mock_user_callback):
        """Test initialization with tools."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            tools=[web_search_tool],
            user_prompt_callback=mock_user_callback,
        )

        assert len(agent.tools) == 1

    def test_init_with_custom_max_turns(self, mock_llm_provider, mock_user_callback):
        """Test initialization with custom max_turns."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            max_turns=5,
        )

        assert agent.max_turns == 5

    def test_init_with_callbacks(self, mock_llm_provider, mock_user_callback):
        """Test initialization with callbacks."""
        mock_callback = MagicMock()
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            callbacks=[mock_callback],
        )

        assert agent.callbacks == [mock_callback]

    def test_init_with_verbose(self, mock_llm_provider, mock_user_callback):
        """Test initialization with verbose mode."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            verbose=True,
        )

        assert agent.verbose is True

    def test_graph_property(self, mock_llm_provider, mock_user_callback):
        """Test graph property returns compiled graph."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        assert agent.graph is not None
        assert agent.graph == agent._graph

    def test_get_llm(self, mock_llm_provider, mock_llm, mock_user_callback):
        """Test _get_llm returns LLM from provider."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        result = agent._get_llm()

        mock_llm_provider.get.assert_called_with(LLMRole.CLARIFIER)
        assert result == mock_llm


class TestClarifierAgentPromptLoading:
    """Tests for prompt loading functionality."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)
        return provider

    @pytest.fixture
    def mock_user_callback(self):
        """Create a mock user prompt callback."""
        return AsyncMock(return_value="Response")

    def test_load_prompt_fallback(self, mock_llm_provider, mock_user_callback):
        """Test fallback to default prompt when file not found."""
        with patch(
            "aiq_agent.agents.clarifier.agent.load_prompt",
            side_effect=FileNotFoundError(),
        ):
            agent = ClarifierAgent(
                llm_provider=mock_llm_provider,
                user_prompt_callback=mock_user_callback,
            )
            assert agent.system_prompt == DEFAULT_CLARIFICATION_PROMPT

    def test_load_prompt_success(self, mock_llm_provider, mock_user_callback):
        """Test successful prompt loading."""
        custom_prompt = "Custom clarification prompt"
        with patch(
            "aiq_agent.agents.clarifier.agent.load_prompt",
            return_value=custom_prompt,
        ):
            agent = ClarifierAgent(
                llm_provider=mock_llm_provider,
                user_prompt_callback=mock_user_callback,
            )
            assert agent.system_prompt == custom_prompt


class TestClarifierAgentParsing:
    """Tests for JSON response parsing."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing parsing methods."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_parse_response_valid_json(self, agent):
        """Test parsing valid JSON response."""
        text = '{"needs_clarification": true, "clarification_question": "What scope?"}'
        result = agent._parse_response(text)

        assert result is not None
        assert result.needs_clarification is True
        assert result.clarification_question == "What scope?"

    def test_parse_response_with_code_block(self, agent):
        """Test parsing JSON wrapped in code block."""
        text = '```json\n{"needs_clarification": false, "clarification_question": null}\n```'
        result = agent._parse_response(text)

        assert result is not None
        assert result.needs_clarification is False

    def test_parse_plan_response_extracts_wrapped_json(self, agent):
        """Plan parsing should tolerate surrounding prose and code fences."""
        text = (
            "Here is the plan:\n"
            "```json\n"
            '{"title":"Focused Research Plan","sections":["Scope and Criteria","Current Evidence Base"]}'
            "\n```"
        )
        title, sections = agent._parse_plan_response(text)

        assert title == "Focused Research Plan"
        assert sections == ["Scope and Criteria", "Current Evidence Base"]

    def test_parse_response_invalid_json(self, agent):
        """Test parsing invalid JSON returns None."""
        result = agent._parse_response("not valid json")
        assert result is None

    def test_parse_response_empty_string(self, agent):
        """Test parsing empty string returns None."""
        result = agent._parse_response("")
        assert result is None

    def test_parse_response_none(self, agent):
        """Test parsing None returns None."""
        result = agent._parse_response(None)
        assert result is None

    def test_is_needed_true(self, agent):
        """Test _is_needed returns True when needed."""
        text = '{"needs_clarification": true, "clarification_question": "What?"}'
        assert agent._is_needed(text) is True

    def test_is_needed_false(self, agent):
        """Test _is_needed returns False when not needed."""
        text = '{"needs_clarification": false, "clarification_question": null}'
        assert agent._is_needed(text) is False

    def test_is_needed_invalid_json(self, agent):
        """Test _is_needed returns True for invalid JSON (safe default)."""
        assert agent._is_needed("invalid") is True

    def test_is_complete_true(self, agent):
        """Test _is_complete returns True when complete."""
        text = '{"needs_clarification": false, "clarification_question": null}'
        assert agent._is_complete(text) is True

    def test_is_complete_false(self, agent):
        """Test _is_complete returns False when not complete."""
        text = '{"needs_clarification": true, "clarification_question": "What?"}'
        assert agent._is_complete(text) is False

    def test_is_complete_invalid_json(self, agent):
        """Test _is_complete returns False for invalid JSON."""
        assert agent._is_complete("invalid") is False

    def test_valid_needed_true(self, agent):
        """Test _valid_needed returns True for valid response."""
        text = '{"needs_clarification": true, "clarification_question": "What scope?"}'
        assert agent._valid_needed(text) is True

    def test_valid_needed_no_question_mark(self, agent):
        """Test _valid_needed returns True even without question mark."""
        text = '{"needs_clarification": true, "clarification_question": "Tell me more"}'
        assert agent._valid_needed(text) is True

    def test_valid_needed_invalid_json(self, agent):
        """Test _valid_needed returns False for invalid JSON."""
        assert agent._valid_needed("invalid") is False

    def test_get_clarification_question(self, agent):
        """Test extracting clarification question."""
        text = '{"needs_clarification": true, "clarification_question": "What aspect?"}'
        result = agent._get_clarification_question(text)
        assert result == "What aspect?"

    def test_get_clarification_question_fallback(self, agent):
        """Test fallback question for invalid response."""
        result = agent._get_clarification_question("invalid")
        assert "provide more details" in result.lower()


class TestClarifierAgentSkipCommands:
    """Tests for skip command detection."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    @pytest.mark.parametrize("command", ["skip", "done", "exit", "quit", "proceed", "continue", "no", "n", ""])
    def test_is_skip_command_recognized(self, agent, command):
        """Test all skip commands are recognized."""
        assert agent._is_skip_command(command) is True

    @pytest.mark.parametrize("command", ["SKIP", "Done", "EXIT", "  skip  ", "QUIT"])
    def test_is_skip_command_case_insensitive(self, agent, command):
        """Test skip commands are case insensitive."""
        assert agent._is_skip_command(command) is True

    def test_is_skip_command_not_recognized(self, agent):
        """Test non-skip responses are not recognized."""
        assert agent._is_skip_command("option 1") is False
        assert agent._is_skip_command("technical deep dive") is False

    def test_is_skip_command_whitespace_handling(self, agent):
        """Test whitespace is stripped."""
        assert agent._is_skip_command("  skip  ") is True
        # "\n\n" strips to "", which is a skip command (empty string)
        assert agent._is_skip_command("\n\n") is True
        assert agent._is_skip_command("some text") is False


class TestClarifierAgentFallback:
    """Tests for fallback clarification."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_get_fallback_clarification(self, agent):
        """Test fallback clarification returns valid JSON."""
        result = agent._get_fallback_clarification()

        # Should be valid JSON
        data = json.loads(result)
        assert data["needs_clarification"] is True
        assert "?" in data["clarification_question"]

    def test_fallback_is_valid_response(self, agent):
        """Test fallback response passes validation."""
        result = agent._get_fallback_clarification()
        response = ClarificationResponse.model_validate_json(result)

        assert response.needs_clarification is True
        assert response.is_valid() is True


class TestClarifierAgentRun:
    """Tests for the run method."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_run_immediate_completion(self, mock_llm_provider, mock_llm):
        """Test run when LLM immediately returns complete."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert isinstance(result, ClarifierResult)

    @pytest.mark.asyncio
    async def test_run_with_skip_command(self, mock_llm_provider, mock_llm):
        """Test run when user skips clarification."""
        clarification_response = ClarificationResponse(needs_clarification=True, clarification_question="What scope?")
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=clarification_response.model_dump_json()))

        mock_user_callback = AsyncMock(return_value="skip")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        mock_user_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_max_turns_reached(self, mock_llm_provider, mock_llm):
        """Test run when max turns is 0."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
            max_turns=0,
        )

        state = ClarifierAgentState(
            messages=[HumanMessage(content="Research AI")],
            max_turns=0,
        )
        result = await agent.run(state)

        assert result is not None

    @pytest.mark.asyncio
    async def test_run_asks_financial_screen_clarification_once(self, mock_llm_provider, mock_llm):
        """Missing stock universe/source should trigger one deterministic clarification."""
        mock_llm.ainvoke = AsyncMock()
        mock_user_callback = AsyncMock(return_value="1")
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
        )

        state = ClarifierAgentState(
            messages=[
                HumanMessage(content="Which stocks are currently trading at 40–50% below their estimated fair value")
            ]
        )
        result = await agent.run(state)

        assert result is not None
        assert "Universe and fair-value basis" in result.clarifier_log
        mock_user_callback.assert_called_once()
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_logs_query(self, mock_llm_provider, mock_llm, caplog):
        """Test that run logs the query."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Test query")])

        with caplog.at_level("INFO"):
            await agent.run(state)

        assert "Clarifier: Starting" in caplog.text


class TestClarifierAgentPlanParsing:
    """Tests for plan response parsing."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing parsing methods."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_parse_plan_response_valid_json(self, agent):
        """Test parsing valid plan JSON response."""
        text = '{"title": "AI Research Report", "sections": ["Introduction", "Methods", "Results"]}'
        title, sections = agent._parse_plan_response(text)

        assert title == "AI Research Report"
        assert sections == ["Introduction", "Methods", "Results"]

    def test_parse_plan_response_with_code_block(self, agent):
        """Test parsing plan JSON wrapped in code block."""
        text = '```json\n{"title": "Research Plan", "sections": ["Overview", "Analysis"]}\n```'
        title, sections = agent._parse_plan_response(text)

        assert title == "Research Plan"
        assert sections == ["Overview", "Analysis"]

    def test_parse_plan_response_embedded_json(self, agent):
        """Plan parser should recover JSON if the model adds surrounding text."""
        text = 'Here is the plan:\n{"title": "AI Agent Reliability", "sections": ["Failure Modes", "Guardrails"]}'
        title, sections = agent._parse_plan_response(text)

        assert title == "AI Agent Reliability"
        assert sections == ["Failure Modes", "Guardrails"]

    def test_parse_plan_response_dict_sections(self, agent):
        """Plan parser should handle common structured section shapes."""
        text = (
            '{"plan_title": "MiniMax Search Quality", '
            '"outline": [{"heading": "Search Coverage"}, {"heading": "Citation Quality"}]}'
        )
        title, sections = agent._parse_plan_response(text)

        assert title == "MiniMax Search Quality"
        assert sections == ["Search Coverage", "Citation Quality"]

    def test_parse_plan_response_markdown_title_and_sections(self, agent):
        """Plan parser should recover when a model emits Markdown instead of JSON."""
        text = """
        Title: Living Well in the Age of AI

        Sections:
        1. AI-Era Human Priorities
        2. Work and Learning Strategy
        3. Attention, Relationships, and Wellbeing
        4. Risk Boundaries and Agency
        """

        title, sections = agent._parse_plan_response(text)

        assert title == "Living Well in the Age of AI"
        assert sections == [
            "AI-Era Human Priorities",
            "Work and Learning Strategy",
            "Attention, Relationships, and Wellbeing",
            "Risk Boundaries and Agency",
        ]

    def test_parse_plan_response_invalid_json(self, agent):
        """Test parsing invalid JSON returns None and empty list."""
        title, sections = agent._parse_plan_response("not valid json")
        assert title is None
        assert sections == []

    def test_parse_plan_response_empty_string(self, agent):
        """Test parsing empty string returns None and empty list."""
        title, sections = agent._parse_plan_response("")
        assert title is None
        assert sections == []

    def test_parse_plan_response_none(self, agent):
        """Test parsing None returns None and empty list."""
        title, sections = agent._parse_plan_response(None)
        assert title is None
        assert sections == []

    def test_parse_plan_response_missing_sections(self, agent):
        """Test parsing response with missing sections."""
        text = '{"title": "Research Plan"}'
        title, sections = agent._parse_plan_response(text)

        assert title is None
        assert sections == []

    def test_parse_plan_response_invalid_sections_type(self, agent):
        """Test parsing response with non-list sections."""
        text = '{"title": "Research Plan", "sections": "not a list"}'
        title, sections = agent._parse_plan_response(text)

        assert title is None
        assert sections == []

    def test_parse_plan_response_non_string_sections(self, agent):
        """Test parsing response with non-string section items."""
        text = '{"title": "Research Plan", "sections": [1, 2, 3]}'
        title, sections = agent._parse_plan_response(text)

        assert title is None
        assert sections == []


class TestClarifierAgentApprovalParsing:
    """Tests for approval response parsing."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing parsing methods."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    @pytest.mark.parametrize(
        "response", ["approve", "approved", "yes", "ok", "proceed", "continue", "go ahead", "looks good", "y"]
    )
    def test_parse_approval_approved(self, agent, response):
        """Test all approval keywords are recognized."""
        approved, rejected, feedback = agent._parse_approval(response)
        assert approved is True
        assert rejected is False
        assert feedback is None

    @pytest.mark.parametrize("response", ["reject", "rejected", "no", "cancel", "stop", "abort", "n"])
    def test_parse_approval_rejected(self, agent, response):
        """Test all rejection keywords are recognized."""
        approved, rejected, feedback = agent._parse_approval(response)
        assert approved is False
        assert rejected is True
        assert feedback is None

    def test_parse_approval_feedback(self, agent):
        """Test feedback response is captured."""
        approved, rejected, feedback = agent._parse_approval("Please add a section about security")
        assert approved is False
        assert rejected is False
        assert feedback == "Please add a section about security"

    def test_parse_approval_case_insensitive(self, agent):
        """Test approval parsing is case insensitive."""
        approved, rejected, feedback = agent._parse_approval("APPROVE")
        assert approved is True

        approved, rejected, feedback = agent._parse_approval("REJECT")
        assert rejected is True

    def test_parse_approval_with_whitespace(self, agent):
        """Test approval parsing handles whitespace."""
        approved, rejected, feedback = agent._parse_approval("  approve  ")
        assert approved is True

    def test_parse_approval_json_wrapped(self, agent):
        """Test approval parsing extracts query from JSON."""
        approved, rejected, feedback = agent._parse_approval('{"query": "approve", "context": "test"}')
        assert approved is True

    def test_parse_approval_json_wrapped_feedback(self, agent):
        """Test feedback extraction from JSON-wrapped response."""
        approved, rejected, feedback = agent._parse_approval('{"query": "add more sections"}')
        assert approved is False
        assert rejected is False
        assert feedback == "add more sections"


class TestClarifierAgentPlanFormatting:
    """Tests for plan formatting."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing formatting methods."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_format_plan_for_user(self, agent):
        """Test plan formatting for user display."""
        title = "AI Research Report"
        sections = ["Introduction", "Background", "Analysis"]

        result = agent._format_plan_for_user(title, sections)

        assert "**Research Plan Preview**" in result
        assert "**Title:** AI Research Report" in result
        assert "1. Introduction" in result
        assert "2. Background" in result
        assert "3. Analysis" in result
        assert "approve" in result.lower()
        assert "reject" in result.lower()

    def test_format_plan_for_user_empty_sections(self, agent):
        """Test plan formatting with empty sections list."""
        result = agent._format_plan_for_user("Test Plan", [])

        assert "**Title:** Test Plan" in result
        assert "**Sections:**" in result


class TestClarifierAgentPlanScopeGuards:
    """Tests for deterministic plan scope guardrails."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing scope helpers."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)

        return ClarifierAgent(
            llm_provider=provider,
            user_prompt_callback=AsyncMock(),
        )

    def test_precise_stock_screen_gets_compact_plan(self, agent):
        """Specific valuation screens should not keep a broad educational TOC."""
        query = "Which stocks are currently trading at 40–50% below their estimated fair value"
        broad_sections = [
            "Introduction and Background on Value Investing",
            "Methodologies for Estimating Stock Fair Value",
            "Current Market Conditions and Undervalued Opportunities",
            "Screening Criteria for 40-50% Discount Stocks",
            "Sector Analysis and Industry Trends",
            "Risk Factors and Considerations",
            "Conclusion and Investment Recommendations",
        ]

        title, sections = agent._compact_plan_for_query("Identifying Stocks", broad_sections, query)

        assert title == "Stocks Trading 40-50% Below Fair Value"
        assert sections == [
            "Screening Assumptions",
            "Candidate Price/Fair Value Table",
            "Evidence and Caveats",
        ]

    def test_precise_stock_screen_missing_assumptions_needs_clarification(self, agent):
        """Ambiguous fair-value screens should ask for universe/source before planning."""
        ambiguous = "Which stocks are currently trading at 40–50% below their estimated fair value"
        specified = "Which U.S. large-cap stocks are trading 40-50% below Morningstar fair value?"

        assert agent._needs_financial_screen_clarification(ambiguous) is True
        assert agent._needs_financial_screen_clarification(specified) is False

    def test_focus_feedback_selects_numbered_section(self, agent):
        """Plan feedback like 'focus on number 3' should be parsed deterministically."""
        sections = ["Intro", "Methods", "Current Opportunities", "Risks"]

        assert agent._extract_section_focus_index("just focus on number 3", sections) == 2
        assert agent._normalize_plan_feedback("just focus on number 3", sections) == (
            "Focus only on section 3: Current Opportunities. "
            "Remove unrelated sections and revise the plan around that narrow focus."
        )

    def test_ai_life_question_gets_specific_fallback_plan(self, agent):
        """Broad AI-life strategy prompts should not show generic fallback headings."""
        query = "What’s the best way to live life in the age of AI"

        assert agent._fallback_plan_title(query) == "Living Well in the Age of AI"
        assert agent._fallback_plan_sections(query) == [
            "AI-Era Human Priorities",
            "Work and Learning Strategy",
            "Attention, Relationships, and Wellbeing",
            "Risk Boundaries and Agency",
            "Practical Life Operating System",
        ]

    def test_placeholder_sections_are_replaced_before_display(self, agent):
        """Copied schema/example sections should be repaired before plan approval."""
        title, sections = agent._sanitize_plan_for_query(
            "What’s the best way to live life in the age of AI",
            [
                "Scope and Criteria",
                "Current Evidence Base",
                "Key Findings and Trade-offs",
                "Recommended Next Steps",
            ],
            "What’s the best way to live life in the age of AI",
        )

        assert title == "What’s the best way to live life in the age of AI"
        assert sections == [
            "AI-Era Human Priorities",
            "Work and Learning Strategy",
            "Attention, Relationships, and Wellbeing",
            "Risk Boundaries and Agency",
            "Practical Life Operating System",
        ]

    def test_ai_curiosity_engine_build_prompt_gets_rich_user_facing_plan(self, agent):
        """Product-concept build prompts should not fall back to generic engineering buckets."""
        query = (
            "Deep reserach on all the ingredietns required to build:"
            "” Lifelong curiosity engines that surface and teach niche topics matched to user interests” using AI"
        )

        assert agent._fallback_plan_title(query) == "Building AI Curiosity Engines"
        assert agent._fallback_plan_sections(query) == [
            "Learning Science and Curiosity Foundations",
            "Dynamic Interest Modeling Over Time",
            "Long-Tail Discovery and Serendipity Architecture",
            "Adaptive Teaching, Scaffolding, and Dialogue",
            "RAG, Knowledge Graphs, and Agentic Workflows",
            "Niche Content Verification and Hallucination Controls",
            "Engagement, Retention, and Learning Outcome Metrics",
        ]

    def test_weak_ai_build_preview_is_repaired_before_display(self, agent):
        """A typo-heavy echoed title and generic build sections should be replaced."""
        query = (
            "Deep reserach on all the ingredietns required to build:"
            "” Lifelong curiosity engines that surface and teach niche topics matched to user interests” using AI"
        )

        title, sections = agent._sanitize_plan_for_query(
            "Deep reserach on all the ingredietns required to build:” Lifelong curiosity engines that s",
            [
                "Deep reserach on all ingredietns Requirements",
                "Architecture and Interfaces",
                "Failure Paths and Guardrails",
                "Implementation Plan",
            ],
            query,
        )

        assert title == "Building AI Curiosity Engines"
        assert sections == [
            "Learning Science and Curiosity Foundations",
            "Dynamic Interest Modeling Over Time",
            "Long-Tail Discovery and Serendipity Architecture",
            "Adaptive Teaching, Scaffolding, and Dialogue",
            "RAG, Knowledge Graphs, and Agentic Workflows",
            "Niche Content Verification and Hallucination Controls",
            "Engagement, Retention, and Learning Outcome Metrics",
        ]

    def test_weak_ai_build_preview_is_quality_issue_before_safety_net(self, agent):
        """Weak build plans should be sent back to the model before deterministic repair."""
        query = (
            "Deep reserach on all the ingredietns required to build:"
            "” Lifelong curiosity engines that surface and teach niche topics matched to user interests” using AI"
        )

        issue = agent._plan_quality_issue(
            "Deep reserach on all the ingredietns required to build:” Lifelong curiosity engines that s",
            [
                "Deep reserach on all ingredietns Requirements",
                "Architecture and Interfaces",
                "Failure Paths and Guardrails",
                "Implementation Plan",
            ],
            query,
        )

        assert issue is not None
        assert "generic engineering buckets" in issue

    def test_ranked_ai_use_case_report_compiles_explicit_plan(self, agent):
        """Explicit report dimensions should become the approval plan contract."""
        query = (
            "Conduct a comprehensive deep research report on the top 10 highest-value use cases of AI in 2026. "
            "For each use case, cover the following dimensions: "
            "1. What it is – A clear, jargon-free explanation of the use case and how AI is applied "
            "2. Why it’s high-value – Quantified business impact (ROI, cost savings, revenue uplift, efficiency gains) "
            "3. Real-world examples – Specific companies or industries deploying this use case successfully "
            "in 2025–2026 "
            "4. Maturity level – Is this emerging, scaling, or mainstream in 2026? "
            "5. Key enabling technologies – Which AI models, platforms, or techniques power it "
            "6. Barriers to adoption – Top obstacles organisations face "
            "7. Who benefits most – Which company sizes, roles, or industries get the most value "
            "Rank the 10 use cases by overall business value and transformative potential in 2026. "
            "Present findings in a structured report with an executive summary, ranked list with detailed sections, "
            "and a final insight on where AI value creation is heading in 2027–2028."
        )

        assert agent._fallback_plan_title(query) == "Top 10 Highest-Value AI Use Cases in 2026"
        assert agent._fallback_plan_sections(query) == [
            "Executive Summary and Ranking Criteria",
            "Ranked Top 10 AI Use Cases",
            "Business Impact, ROI, and Efficiency Evidence",
            "2025-2026 Real-World Deployment Examples",
            "Maturity Levels and Enabling Technologies",
            "Adoption Barriers and Best-Fit Beneficiaries",
            "2027-2028 AI Value Creation Outlook",
        ]

    def test_generic_latest_fallback_plan_is_replaced_for_explicit_ranked_report(self, agent):
        """The old Landscape/Signals/Risks plan should never reach approval for explicit reports."""
        query = (
            "Conduct a comprehensive deep research report on the top 10 highest-value use cases of AI in 2026. "
            "For each use case cover ROI, real-world examples, maturity, enabling technologies, barriers, "
            "and who benefits most. Rank the 10 use cases by overall business value."
        )

        issue = agent._plan_quality_issue(
            "Conduct a comprehensive deep research report on the top 10 highest-value use cases of AI i",
            [
                "Conduct comprehensive deep research report Landscape",
                "Recent Evidence and Signals",
                "Capability Gaps",
                "Adoption Risks and Recommendations",
            ],
            query,
        )
        assert issue is not None
        assert "explicit report" in issue or "fallback-like" in issue

        title, sections = agent._sanitize_plan_for_query(
            "Conduct a comprehensive deep research report on the top 10 highest-value use cases of AI i",
            [
                "Conduct comprehensive deep research report Landscape",
                "Recent Evidence and Signals",
                "Capability Gaps",
                "Adoption Risks and Recommendations",
            ],
            query,
        )

        assert title == "Top 10 Highest-Value AI Use Cases in 2026"
        assert sections[0] == "Executive Summary and Ranking Criteria"
        assert "Ranked Top 10 AI Use Cases" in sections

    def test_debate_training_prompt_gets_query_contract_before_llm(self, agent):
        """Lesson/debate prompts should not be reduced to the format label or generic buckets."""
        query = (
            "Create a complete slide-deck-ready research document on social change, social justice, "
            "and social movements to create a world-class WSDC/BP competitive debate training session. "
            "Cover definitions and philosophy, major social movement theories, success/failure metrics, "
            "tactics, digital media, intersectionality, comparative case studies, backlash, and debate "
            "applications with sample motions and speaker notes."
        )

        title, sections, source = agent._plan_contract_from_query(query)

        assert source == "training"
        assert title == "Social Change, Social Justice, and Social Movements Training Research Dossier"
        assert "Definitions, Philosophical Foundations, and Core Terms" in sections
        assert "Comparative Case Studies and Debate Applications" in sections

    def test_generic_training_plan_violates_query_contract(self, agent):
        """Generic plans for rich training requests should be repaired or replaced before approval."""
        query = (
            "Create a complete slide-deck-ready research document on social change, social justice, "
            "and social movements to create a world-class WSDC/BP competitive debate training session. "
            "Cover definitions and philosophy, major social movement theories, success/failure metrics, "
            "tactics, digital media, intersectionality, comparative case studies, backlash, and debate "
            "applications with sample motions and speaker notes."
        )
        contract_title, contract_sections, contract_source = agent._plan_contract_from_query(query)

        issue = agent._plan_quality_issue(
            "WSDC/BP Debate Training Plan",
            [
                "Training Landscape",
                "Recent Evidence and Signals",
                "Capability Gaps",
                "Adoption Risks and Recommendations",
            ],
            query,
            contract_title=contract_title,
            contract_sections=contract_sections,
            contract_source=contract_source,
        )

        assert issue is not None
        assert "plan contract" in issue or "fallback-like" in issue


class TestClarifierAgentPlanApproval:
    """Tests for plan approval workflow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    @pytest.fixture
    def mock_planner_llm(self):
        """Create a mock planner LLM."""
        llm = MagicMock()
        return llm

    @pytest.fixture
    def mock_llm_provider(self, mock_llm):
        """Create a mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=mock_llm)
        return provider

    @pytest.mark.asyncio
    async def test_run_with_plan_approval_approved(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Test run with plan approval when user approves."""
        # First, LLM completes clarification
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        # Planner LLM returns a valid plan
        plan_response = json.dumps(
            {
                "title": "Test Research Plan",
                "sections": ["AI Capabilities", "Human Priorities", "Practical Choices"],
            }
        )
        mock_planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content=plan_response))

        # User approves
        mock_user_callback = AsyncMock(return_value="approve")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert result.plan_approved is True
        assert result.plan_rejected is False
        assert result.plan_title == "Test Research Plan"
        assert result.plan_sections == ["AI Capabilities", "Human Priorities", "Practical Choices"]

    @pytest.mark.asyncio
    async def test_weak_plan_is_model_repaired_before_approval(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Weak approval previews should be repaired by the planner model before display."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        weak_plan = json.dumps(
            {
                "title": "Deep reserach on all the ingredietns required to build: Lifelong curiosity engines that s",
                "sections": [
                    "Deep reserach on all ingredietns Requirements",
                    "Architecture and Interfaces",
                    "Failure Paths and Guardrails",
                    "Implementation Plan",
                ],
            }
        )
        repaired_plan = json.dumps(
            {
                "title": "AI Curiosity Engine Build Blueprint",
                "sections": [
                    "Cognitive Science of Curiosity",
                    "Evolving Interest Graphs",
                    "Serendipitous Long-Tail Discovery",
                    "Adaptive Teaching Loops",
                    "RAG and Knowledge Graph Stack",
                    "Content Verification and Evaluation",
                ],
            }
        )
        mock_planner_llm.ainvoke = AsyncMock(
            side_effect=[AIMessage(content=weak_plan), AIMessage(content=repaired_plan)]
        )
        mock_user_callback = AsyncMock(return_value="approve")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(
            messages=[
                HumanMessage(
                    content=(
                        "Deep reserach on all the ingredietns required to build:"
                        "” Lifelong curiosity engines that surface and teach niche topics matched to user interests” "
                        "using AI"
                    )
                )
            ]
        )
        result = await agent.run(state)

        assert result.plan_approved is True
        assert result.plan_title == "AI Curiosity Engine Build Blueprint"
        assert result.plan_sections == [
            "Cognitive Science of Curiosity",
            "Evolving Interest Graphs",
            "Serendipitous Long-Tail Discovery",
            "Adaptive Teaching Loops",
            "RAG and Knowledge Graph Stack",
            "Content Verification and Evaluation",
        ]
        assert mock_planner_llm.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_training_plan_uses_contract_when_model_and_repair_are_generic(
        self, mock_llm_provider, mock_llm, mock_planner_llm
    ):
        """The approval UI should show the query contract if the planner keeps underfitting."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        weak_plan = json.dumps(
            {
                "title": "WSDC/BP Debate Training Plan",
                "sections": [
                    "Training Landscape",
                    "Recent Evidence and Signals",
                    "Capability Gaps",
                    "Adoption Risks and Recommendations",
                ],
            }
        )
        invalid_repair = "not json"
        call_index = 0

        async def planner_side_effect(messages):
            nonlocal call_index
            rendered_prompt = messages[0].content
            assert "DERIVED PLAN CONTRACT:" in rendered_prompt
            assert "Comparative Case Studies and Debate Applications" in rendered_prompt
            content = weak_plan if call_index == 0 else invalid_repair
            call_index += 1
            return AIMessage(content=content)

        mock_planner_llm.ainvoke = AsyncMock(side_effect=planner_side_effect)
        mock_user_callback = AsyncMock(return_value="approve")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        query = (
            "Create a complete slide-deck-ready research document on social change, social justice, "
            "and social movements to create a world-class WSDC/BP competitive debate training session. "
            "Cover definitions and philosophy, major social movement theories, success/failure metrics, "
            "tactics, digital media, intersectionality, comparative case studies, backlash, and debate "
            "applications with sample motions and speaker notes."
        )
        result = await agent.run(ClarifierAgentState(messages=[HumanMessage(content=query)]))

        assert result.plan_approved is True
        assert result.plan_title == "Social Change, Social Justice, and Social Movements Training Research Dossier"
        assert result.plan_sections == [
            "Research Scope, Audience, and Training Objectives",
            "Definitions, Philosophical Foundations, and Core Terms",
            "Major Theories, Mechanisms, and Causal Models",
            "Success Metrics, Evidence Standards, and Critiques",
            "Tactics, Organization, Digital Media, and Repression",
            "Intersectionality, Backlash, and Global Perspectives",
            "Comparative Case Studies and Debate Applications",
        ]
        assert mock_planner_llm.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_plan_preview_preserves_original_query_after_numeric_clarification_reply(
        self, mock_llm_provider, mock_llm, mock_planner_llm
    ):
        """Numeric clarification answers must not become the plan title or scope."""
        original_query = "Research the highest-impact AI applications trending in 2026"
        clarify_response = ClarificationResponse(
            needs_clarification=True,
            clarification_question=(
                "Context: AI applications vary widely. Choose 1 business, 4 developer, or 5 overview."
            ),
        )
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content=clarify_response.model_dump_json()),
                AIMessage(content=complete_response.model_dump_json()),
            ]
        )

        async def planner_side_effect(messages):
            rendered_prompt = messages[0].content
            assert f"ORIGINAL USER REQUEST:\n{original_query}" in rendered_prompt
            assert "ORIGINAL USER REQUEST:\n1 and 4" not in rendered_prompt
            return AIMessage(
                content=json.dumps(
                    {
                        "title": "2026 AI Applications for Business and Developers",
                        "sections": [
                            "Business Automation Opportunities",
                            "Developer and API Workflows",
                            "Cross-Cutting 2026 Trends",
                            "Adoption Risks and Trade-offs",
                            "Practical Implementation Roadmap",
                        ],
                    }
                )
            )

        mock_planner_llm.ainvoke = AsyncMock(side_effect=planner_side_effect)
        mock_user_callback = AsyncMock(side_effect=["1 and 4", "approve"])

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content=original_query)])
        result = await agent.run(state)

        assert result.plan_approved is True
        assert result.plan_title == "2026 AI Applications for Business and Developers"
        assert result.plan_title != "1 and 4"

    @pytest.mark.asyncio
    async def test_run_with_plan_approval_rejected(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Test run with plan approval when user rejects."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        plan_response = '{"title": "Test Plan", "sections": ["Section 1", "Section 2"]}'
        mock_planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content=plan_response))

        mock_user_callback = AsyncMock(return_value="reject")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert result.plan_approved is False
        assert result.plan_rejected is True

    @pytest.mark.asyncio
    async def test_run_with_plan_approval_feedback_then_approve(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Test run with plan approval when user provides feedback then approves."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        # First plan, then revised plan
        plan_response_1 = '{"title": "Initial Plan", "sections": ["Intro", "Analysis"]}'
        plan_response_2 = '{"title": "Revised Plan", "sections": ["Intro", "Security", "Analysis"]}'
        mock_planner_llm.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(content=plan_response_1),
                AIMessage(content=plan_response_2),
            ]
        )

        # User provides feedback, then approves
        mock_user_callback = AsyncMock(side_effect=["add a security section", "approve"])

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert result.plan_approved is True
        assert result.plan_title == "Revised Plan"
        assert "Security" in result.plan_sections

    @pytest.mark.asyncio
    async def test_plan_approval_focus_number_feedback_then_approve(
        self, mock_llm_provider, mock_llm, mock_planner_llm
    ):
        """Terse section-focus feedback should not regenerate the same broad plan."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        broad_plan = json.dumps(
            {
                "title": "Identifying Stocks Trading 40-50% Below Estimated Fair Value",
                "sections": [
                    "Introduction and Background on Value Investing",
                    "Methodologies for Estimating Stock Fair Value",
                    "Current Market Conditions and Undervalued Opportunities",
                    "Screening Criteria for 40-50% Discount Stocks",
                    "Sector Analysis and Industry Trends",
                    "Risk Factors and Considerations",
                    "Conclusion and Investment Recommendations",
                ],
            }
        )
        mock_planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content=broad_plan))
        mock_user_callback = AsyncMock(side_effect=["skip", "just focus on number 3", "approve"])

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(
            messages=[
                HumanMessage(content="Which stocks are currently trading at 40–50% below their estimated fair value")
            ]
        )
        result = await agent.run(state)

        assert result.plan_approved is True
        assert result.plan_title == "Current 40-50% Fair-Value Discount Candidates"
        assert result.plan_sections == [
            "Candidate Price/Fair Value Table",
            "Source Quality and Caveats",
        ]
        assert mock_user_callback.await_count == 3

    @pytest.mark.asyncio
    async def test_run_with_plan_approval_max_iterations(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Test plan approval auto-approves after max iterations."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        plan_response = '{"title": "Test Plan", "sections": ["Section 1"]}'
        mock_planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content=plan_response))

        # User keeps providing feedback
        mock_user_callback = AsyncMock(return_value="make it better")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
            max_plan_iterations=2,  # Low iteration limit
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert result.plan_approved is True  # Auto-approved after max iterations

    @pytest.mark.asyncio
    async def test_run_with_plan_approval_fallback_plan(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Test plan approval uses fallback when LLM returns invalid plan."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        # LLM returns invalid plan
        mock_planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content="not valid json"))

        mock_user_callback = AsyncMock(return_value="approve")

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=mock_user_callback,
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        assert result.plan_approved is True
        # Honest behavior: when only the generic safety-net fallback is left,
        # no fabricated plan is presented or bound to the run.
        assert result.plan_title is None
        assert result.plan_sections == []
        assert result.get_approved_plan_context() is None
        # The user saw an honest "preview unavailable" message, not a fake plan.
        displayed = mock_user_callback.call_args[0][0]
        assert "Plan preview unavailable" in displayed

    @pytest.mark.asyncio
    async def test_run_with_plan_approval_zero_iterations(self, mock_llm_provider, mock_llm, mock_planner_llm):
        """Test plan approval with zero max_plan_iterations uses fallback values."""
        complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=complete_response.model_dump_json()))

        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
            enable_plan_approval=True,
            planner_llm=mock_planner_llm,
            max_plan_iterations=0,  # Zero iterations
        )

        state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        result = await agent.run(state)

        assert result is not None
        # Auto-approves, but never binds the generic safety-net fallback as a plan.
        assert result.plan_approved is True
        assert result.plan_title is None
        assert result.plan_sections == []
        assert result.get_approved_plan_context() is None


class TestClarifierAgentPlanApprovalInit:
    """Tests for plan approval initialization settings."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=llm)
        provider = MagicMock(spec=LLMProvider)
        provider.get = MagicMock(return_value=llm)
        return provider

    def test_init_with_plan_approval_disabled(self, mock_llm_provider):
        """Test initialization with plan approval disabled (default)."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
        )

        assert agent.enable_plan_approval is False
        assert agent.max_plan_iterations == 10

    def test_init_with_plan_approval_enabled(self, mock_llm_provider):
        """Test initialization with plan approval enabled."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
            enable_plan_approval=True,
        )

        assert agent.enable_plan_approval is True

    def test_init_with_custom_max_plan_iterations(self, mock_llm_provider):
        """Test initialization with custom max_plan_iterations."""
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
            max_plan_iterations=5,
        )

        assert agent.max_plan_iterations == 5

    def test_init_with_planner_llm(self, mock_llm_provider):
        """Test initialization with separate planner LLM."""
        planner_llm = MagicMock()
        agent = ClarifierAgent(
            llm_provider=mock_llm_provider,
            user_prompt_callback=AsyncMock(),
            planner_llm=planner_llm,
        )

        assert agent.planner_llm == planner_llm

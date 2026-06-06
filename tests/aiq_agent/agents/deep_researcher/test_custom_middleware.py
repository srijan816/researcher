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

"""Tests for custom middleware."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from aiq_agent.agents.deep_researcher.custom_middleware import ArtifactWriteValidationMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import PlanFileValidationMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import PlannerCommitGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import PostWriteReadbackGuardMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import SearchBudgetExhaustionRepairMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import TaskSearchBudgetMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ThinkingOnlyRepairMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolArgumentNormalizationMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolBudgetMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolNameSanitizationMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import ToolResultPruningMiddleware
from aiq_agent.agents.deep_researcher.custom_middleware import _budget_key
from aiq_agent.agents.deep_researcher.custom_middleware import _scoped_limit
from aiq_agent.agents.deep_researcher.custom_middleware import get_session_budget_snapshot
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_exhausted_tools
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_plan_validation_failures
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_planner_model_turns
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_recent_artifact_writes
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_task_search_counts
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_tool_counts
from aiq_agent.agents.deep_researcher.custom_middleware import reset_session_tool_limits
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_exhausted_tools
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_plan_validation_failures
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_planner_model_turns
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_recent_artifact_writes
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_task_search_counts
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_tool_counts
from aiq_agent.agents.deep_researcher.custom_middleware import set_session_tool_limits


class TestToolNameSanitizationMiddleware:
    """Tests for ToolNameSanitizationMiddleware."""

    @pytest.fixture
    def valid_tool_names(self):
        return ["advanced_web_search_tool", "paper_search_tool", "read_file", "write_file", "grep", "glob", "think"]

    @pytest.fixture
    def middleware(self, valid_tool_names):
        return ToolNameSanitizationMiddleware(valid_tool_names=valid_tool_names)

    def test_sanitize_channel_suffix(self, middleware):
        """Strip <|channel|> and everything after it."""
        assert (
            middleware._sanitize_tool_name("advanced_web_search_tool<|channel|>commentary")
            == "advanced_web_search_tool"
        )

    def test_sanitize_channel_json_suffix(self, middleware):
        """Strip <|channel|>json suffix."""
        assert middleware._sanitize_tool_name("advanced_web_search_tool<|channel|>json") == "advanced_web_search_tool"

    def test_sanitize_dot_suffix(self, middleware):
        """Strip .commentary suffix when base name is valid."""
        assert middleware._sanitize_tool_name("advanced_web_search_tool.commentary") == "advanced_web_search_tool"

    def test_sanitize_dot_exec_suffix(self, middleware):
        """Strip .exec suffix when base name is valid."""
        assert middleware._sanitize_tool_name("advanced_web_search_tool.exec") == "advanced_web_search_tool"

    def test_sanitize_paper_search_channel(self, middleware):
        """Strip channel suffix from paper_search_tool too."""
        assert middleware._sanitize_tool_name("paper_search_tool<|channel|>commentary") == "paper_search_tool"

    def test_map_open_file_to_read_file(self, middleware):
        """Map hallucinated open_file to read_file."""
        assert middleware._sanitize_tool_name("open_file") == "read_file"

    def test_map_find_to_grep(self, middleware):
        """Map hallucinated find to grep."""
        assert middleware._sanitize_tool_name("find") == "grep"

    def test_map_find_file_to_glob(self, middleware):
        """Map hallucinated find_file to glob."""
        assert middleware._sanitize_tool_name("find_file") == "glob"

    def test_passthrough_valid_name(self, middleware):
        """Valid tool names pass through unchanged."""
        assert middleware._sanitize_tool_name("advanced_web_search_tool") == "advanced_web_search_tool"

    def test_passthrough_unknown_invalid_name(self, middleware):
        """Unknown invalid names pass through unchanged (let framework report the error)."""
        assert middleware._sanitize_tool_name("totally_fake_tool") == "totally_fake_tool"

    def test_dot_suffix_with_invalid_base_passes_through(self, middleware):
        """Dot suffix stripping only applies when base name is valid."""
        assert middleware._sanitize_tool_name("fake_tool.commentary") == "fake_tool.commentary"

    @pytest.mark.asyncio
    async def test_awrap_model_call_sanitizes_tool_calls(self, middleware):
        """Integration: middleware sanitizes tool_calls in AIMessage."""
        from langchain.agents.middleware.types import ModelResponse

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "advanced_web_search_tool<|channel|>commentary", "args": {"question": "test"}, "id": "tc1"},
            ],
        )
        mock_response = ModelResponse(result=[ai_msg])
        mock_handler = AsyncMock(return_value=mock_response)
        mock_request = MagicMock()

        result = await middleware.awrap_model_call(mock_request, mock_handler)

        assert result.result[0].tool_calls[0]["name"] == "advanced_web_search_tool"

    @pytest.mark.asyncio
    async def test_awrap_model_call_no_tool_calls_passthrough(self, middleware):
        """Messages without tool_calls pass through unchanged."""
        from langchain.agents.middleware.types import ModelResponse

        ai_msg = AIMessage(content="Just text, no tools")
        mock_response = ModelResponse(result=[ai_msg])
        mock_handler = AsyncMock(return_value=mock_response)
        mock_request = MagicMock()

        result = await middleware.awrap_model_call(mock_request, mock_handler)

        assert result.result[0].content == "Just text, no tools"
        assert not result.result[0].tool_calls


class TestToolArgumentNormalizationMiddleware:
    """Tests for narrow MiniMax structured-tool argument repair."""

    @pytest.fixture
    def middleware(self):
        return ToolArgumentNormalizationMiddleware()

    @pytest.mark.asyncio
    async def test_normalizes_write_plan_aliases(self, middleware):
        ai_msg = AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "tc1",
                    "name": "write_plan",
                    "input": {
                        "title": "AI Use Cases",
                        "sections": ["Executive Summary"],
                        "search_queries": [
                            {
                                "query_string": "AI use cases ROI 2026 analyst report",
                                "purpose": "Find quantified value evidence",
                                "section": "Executive Summary",
                            }
                        ],
                        "requirements": "Use authoritative sources",
                    },
                }
            ],
            tool_calls=[
                {
                    "name": "write_plan",
                    "args": {
                        "title": "AI Use Cases",
                        "sections": ["Executive Summary"],
                        "search_queries": [
                            {
                                "query_string": "AI use cases ROI 2026 analyst report",
                                "purpose": "Find quantified value evidence",
                                "section": "Executive Summary",
                            }
                        ],
                        "requirements": "Use authoritative sources",
                    },
                    "id": "tc1",
                }
            ],
        )
        handler = AsyncMock(return_value=ModelResponse(result=[ai_msg]))
        request = MagicMock()

        result = await middleware.awrap_model_call(request, handler)

        args = result.result[0].tool_calls[0]["args"]
        assert args["report_title"] == "AI Use Cases"
        assert args["report_toc"] == [{"title": "Executive Summary"}]
        assert args["queries"][0]["query"] == "AI use cases ROI 2026 analyst report"
        assert args["queries"][0]["rationale"] == "Find quantified value evidence"
        assert args["queries"][0]["target_sections"] == ["Executive Summary"]
        assert args["constraints"] == ["Use authoritative sources"]
        assert result.result[0].content[0]["input"]["queries"][0]["query"] == "AI use cases ROI 2026 analyst report"

    @pytest.mark.asyncio
    async def test_defaults_missing_write_plan_constraints(self, middleware):
        ai_msg = AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "tc1",
                    "name": "write_plan",
                    "input": {
                        "report_title": "High-Leverage Debate Content",
                        "report_toc": [{"title": "Evidence Base"}],
                        "queries": [{"query": "secondary debate pedagogy evidence content blocks"}],
                    },
                }
            ],
            tool_calls=[
                {
                    "name": "write_plan",
                    "args": {
                        "report_title": "High-Leverage Debate Content",
                        "report_toc": [{"title": "Evidence Base"}],
                        "queries": [{"query": "secondary debate pedagogy evidence content blocks"}],
                    },
                    "id": "tc1",
                }
            ],
        )
        handler = AsyncMock(return_value=ModelResponse(result=[ai_msg]))
        request = MagicMock()

        result = await middleware.awrap_model_call(request, handler)

        args = result.result[0].tool_calls[0]["args"]
        assert args["constraints"] == [
            "Satisfy the user request with source-backed evidence and clearly note uncertainty or gaps."
        ]
        assert result.result[0].content[0]["input"]["constraints"] == args["constraints"]

    @pytest.mark.asyncio
    async def test_flattens_nested_write_plan_list_wrappers(self, middleware):
        ai_msg = AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "tc1",
                    "name": "write_plan",
                    "input": {
                        "report_title": "CRO Video Strategy",
                        "report_toc": {"item": {"item": {"title": "Mute-to-Unmute UX"}}},
                        "queries": {
                            "item": {
                                "item": {
                                    "query": "Research mute autoplay UX and browser policy evidence",
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
                                                {"claim_id": "C1", "claim": "Browser autoplay policy evidence"},
                                                {
                                                    "item": [
                                                        {"claim_id": "C2", "claim": "Kinetic typography case evidence"},
                                                        {
                                                            "item": {
                                                                "claim_id": "C3",
                                                                "claim": "Unmute UX recommendation",
                                                            }
                                                        },
                                                    ]
                                                },
                                            ]
                                        }
                                    },
                                }
                            }
                        },
                        "constraints": {"item": {"item": "Use first-party browser policy sources."}},
                    },
                }
            ],
            tool_calls=[
                {
                    "name": "write_plan",
                    "args": {
                        "report_title": "CRO Video Strategy",
                        "report_toc": {"item": {"item": {"title": "Mute-to-Unmute UX"}}},
                        "queries": {
                            "item": {
                                "item": {
                                    "query": "Research mute autoplay UX and browser policy evidence",
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
                                                {"claim_id": "C1", "claim": "Browser autoplay policy evidence"},
                                                {
                                                    "item": [
                                                        {"claim_id": "C2", "claim": "Kinetic typography case evidence"},
                                                        {
                                                            "item": {
                                                                "claim_id": "C3",
                                                                "claim": "Unmute UX recommendation",
                                                            }
                                                        },
                                                    ]
                                                },
                                            ]
                                        }
                                    },
                                }
                            }
                        },
                        "constraints": {"item": {"item": "Use first-party browser policy sources."}},
                    },
                    "id": "tc1",
                }
            ],
        )
        handler = AsyncMock(return_value=ModelResponse(result=[ai_msg]))
        request = MagicMock()

        result = await middleware.awrap_model_call(request, handler)

        args = result.result[0].tool_calls[0]["args"]
        assert args["report_toc"] == [{"title": "Mute-to-Unmute UX"}]
        assert args["queries"][0]["seed_queries"] == [
            "Chrome autoplay policy muted video",
            "iOS Safari autoplay muted playsinline policy",
        ]
        assert args["queries"][0]["target_sections"] == ["Mute-to-Unmute UX"]
        assert args["queries"][0]["target_claims"] == [
            {"claim_id": "C1", "claim": "Browser autoplay policy evidence"},
            {"claim_id": "C2", "claim": "Kinetic typography case evidence"},
            {"claim_id": "C3", "claim": "Unmute UX recommendation"},
        ]
        assert args["constraints"] == ["Use first-party browser policy sources."]

    @pytest.mark.asyncio
    async def test_normalizes_file_path_alias(self, middleware):
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "/shared/claim_table.json"},
                    "id": "tc1",
                }
            ],
        )
        handler = AsyncMock(return_value=ModelResponse(result=[ai_msg]))
        request = MagicMock()

        result = await middleware.awrap_model_call(request, handler)

        assert result.result[0].tool_calls[0]["args"]["file_path"] == "/shared/claim_table.json"


class TestSourceRegistryMiddleware:
    """Tests for SourceRegistryMiddleware allowlist + source extraction."""

    @pytest.fixture
    def source_tools(self):
        return {"advanced_web_search_tool", "knowledge_search", "paper_search_tool"}

    @pytest.fixture
    def middleware(self, source_tools):
        return SourceRegistryMiddleware(source_tool_names=source_tools)

    def _make_request(self, tool_name: str):
        req = MagicMock()
        req.tool_call = {"name": tool_name}
        return req

    def _make_tool_result(self, content: str):
        return ToolMessage(content=content, tool_call_id="tc1")

    # -- URL extraction --

    @pytest.mark.asyncio
    async def test_url_source_captured(self, middleware):
        """URLs in tool output are extracted and registered."""
        content = "Found result at https://arxiv.org/abs/2401.00001"
        handler = AsyncMock(return_value=self._make_tool_result(content))
        request = self._make_request("advanced_web_search_tool")

        await middleware.awrap_tool_call(request, handler)

        sources = middleware.registry.all_sources()
        assert len(sources) == 1
        assert sources[0].url == "https://arxiv.org/abs/2401.00001"

    @pytest.mark.asyncio
    async def test_multiple_urls_captured(self, middleware):
        """Multiple URLs from a single tool call are all captured."""
        content = "Result from https://a.com/page and also https://b.com/page"
        handler = AsyncMock(return_value=self._make_tool_result(content))
        request = self._make_request("advanced_web_search_tool")

        await middleware.awrap_tool_call(request, handler)

        urls = {s.url for s in middleware.registry.all_sources()}
        assert urls == {"https://a.com/page", "https://b.com/page"}

    @pytest.mark.asyncio
    async def test_knowledge_layer_citation_key_captured(self, middleware):
        """Knowledge layer citation keys are captured via regex."""
        content = (
            "--- Result 1 ---\n"
            "Source: report.pdf\n"
            "Page: 5\n"
            "Citation: report.pdf, p.5\n"
            "Content Type: pdf\n"
            "\nSome content here."
        )
        handler = AsyncMock(return_value=self._make_tool_result(content))
        request = self._make_request("knowledge_search")

        await middleware.awrap_tool_call(request, handler)

        sources = middleware.registry.all_sources()
        assert len(sources) == 1
        assert sources[0].citation_key == "report.pdf, p.5"

    # -- Allowlist filtering --

    @pytest.mark.asyncio
    async def test_think_tool_ignored(self, middleware):
        """Internal tools not in the allowlist are ignored."""
        content = "Thinking about https://hallucinated.com"
        handler = AsyncMock(return_value=self._make_tool_result(content))
        request = self._make_request("think")

        await middleware.awrap_tool_call(request, handler)

        assert len(middleware.registry.all_sources()) == 0

    @pytest.mark.asyncio
    async def test_unknown_tool_ignored(self, middleware):
        """Tools not in the allowlist are ignored."""
        content = "https://unknown.com/data"
        handler = AsyncMock(return_value=self._make_tool_result(content))
        request = self._make_request("some_random_tool")

        await middleware.awrap_tool_call(request, handler)

        assert len(middleware.registry.all_sources()) == 0

    @pytest.mark.asyncio
    async def test_mixed_source_tools(self, middleware):
        """Multiple tool calls — only allowlisted tools contribute sources."""
        h1 = AsyncMock(return_value=self._make_tool_result("See https://a.com"))
        h2 = AsyncMock(return_value=self._make_tool_result("See https://b.com"))

        await middleware.awrap_tool_call(self._make_request("advanced_web_search_tool"), h1)
        await middleware.awrap_tool_call(self._make_request("paper_search_tool"), h2)

        urls = {s.url for s in middleware.registry.all_sources()}
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    # -- Edge cases --

    @pytest.mark.asyncio
    async def test_empty_content_skipped(self, middleware):
        """Empty content is ignored gracefully."""
        handler = AsyncMock(return_value=self._make_tool_result(""))
        request = self._make_request("advanced_web_search_tool")

        await middleware.awrap_tool_call(request, handler)

        assert len(middleware.registry.all_sources()) == 0

    @pytest.mark.asyncio
    async def test_non_tool_message_passthrough(self, middleware):
        """Non-ToolMessage results pass through without error."""
        handler = AsyncMock(return_value=AIMessage(content="just an AI reply"))
        request = self._make_request("advanced_web_search_tool")

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, AIMessage)
        assert len(middleware.registry.all_sources()) == 0

    @pytest.mark.asyncio
    async def test_default_empty_allowlist_captures_nothing(self):
        """Middleware with no source_tool_names captures nothing."""
        mw = SourceRegistryMiddleware()
        content = "See https://should-not-be-captured.com"
        handler = AsyncMock(return_value=ToolMessage(content=content, tool_call_id="tc1"))
        request = MagicMock()
        request.tool_call = {"name": "advanced_web_search_tool"}

        await mw.awrap_tool_call(request, handler)

        assert len(mw.registry.all_sources()) == 0

    @pytest.mark.asyncio
    async def test_content_returned_unchanged(self, middleware):
        """Tool result content is not modified by the middleware."""
        content = "Results from https://example.com/page"
        handler = AsyncMock(return_value=self._make_tool_result(content))
        request = self._make_request("advanced_web_search_tool")

        result = await middleware.awrap_tool_call(request, handler)

        assert result.content == content


class TestSearchBudgetFamilies:
    """Search tools should share a single family budget."""

    def test_search_tools_share_budget_keys(self):
        assert _budget_key("advanced_web_search_tool", "planner") == "planner:search"
        assert _budget_key("web_search_tool", "researcher") == "researcher:search"
        assert _budget_key("exa_web_search_tool") == "search"

    def test_scoped_limit_prefers_family_budget(self):
        limits = {"search": 12, "planner:search": 4, "advanced_web_search_tool": 99}

        assert _scoped_limit(limits, "advanced_web_search_tool") == 12
        assert _scoped_limit(limits, "web_search_tool", "planner") == 4


class TestToolBudgetMiddleware:
    """Tests for scoped tool budgets."""

    class Request:
        def __init__(self, tool_name: str, tool_id: str):
            self.tool_call = {"name": tool_name, "id": tool_id}

    @pytest.mark.asyncio
    async def test_planner_budget_does_not_consume_researcher_budget(self):
        """Planner search exhaustion should not spend or block the researcher pool."""
        counts_token = set_session_tool_counts({})
        limits_token = set_session_tool_limits(
            {
                "planner:search": 1,
                "search": 2,
            }
        )
        exhausted_token = set_session_exhausted_tools(set())
        try:
            planner = ToolBudgetMiddleware({"advanced_web_search_tool": 2}, scope="planner")
            researcher = ToolBudgetMiddleware({"advanced_web_search_tool": 2}, scope="researcher")
            handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="tc"))

            first = await planner.awrap_tool_call(self.Request("advanced_web_search_tool", "p1"), handler)
            second = await planner.awrap_tool_call(self.Request("advanced_web_search_tool", "p2"), handler)
            third = await researcher.awrap_tool_call(self.Request("advanced_web_search_tool", "r1"), handler)

            assert first.content == "ok"
            assert "GLOBAL_SEARCH_BUDGET_EXHAUSTED" in second.content
            assert third.content == "ok"
        finally:
            reset_session_tool_counts(counts_token)
            reset_session_tool_limits(limits_token)
            reset_session_exhausted_tools(exhausted_token)

    def test_session_budget_snapshot_reports_remaining_counts(self):
        counts_token = set_session_tool_counts({"researcher:search": 3})
        limits_token = set_session_tool_limits({"researcher:search": 8})
        exhausted_token = set_session_exhausted_tools({"planner:search"})
        try:
            snapshot = get_session_budget_snapshot()

            budgets = {entry["key"]: entry for entry in snapshot["budgets"]}
            assert budgets["researcher:search"]["used"] == 3
            assert budgets["researcher:search"]["remaining"] == 5
            assert snapshot["exhausted"] == ["planner:search"]
        finally:
            reset_session_tool_counts(counts_token)
            reset_session_tool_limits(limits_token)
            reset_session_exhausted_tools(exhausted_token)


class TestTaskSearchBudgetMiddleware:
    """Tests for per-researcher-task search budgets."""

    class Request:
        def __init__(self, tool_name: str = "advanced_web_search_tool"):
            self.tool_call = {"name": tool_name, "id": f"{tool_name}-1"}
            self.messages = [
                HumanMessage(
                    content=(
                        "You are researching task_id: Q2.\n"
                        "task_category: primary_data\n"
                        "budget_percent: 25\n"
                        "Search budget: 2 search calls for this task.\n"
                    )
                )
            ]

    @pytest.mark.asyncio
    async def test_enforces_per_task_search_budget(self):
        middleware = TaskSearchBudgetMiddleware({"advanced_web_search_tool", "web_search_tool"})
        token = set_session_task_search_counts({})
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="search-1"))
        try:
            first = await middleware.awrap_tool_call(self.Request(), handler)
            second = await middleware.awrap_tool_call(self.Request(), handler)
            third = await middleware.awrap_tool_call(self.Request(), handler)

            assert first.content == "ok"
            assert second.content == "ok"
            assert "TASK_SEARCH_BUDGET_EXHAUSTED" in third.content
            assert handler.await_count == 2
        finally:
            reset_session_task_search_counts(token)

    @pytest.mark.asyncio
    async def test_ignores_non_search_tools(self):
        middleware = TaskSearchBudgetMiddleware({"advanced_web_search_tool", "web_search_tool"})
        token = set_session_task_search_counts({})
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="write-1"))
        try:
            result = await middleware.awrap_tool_call(self.Request("write_file"), handler)

            assert result.content == "written"
            assert handler.await_count == 1
        finally:
            reset_session_task_search_counts(token)


class TestPlanFileValidationMiddleware:
    """Tests for hard validation of planner writes."""

    class Request:
        def __init__(self, content: str, path: str = "/shared/plan.json"):
            self.tool_call = {
                "name": "write_file",
                "id": "plan-write",
                "args": {
                    "file_path": path,
                    "content": content,
                },
            }

    @staticmethod
    def _valid_plan() -> str:
        return json.dumps(
            {
                "task_analysis": {"user_intent": "Research AI use cases"},
                "report_title": "AI Use Cases in 2026",
                "report_toc": [{"id": "1", "title": "Ranked Use Cases"}],
                "constraints": ["Use current sources"],
                "output_style": {"mode": "standard_report"},
                "queries": [
                    {
                        "query": "highest value AI use cases 2026 ROI analyst report",
                        "tool": "advanced_web_search_tool",
                        "target_sections": ["Ranked Use Cases"],
                    }
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_rejects_plan_write_missing_queries(self):
        middleware = PlanFileValidationMiddleware()
        bad_plan = json.dumps(
            {
                "task_analysis": {"user_intent": "Research AI use cases"},
                "report_title": "AI Use Cases in 2026",
                "report_toc": [{"id": "1", "title": "Ranked Use Cases"}],
                "constraints": ["Use current sources"],
                "output_style": {"mode": "standard_report"},
            }
        )
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="plan-write"))

        result = await middleware.awrap_tool_call(self.Request(bad_plan), handler)

        assert handler.await_count == 0
        assert "PLAN_FILE_VALIDATION_FAILED" in result.content
        assert "missing top-level field: queries" in result.content

    @pytest.mark.asyncio
    async def test_rejects_truncated_plan_write(self):
        middleware = PlanFileValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="plan-write"))

        result = await middleware.awrap_tool_call(
            self.Request('{"task_analysis": {}\n\n[... truncated tool argument from 40000 chars ...]'),
            handler,
        )

        assert handler.await_count == 0
        assert "PLAN_FILE_VALIDATION_FAILED" in result.content
        assert "truncation/omission marker" in result.content

    @pytest.mark.asyncio
    async def test_second_plan_validation_failure_escalates_to_write_plan(self):
        middleware = PlanFileValidationMiddleware()
        token = set_session_plan_validation_failures(0)
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="plan-write"))
        try:
            first = await middleware.awrap_tool_call(self.Request("{bad json"), handler)
            second = await middleware.awrap_tool_call(self.Request("{bad json again"), handler)

            assert "Use the typed write_plan tool now" in first.content
            assert "second planner JSON validation failure" in second.content
            assert "Do not call write_file for /shared/plan.json again" in second.content
        finally:
            reset_session_plan_validation_failures(token)

    @pytest.mark.asyncio
    async def test_allows_complete_plan_write(self):
        middleware = PlanFileValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="plan-write"))

        result = await middleware.awrap_tool_call(self.Request(self._valid_plan()), handler)

        assert handler.await_count == 1
        assert result.content == "written"

    @pytest.mark.asyncio
    async def test_ignores_non_plan_file_writes(self):
        middleware = PlanFileValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="notes-write"))

        result = await middleware.awrap_tool_call(self.Request("notes", path="/shared/notes.md"), handler)

        assert handler.await_count == 1
        assert result.content == "written"


class TestArtifactWriteValidationMiddleware:
    """Tests for rejecting artifact writes polluted by truncation markers."""

    class Request:
        def __init__(
            self, content: str, *, tool_name: str = "write_file", path: str = "/shared/section_briefs/topic.md"
        ):
            args = {"file_path": path}
            if tool_name == "edit_file":
                args.update({"old_string": "old", "new_string": content})
            else:
                args["content"] = content
            self.tool_call = {"name": tool_name, "id": "artifact-write", "args": args}

    @pytest.mark.asyncio
    async def test_rejects_shared_artifact_with_argument_truncation_marker(self):
        middleware = ArtifactWriteValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="artifact-write"))

        result = await middleware.awrap_tool_call(
            self.Request("- C4 — Bologn...(argument truncated)"),
            handler,
        )

        assert handler.await_count == 0
        assert "ARTIFACT_WRITE_VALIDATION_FAILED" in result.content
        assert "truncation/omission marker" in result.content
        assert "(argument truncated)" not in result.content

    @pytest.mark.asyncio
    async def test_rejects_copied_historical_write_redaction_without_repeating_marker(self):
        middleware = ArtifactWriteValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="artifact-write"))

        result = await middleware.awrap_tool_call(
            self.Request(
                "WRITE_FILE_CONTENT_STORED_SUCCESSFULLY at /shared/test.txt; 35 characters are hidden",
                path="/shared/test.txt",
            ),
            handler,
        )

        assert handler.await_count == 0
        assert "ARTIFACT_WRITE_VALIDATION_FAILED" in result.content
        assert "historical write-redaction notice" in result.content
        assert "WRITE_FILE_CONTENT_STORED_SUCCESSFULLY" not in result.content
        assert "Do not test the write tool" in result.content

    @pytest.mark.asyncio
    async def test_rejects_edit_file_new_string_with_truncation_marker(self):
        middleware = ArtifactWriteValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="edited", tool_call_id="artifact-write"))

        result = await middleware.awrap_tool_call(
            self.Request("replacement [... truncated tool argument from 9000 chars ...]", tool_name="edit_file"),
            handler,
        )

        assert handler.await_count == 0
        assert "ARTIFACT_WRITE_VALIDATION_FAILED" in result.content

    @pytest.mark.asyncio
    async def test_allows_legitimate_shortened_evidence_extract(self):
        middleware = ArtifactWriteValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="artifact-write"))

        result = await middleware.awrap_tool_call(
            self.Request("source excerpt [... truncated from 9000 chars]", path="/shared/extracts/topic.json"),
            handler,
        )

        assert handler.await_count == 1
        assert result.content == "written"

    @pytest.mark.asyncio
    async def test_allows_clean_shared_artifact_write(self):
        middleware = ArtifactWriteValidationMiddleware()
        handler = AsyncMock(return_value=ToolMessage(content="written", tool_call_id="artifact-write"))

        result = await middleware.awrap_tool_call(self.Request("complete concise artifact"), handler)

        assert handler.await_count == 1
        assert result.content == "written"


class TestPostWriteReadbackGuardMiddleware:
    """Tests for suppressing immediate artifact self-verification reads."""

    class Request:
        def __init__(self, tool_name: str, path: str = "/shared/notes_topic.md"):
            self.tool_call = {
                "name": tool_name,
                "id": f"{tool_name}-1",
                "args": {
                    "file_path": path,
                    **({"content": "complete notes"} if tool_name == "write_file" else {}),
                },
            }

    @pytest.mark.asyncio
    async def test_skips_immediate_readback_after_successful_write(self):
        middleware = PostWriteReadbackGuardMiddleware(suppress_read_count=1)
        token = set_session_recent_artifact_writes({})
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content="written", tool_call_id="write_file-1", name="write_file"),
                ToolMessage(content="actual file content", tool_call_id="read_file-1", name="read_file"),
            ]
        )
        try:
            write_result = await middleware.awrap_tool_call(self.Request("write_file"), handler)
            read_result = await middleware.awrap_tool_call(self.Request("read_file"), handler)
            second_read_result = await middleware.awrap_tool_call(self.Request("read_file"), handler)

            assert write_result.content == "written"
            assert "READ_AFTER_WRITE_CONFIRMED" in read_result.content
            assert second_read_result.content == "actual file content"
            assert handler.await_count == 2
        finally:
            reset_session_recent_artifact_writes(token)

    @pytest.mark.asyncio
    async def test_does_not_track_failed_write(self):
        middleware = PostWriteReadbackGuardMiddleware(suppress_read_count=1)
        token = set_session_recent_artifact_writes({})
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content="Error: File already exists", tool_call_id="write_file-1", name="write_file"),
                ToolMessage(content="actual file content", tool_call_id="read_file-1", name="read_file"),
            ]
        )
        try:
            await middleware.awrap_tool_call(self.Request("write_file"), handler)
            read_result = await middleware.awrap_tool_call(self.Request("read_file"), handler)

            assert read_result.content == "actual file content"
            assert handler.await_count == 2
        finally:
            reset_session_recent_artifact_writes(token)

    @pytest.mark.asyncio
    async def test_skips_immediate_grep_after_successful_write(self):
        middleware = PostWriteReadbackGuardMiddleware(suppress_read_count=1)
        token = set_session_recent_artifact_writes({})
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content="written", tool_call_id="write_file-1", name="write_file"),
                ToolMessage(content="grep result", tool_call_id="grep-1", name="grep"),
            ]
        )
        try:
            write_result = await middleware.awrap_tool_call(self.Request("write_file"), handler)
            grep_result = await middleware.awrap_tool_call(self.Request("grep"), handler)
            second_grep_result = await middleware.awrap_tool_call(self.Request("grep"), handler)

            assert write_result.content == "written"
            assert "READ_AFTER_WRITE_CONFIRMED" in grep_result.content
            assert second_grep_result.content == "grep result"
            assert handler.await_count == 2
        finally:
            reset_session_recent_artifact_writes(token)

    @pytest.mark.asyncio
    async def test_readback_guard_normalizes_shared_paths(self):
        middleware = PostWriteReadbackGuardMiddleware(suppress_read_count=1)
        token = set_session_recent_artifact_writes({})
        handler = AsyncMock(
            side_effect=[
                ToolMessage(content="written", tool_call_id="write_file-1", name="write_file"),
                ToolMessage(content="actual file content", tool_call_id="read_file-1", name="read_file"),
            ]
        )
        try:
            await middleware.awrap_tool_call(self.Request("write_file", path="/shared/notes_topic.md"), handler)
            read_result = await middleware.awrap_tool_call(
                self.Request("read_file", path="shared/notes_topic.md"),
                handler,
            )

            assert "READ_AFTER_WRITE_CONFIRMED" in read_result.content
            assert handler.await_count == 1
        finally:
            reset_session_recent_artifact_writes(token)


class TestPlannerCommitGuardMiddleware:
    """Tests for planner turn-budget commit enforcement."""

    class Request:
        def __init__(self, messages=None, tools=None, tool_choice=None):
            self.messages = messages or [HumanMessage(content="plan this")]
            self.tools = tools or [
                SimpleNamespace(name="think"),
                SimpleNamespace(name="write_plan"),
                SimpleNamespace(name="advanced_web_search_tool"),
            ]
            self.tool_choice = tool_choice

        def override(self, **kwargs):
            return TestPlannerCommitGuardMiddleware.Request(
                messages=kwargs.get("messages", self.messages),
                tools=kwargs.get("tools", self.tools),
                tool_choice=kwargs.get("tool_choice", self.tool_choice),
            )

    @pytest.mark.asyncio
    async def test_forces_write_plan_after_turn_budget(self):
        from langchain.agents.middleware.types import ModelResponse

        middleware = PlannerCommitGuardMiddleware(max_model_turns=4, max_repairs=0)
        token = set_session_planner_model_turns(3)
        try:
            thinking_detour = AIMessage(
                content="I have enough grounding; let me think once more.",
                tool_calls=[{"name": "think", "args": {"thought": "more planning"}, "id": "think-1"}],
            )
            committed = AIMessage(
                content="",
                tool_calls=[{"name": "write_plan", "args": {"report_title": "Plan"}, "id": "write-plan-1"}],
            )
            handler = AsyncMock(
                side_effect=[ModelResponse(result=[thinking_detour]), ModelResponse(result=[committed])]
            )

            result = await middleware.awrap_model_call(self.Request(), handler)

            assert handler.await_count == 2
            forced_request = handler.await_args_list[1].args[0]
            assert [tool.name for tool in forced_request.tools] == ["write_plan"]
            assert forced_request.tool_choice == "write_plan"
            assert "only action must be the `write_plan` tool" in forced_request.messages[-1].content
            assert result.result[0].tool_calls[0]["name"] == "write_plan"
        finally:
            reset_session_planner_model_turns(token)

    @pytest.mark.asyncio
    async def test_allows_pre_budget_thinking_turn(self):
        from langchain.agents.middleware.types import ModelResponse

        middleware = PlannerCommitGuardMiddleware(max_model_turns=4, max_repairs=0)
        token = set_session_planner_model_turns(0)
        try:
            thinking_detour = AIMessage(
                content="I should reason about source strategy.",
                tool_calls=[{"name": "think", "args": {"thought": "source strategy"}, "id": "think-1"}],
            )
            handler = AsyncMock(return_value=ModelResponse(result=[thinking_detour]))

            result = await middleware.awrap_model_call(self.Request(), handler)

            assert handler.await_count == 1
            assert result.result[0].tool_calls[0]["name"] == "think"
        finally:
            reset_session_planner_model_turns(token)


class TestThinkingOnlyRepairMiddleware:
    """Tests for MiniMax thinking-only response repair."""

    @pytest.mark.asyncio
    async def test_repairs_thinking_only_response(self):
        from langchain.agents.middleware.types import ModelResponse

        middleware = ThinkingOnlyRepairMiddleware(max_repairs=1)
        thinking_only = AIMessage(content=[{"type": "thinking", "thinking": "I should write the report now."}])
        repaired = AIMessage(content="Final report prose\n\n## Sources\n\n[1] https://example.com")
        handler = AsyncMock(side_effect=[ModelResponse(result=[thinking_only]), ModelResponse(result=[repaired])])

        request = MagicMock()
        request.messages = [HumanMessage(content="Research question")]

        def override(**kwargs):
            updated = MagicMock()
            updated.messages = kwargs.get("messages", request.messages)
            updated.override = override
            return updated

        request.override = override

        response = await middleware.awrap_model_call(request, handler)

        assert response.result[0].content == repaired.content
        assert handler.await_count == 2
        repaired_request = handler.await_args_list[1].args[0]
        assert "thinking-only" in repaired_request.messages[-1].content
        assert "write_file" in repaired_request.messages[-1].content

    @pytest.mark.asyncio
    async def test_allows_thinking_with_tool_call(self):
        from langchain.agents.middleware.types import ModelResponse

        middleware = ThinkingOnlyRepairMiddleware(max_repairs=1)
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "I need a source."},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "advanced_web_search_tool",
                    "input": {"question": "x"},
                },
            ],
            tool_calls=[{"name": "advanced_web_search_tool", "args": {"question": "x"}, "id": "call-1"}],
        )
        handler = AsyncMock(return_value=ModelResponse(result=[message]))
        request = MagicMock()
        request.messages = [HumanMessage(content="Research question")]

        response = await middleware.awrap_model_call(request, handler)

        assert response.result[0] == message
        assert handler.await_count == 1


class TestSearchBudgetExhaustionRepairMiddleware:
    """Tests for stopping exhausted-search retry loops."""

    class Request:
        def __init__(self, messages):
            self.messages = messages

        def override(self, **kwargs):
            return TestSearchBudgetExhaustionRepairMiddleware.Request(kwargs.get("messages", self.messages))

    def _search_call_message(self):
        return AIMessage(
            content=[
                {"type": "thinking", "thinking": "The search budget is exhausted. Let me try one more time."},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "advanced_web_search_tool",
                    "input": {"question": "one more search"},
                },
            ],
            tool_calls=[{"name": "advanced_web_search_tool", "args": {"question": "one more search"}, "id": "call-1"}],
        )

    @pytest.mark.asyncio
    async def test_repairs_exhausted_search_call_into_write_file(self):
        from langchain.agents.middleware.types import ModelResponse

        middleware = SearchBudgetExhaustionRepairMiddleware(
            {"advanced_web_search_tool", "web_search_tool"},
            max_repairs=1,
        )
        token = set_session_exhausted_tools({"search"})
        try:
            write_file_message = AIMessage(
                content="Writing notes from existing evidence.",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/shared/notes.txt", "content": "Partial notes"},
                        "id": "write-1",
                    }
                ],
            )
            handler = AsyncMock(
                side_effect=[
                    ModelResponse(result=[self._search_call_message()]),
                    ModelResponse(result=[write_file_message]),
                ]
            )
            request = self.Request([HumanMessage(content="Research question")])

            response = await middleware.awrap_model_call(request, handler)

            assert handler.await_count == 2
            assert response.result[0].tool_calls[0]["name"] == "write_file"
            repaired_request = handler.await_args_list[1].args[0]
            assert "forbidden from calling" in repaired_request.messages[-1].content
            assert "write_file" in repaired_request.messages[-1].content
        finally:
            reset_session_exhausted_tools(token)

    @pytest.mark.asyncio
    async def test_strips_exhausted_search_call_after_failed_repair(self):
        from langchain.agents.middleware.types import ModelResponse

        middleware = SearchBudgetExhaustionRepairMiddleware(
            {"advanced_web_search_tool", "web_search_tool"},
            max_repairs=1,
        )
        token = set_session_exhausted_tools({"search"})
        try:
            handler = AsyncMock(return_value=ModelResponse(result=[self._search_call_message()]))
            request = self.Request([HumanMessage(content="Research question")])

            response = await middleware.awrap_model_call(request, handler)

            assert handler.await_count == 2
            assert not response.result[0].tool_calls
            assert "stop searching" in response.result[0].content
        finally:
            reset_session_exhausted_tools(token)


class TestToolResultPruningMiddleware:
    """Tests for model-context tool result pruning."""

    @pytest.mark.asyncio
    async def test_recent_tool_result_is_still_hard_capped(self):
        """Recent giant tool results should not be sent to the next model call intact."""
        middleware = ToolResultPruningMiddleware(keep_last_n=10, max_chars=20, recent_max_chars=50)
        messages = [
            AIMessage(content="call tool"),
            ToolMessage(content="x" * 200, tool_call_id="tc1", name="advanced_web_search_tool"),
        ]
        request = MagicMock()
        request.messages = messages
        request.override.side_effect = lambda **kwargs: MagicMock(messages=kwargs["messages"])

        async def handler(req):
            return req.messages

        pruned = await middleware.awrap_model_call(request, handler)

        assert len(pruned[1].content) < 350
        assert "TOOL_RESULT_DISPLAY_SHORTENED" in pruned[1].content

    @pytest.mark.asyncio
    async def test_historical_write_file_content_is_not_replaced_with_placeholder(self):
        """Artifact writes must not be shown as empty or placeholder content later."""
        middleware = ToolResultPruningMiddleware(max_tool_call_arg_chars=1000)
        original_content = "x" * 5000
        tool_call = {
            "name": "write_file",
            "args": {"file_path": "/shared/report.md", "content": original_content},
            "id": "call-1",
        }
        messages = [
            AIMessage(
                content=[{"type": "tool_use", "id": "call-1", "name": "write_file", "input": tool_call["args"]}],
                tool_calls=[tool_call],
            )
        ]
        request = MagicMock()
        request.messages = messages
        request.override.side_effect = lambda **kwargs: MagicMock(messages=kwargs["messages"])

        async def handler(req):
            return req.messages

        pruned = await middleware.awrap_model_call(request, handler)

        pruned_call = pruned[0].tool_calls[0]
        assert pruned_call["args"]["content"] == original_content
        assert "_aiq_history_content_redacted" not in pruned_call["args"]
        assert pruned[0].content[0]["input"]["content"] == pruned_call["args"]["content"]
        assert "_aiq_history_content_redacted" not in pruned[0].content[0]["input"]

    @pytest.mark.asyncio
    async def test_historical_non_file_tool_args_are_still_pruned(self):
        """Non-artifact arguments can still be shortened to protect context."""
        middleware = ToolResultPruningMiddleware(max_tool_call_arg_chars=1000)
        tool_call = {
            "name": "advanced_web_search_tool",
            "args": {"query": "x" * 5000},
            "id": "call-1",
        }
        messages = [
            AIMessage(
                content=[
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "advanced_web_search_tool",
                        "input": tool_call["args"],
                    }
                ],
                tool_calls=[tool_call],
            )
        ]
        request = MagicMock()
        request.messages = messages
        request.override.side_effect = lambda **kwargs: MagicMock(messages=kwargs["messages"])

        async def handler(req):
            return req.messages

        pruned = await middleware.awrap_model_call(request, handler)

        assert "TOOL_ARGUMENT_DISPLAY_SHORTENED" in pruned[0].tool_calls[0]["args"]["query"]
        assert len(pruned[0].tool_calls[0]["args"]["query"]) < 1200

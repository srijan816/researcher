# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NAT register function for the Claude Code research agent."""

import logging

from langchain_core.messages import HumanMessage
from pydantic import Field

from aiq_agent.common import DEFAULT_RESEARCH_DEPTH
from aiq_agent.common import ResearchDepthTier
from aiq_agent.common import _create_chat_response
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.function import FunctionBaseConfig

from .agent import ClaudeResearcherAgent
from .models import ClaudeResearchAgentState

logger = logging.getLogger(__name__)


class ClaudeResearchAgentConfig(FunctionBaseConfig, name="claude_research_agent"):
    """Configuration for the Claude Code research bridge."""

    timeout_seconds: int = Field(default=3600, description="Maximum wall time for a Claude Code research run.")
    default_depth: ResearchDepthTier = Field(
        default=DEFAULT_RESEARCH_DEPTH,
        description="Depth to use when no async job depth is supplied.",
    )
    verbose: bool = Field(default=True, description="Reserved for parity with other agents.")


@register_function(config_type=ClaudeResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def claude_research_agent(config: ClaudeResearchAgentConfig, _builder: Builder):
    """Claude Code driven research function for NAT/local invocation."""
    agent = ClaudeResearcherAgent(config=config)

    async def _run(state: ClaudeResearchAgentState) -> ChatResponse:
        result = await agent.run(state)
        content = result.messages[-1].content if result.messages else ""
        return _create_chat_response(content)

    try:
        yield FunctionInfo.from_fn(_run, description="Run a Claude Code research job")
    except GeneratorExit:
        logger.debug("Claude research function generator closed")


class ClaudeResearchWorkflowConfig(FunctionBaseConfig, name="claude_research_workflow"):
    """Simple workflow wrapper for evaluation and CLI runs."""


@register_function(config_type=ClaudeResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def claude_research_workflow(_config: ClaudeResearchWorkflowConfig, builder: Builder):
    """Workflow wrapper around ``claude_research_agent``."""
    claude_research_fn = await builder.get_function("claude_research_agent")

    async def _run(input_message: str) -> ChatResponse:
        state = ClaudeResearchAgentState(messages=[HumanMessage(content=input_message)])
        result = await claude_research_fn.ainvoke(state)
        if isinstance(result, ChatResponse):
            return result
        if isinstance(result, ClaudeResearchAgentState) and result.messages:
            return _create_chat_response(result.messages[-1].content)
        return _create_chat_response(str(result))

    try:
        yield FunctionInfo.from_fn(_run, description="Run a Claude Code research workflow")
    except GeneratorExit:
        logger.debug("Claude research workflow generator closed")

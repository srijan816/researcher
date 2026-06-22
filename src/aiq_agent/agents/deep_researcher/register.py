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

"""NAT register function for deep research agent."""

import logging

from langchain_core.messages import HumanMessage
from pydantic import Field

from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common import VerboseTraceCallback
from aiq_agent.common import _create_chat_response
from aiq_agent.common import filter_tools_by_sources
from aiq_agent.common import is_verbose
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatResponse
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from .agent import DeepResearcherAgent
from .deepagents_runtime import SandboxConfig
from .deepagents_runtime import SkillsConfig
from .models import DeepResearchAgentState

logger = logging.getLogger(__name__)


class DeepResearchAgentConfig(FunctionBaseConfig, name="deep_research_agent"):
    """Configuration for the deep research agent."""

    orchestrator_llm: LLMRef = Field(..., description="LLM for orchestrator")
    medium_orchestrator_llm: LLMRef | None = Field(
        default=None,
        description="Optional thinking-off orchestrator/synthesis LLM for the medium tier",
    )
    deeper_orchestrator_llm: LLMRef | None = Field(
        default=None,
        description="Optional orchestrator/synthesis LLM for the deeper tier",
    )
    deeper_planner_llm: LLMRef | None = Field(default=None, description="Optional planner LLM for the deeper tier")
    deeper_researcher_llm: LLMRef | None = Field(
        default=None,
        description="Optional researcher LLM for the deeper tier",
    )
    deep_planner_llm: LLMRef | None = Field(default=None, description="Optional planner LLM for the deep tier")
    deep_researcher_llm: LLMRef | None = Field(
        default=None,
        description="Optional researcher LLM for the deep tier",
    )
    researcher_llm: LLMRef | None = Field(default=None, description="LLM for researcher")
    planner_llm: LLMRef | None = Field(default=None, description="LLM for planner")
    verifier_llm: LLMRef | None = Field(
        default=None,
        description=(
            "Optional dedicated LLM for adversarial claim verification. "
            "Falls back to the tier researcher (non-thinking) LLM when unset."
        ),
    )
    tools: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list,
        description="Explicit tool list. Empty = inherit all from data_source_registry.",
    )
    exclude_tools: list[str] = Field(
        default_factory=list,
        description="Tool names to exclude when inheriting from registry.",
    )
    max_loops: int = Field(default=2)
    verbose: bool = Field(default=True)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    sandbox: SandboxConfig | None = Field(
        default=None,
        description="Optional DeepAgents sandbox backend for execute support.",
    )


@register_function(config_type=DeepResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_research_agent(config: DeepResearchAgentConfig, builder: Builder):
    """Deep research agent using multi-phase workflow."""
    if config.tools:
        tool_refs = config.tools
    else:
        from aiq_agent.common import get_all_tool_refs

        tool_refs = get_all_tool_refs()

    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if config.exclude_tools:
        excluded = set(config.exclude_tools)
        tools = [t for t in tools if getattr(t, "name", "") not in excluded]

    from aiq_agent.common import validate_tool_availability

    is_valid, available_count, unavailable = validate_tool_availability(
        tools,
        research_type="deep research",
    )
    if not is_valid:
        logger.warning(
            "Startup check: no tools available for deep research. "
            "All queries will fail until at least one tool is properly configured.",
        )

    llm = await builder.get_llm(config.orchestrator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    provider = LLMProvider()
    provider.set_default(llm)

    provider.configure(LLMRole.ORCHESTRATOR, llm)
    if config.medium_orchestrator_llm:
        medium_orchestrator_llm = await builder.get_llm(
            config.medium_orchestrator_llm,
            wrapper_type=LLMFrameworkEnum.LANGCHAIN,
        )
        provider.configure(LLMRole.MEDIUM_ORCHESTRATOR, medium_orchestrator_llm)
    if config.deeper_orchestrator_llm:
        deeper_orchestrator_llm = await builder.get_llm(
            config.deeper_orchestrator_llm,
            wrapper_type=LLMFrameworkEnum.LANGCHAIN,
        )
        provider.configure(LLMRole.DEEPER_ORCHESTRATOR, deeper_orchestrator_llm)
    if config.deeper_planner_llm:
        deeper_planner_llm = await builder.get_llm(config.deeper_planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.DEEPER_PLANNER, deeper_planner_llm)
    if config.deeper_researcher_llm:
        deeper_researcher_llm = await builder.get_llm(
            config.deeper_researcher_llm,
            wrapper_type=LLMFrameworkEnum.LANGCHAIN,
        )
        provider.configure(LLMRole.DEEPER_RESEARCHER, deeper_researcher_llm)
    if config.deep_planner_llm:
        deep_planner_llm = await builder.get_llm(config.deep_planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.DEEP_PLANNER, deep_planner_llm)
    if config.deep_researcher_llm:
        deep_researcher_llm = await builder.get_llm(
            config.deep_researcher_llm,
            wrapper_type=LLMFrameworkEnum.LANGCHAIN,
        )
        provider.configure(LLMRole.DEEP_RESEARCHER, deep_researcher_llm)
    if config.researcher_llm:
        researcher_llm = await builder.get_llm(config.researcher_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.RESEARCHER, researcher_llm)
    if config.planner_llm:
        planner_llm = await builder.get_llm(config.planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        provider.configure(LLMRole.PLANNER, planner_llm)

    verifier_llm = None
    if config.verifier_llm:
        try:
            verifier_llm = await builder.get_llm(config.verifier_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        except Exception:  # noqa: BLE001 - verifier is optional; fall back to researcher LLM
            logger.warning(
                "Verifier LLM %r unavailable; adversarial verifier will fall back to the researcher LLM",
                config.verifier_llm,
                exc_info=True,
            )
            verifier_llm = None

    verbose = is_verbose(config.verbose)
    callbacks = [VerboseTraceCallback()] if verbose else []

    agent = DeepResearcherAgent(
        llm_provider=provider,
        tools=tools,
        max_loops=config.max_loops,
        verbose=verbose,
        callbacks=callbacks,
        skills=config.skills,
        sandbox=config.sandbox,
        verifier_llm=verifier_llm,
    )

    async def _run(state: DeepResearchAgentState) -> DeepResearchAgentState:
        """Run deep research with a list of messages or payload."""
        try:
            data_sources = state.data_sources
            selected_tools = filter_tools_by_sources(tools, data_sources)
            active_agent = agent
            if config.sandbox is not None or (data_sources is not None and selected_tools != tools):
                # Scope the Modal sandbox to the async job_id when one is in
                # NAT context (set by aiq_api/jobs/runner.py). Falls back to a
                # per-request uuid in DeepAgentsRuntime when None.
                job_id: str | None = None
                try:
                    from nat.builder.context import Context

                    job_id = Context.get().workflow_run_id
                except Exception:  # noqa: BLE001 - Context may be unavailable in sync/eval paths
                    job_id = None
                active_agent = DeepResearcherAgent(
                    llm_provider=provider,
                    tools=selected_tools,
                    max_loops=config.max_loops,
                    verbose=verbose,
                    callbacks=callbacks,
                    skills=config.skills,
                    sandbox=config.sandbox,
                    job_id=job_id,
                    verifier_llm=verifier_llm,
                )
            elif data_sources is not None and not selected_tools:
                logger.warning("Deep research received data_sources with no matching tools")

            # Validate tool availability before starting deep research
            # At least one tool must be available
            # This prevents the agent from trying to reason about unavailable tools
            # Check selected_tools directly - they already reflect data_sources filtering
            from aiq_agent.common import format_user_facing_tool_error
            from aiq_agent.common import validate_tool_availability

            is_valid, _, unavailable_tools = validate_tool_availability(selected_tools, research_type="deep research")

            # Fail if no tools are available
            if not is_valid:
                error_msg = format_user_facing_tool_error("deep research", unavailable_tools)

                # Return error state with error message - this prevents the agent from running
                from langchain_core.messages import AIMessage

                error_state = DeepResearchAgentState(messages=state.messages + [AIMessage(content=error_msg)])
                return error_state

            result = await active_agent.run(state)
            return result
        except Exception:
            logger.exception("Error in deep research execution")
            raise

    yield FunctionInfo.from_fn(_run, description="Deep research agent for comprehensive multi-phase research.")


########################################################
# Deep Research Workflow (Wrapper for Evaluation)
########################################################
class DeepResearchWorkflowConfig(FunctionBaseConfig, name="deep_research_workflow"):
    """Configuration for the deep research workflow wrapper.

    This wrapper accepts a string query and converts it to messages
    for the deep_research_agent. Use this as the workflow for evaluation.
    """

    pass


@register_function(config_type=DeepResearchWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def deep_research_workflow(config: DeepResearchWorkflowConfig, builder: Builder):
    """Wrapper workflow that accepts string queries for evaluation."""
    deep_research_agent_fn = await builder.get_function("deep_research_agent")
    workflow_id = config.name or config.type

    async def _run(query: str) -> ChatResponse:
        """Run deep research on a query string."""
        state = DeepResearchAgentState(messages=[HumanMessage(content=query)])
        result = await deep_research_agent_fn.ainvoke(state)
        messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
        response_content = ""
        if messages:
            last_msg = messages[-1]
            response_content = getattr(last_msg, "content", "")
            if not response_content and isinstance(last_msg, dict):
                response_content = str(last_msg.get("content", ""))
        return _create_chat_response(response_content, response_id="research_response", model=workflow_id)

    yield FunctionInfo.from_fn(_run, description="Deep research workflow for evaluation (accepts string query).")

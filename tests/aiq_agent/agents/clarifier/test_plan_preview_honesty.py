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

"""Plan preview honesty tests.

The approval UI must show the real generated plan. When plan generation fails
and only the generic safety-net fallback is left, the user sees an honest
"plan preview unavailable" message and NO fabricated plan is bound to the run.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.clarifier.agent import ClarifierAgent
from aiq_agent.agents.clarifier.models import ClarificationResponse
from aiq_agent.agents.clarifier.models import ClarifierAgentState
from aiq_agent.common import LLMProvider

VAGUE_QUERY = "what are hermes agent power-user workflows?"

COMPLETE_JSON = ClarificationResponse(needs_clarification=False, clarification_question=None).model_dump_json()


def _mock_provider() -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=COMPLETE_JSON))
    llm.bind_tools = MagicMock(return_value=llm)
    provider = MagicMock(spec=LLMProvider)
    provider.get = MagicMock(return_value=llm)
    return provider


def test_vague_query_yields_fallback_contract():
    """Precondition for these tests: the query has no richer query-derived contract."""
    _title, _sections, source = ClarifierAgent._plan_contract_from_query(VAGUE_QUERY)
    assert source == "fallback"


@pytest.mark.asyncio
async def test_planner_failure_shows_honest_unavailable_message_and_binds_no_fake_plan():
    """Planner never returns valid JSON -> honest message, no fabricated approved plan."""
    planner_llm = MagicMock()
    planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content="I could not produce a plan, sorry."))

    displayed: list[str] = []

    async def user_callback(prompt: str) -> str:
        displayed.append(prompt)
        return "approve"

    agent = ClarifierAgent(
        llm_provider=_mock_provider(),
        user_prompt_callback=user_callback,
        enable_plan_approval=True,
        planner_llm=planner_llm,
        max_plan_iterations=1,
    )

    state = ClarifierAgentState(messages=[HumanMessage(content=VAGUE_QUERY)])
    result = await agent.run(state)

    assert len(displayed) == 1
    assert "Plan preview unavailable" in displayed[0]
    # The fake fallback sections must not be presented as a real plan
    assert "Evidence and Competing Views" not in displayed[0]

    assert result.plan_approved is True
    assert result.plan_title is None
    assert result.plan_sections == []
    # No fabricated "Approved Research Plan" context is attached downstream
    assert result.get_approved_plan_context() is None


@pytest.mark.asyncio
async def test_real_generated_plan_is_shown_and_bound():
    """A valid planner response is shown verbatim and bound to the approved run."""
    plan_json = (
        '{"title": "Hermes Agent Power-User Workflows", '
        '"sections": ["Hermes Agent Capability Map", "Power-User Automation Recipes", '
        '"Integration and Scripting Patterns", "Limitations and Workarounds"]}'
    )
    planner_llm = MagicMock()
    planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content=plan_json))

    displayed: list[str] = []

    async def user_callback(prompt: str) -> str:
        displayed.append(prompt)
        return "approve"

    agent = ClarifierAgent(
        llm_provider=_mock_provider(),
        user_prompt_callback=user_callback,
        enable_plan_approval=True,
        planner_llm=planner_llm,
        max_plan_iterations=1,
    )

    state = ClarifierAgentState(messages=[HumanMessage(content=VAGUE_QUERY)])
    result = await agent.run(state)

    assert len(displayed) == 1
    assert "Plan preview unavailable" not in displayed[0]
    assert "Hermes Agent Power-User Workflows" in displayed[0]
    assert "Power-User Automation Recipes" in displayed[0]

    assert result.plan_approved is True
    assert result.plan_title == "Hermes Agent Power-User Workflows"
    assert result.plan_sections == [
        "Hermes Agent Capability Map",
        "Power-User Automation Recipes",
        "Integration and Scripting Patterns",
        "Limitations and Workarounds",
    ]
    context = result.get_approved_plan_context()
    assert context is not None
    assert "Hermes Agent Power-User Workflows" in context


@pytest.mark.asyncio
async def test_planner_failure_auto_approve_at_max_iterations_binds_no_fake_plan():
    """Feedback loops that exhaust max iterations on a fallback never bind a fake plan."""
    planner_llm = MagicMock()
    planner_llm.ainvoke = AsyncMock(return_value=AIMessage(content="still not json"))

    responses = iter(["please make it better"])

    async def user_callback(_prompt: str) -> str:
        return next(responses, "make it even better")

    agent = ClarifierAgent(
        llm_provider=_mock_provider(),
        user_prompt_callback=user_callback,
        enable_plan_approval=True,
        planner_llm=planner_llm,
        max_plan_iterations=1,
    )

    state = ClarifierAgentState(messages=[HumanMessage(content=VAGUE_QUERY)])
    result = await agent.run(state)

    assert result.plan_approved is True
    assert result.plan_title is None
    assert result.plan_sections == []
    assert result.get_approved_plan_context() is None

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

"""Approved-plan context must be bound to the query it was generated for.

Regression tests for the defect where a checkpointed clarifier_result (containing
a previous query's **Approved Research Plan**) leaked into a NEW query's deep
research job on the same conversation thread.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from aiq_agent.agents.chat_researcher.agent import ChatResearcherAgent
from aiq_agent.agents.chat_researcher.models import DepthDecision
from aiq_agent.agents.chat_researcher.models import IntentResult
from aiq_agent.agents.clarifier.models import ClarifierResult


async def _deep_orchestration(_state):
    return {
        "user_intent": IntentResult(intent="research", raw=None),
        "depth_decision": DepthDecision(decision="deep", raw_reasoning="Complex"),
    }


async def _shallow_stub(_state):
    raise AssertionError("shallow research should not run in these tests")


def _make_agent(clarifier_fn, deep_fn, *, checkpointer):
    return ChatResearcherAgent(
        intent_classifier_fn=_deep_orchestration,
        shallow_research_fn=_shallow_stub,
        deep_research_fn=deep_fn,
        clarifier_fn=clarifier_fn,
        enable_clarifier=True,
        checkpointer=checkpointer,
    )


class _DeepCapture:
    """Deep research stub that records the clarifier_result it receives."""

    def __init__(self):
        self.clarifier_results = []

    async def __call__(self, state):
        self.clarifier_results.append(state.clarifier_result)

        class _Result:
            messages = [AIMessage(content="report done")]

        return _Result()


@pytest.mark.asyncio
async def test_stale_approved_plan_does_not_leak_when_clarifier_skipped():
    """Turn 1 approves a plan; turn 2 (clarifier skipped) must NOT reuse it."""
    deep = _DeepCapture()

    async def clarifier_fn(_state):
        return ClarifierResult(
            clarifier_log="clarified",
            plan_title="Building AI-Powered 20 Min to 8 Min",
            plan_sections=["Old Section A", "Old Section B"],
            plan_approved=True,
        )

    agent = _make_agent(clarifier_fn, deep, checkpointer=MemorySaver())

    # Turn 1: clarifier runs, plan approved and attached.
    state1 = {
        "messages": [HumanMessage(content="Build an AI workflow that cuts prep from 20 min to 8 min")],
        "force_deep_research": True,
        "skip_clarifier": False,
    }
    await agent.run(state1, thread_id="thread-1")
    assert deep.clarifier_results[0] is not None
    assert "Building AI-Powered 20 Min to 8 Min" in deep.clarifier_results[0]

    # Turn 2 on the SAME thread: clarifier skipped -> no stale plan may leak.
    state2 = {
        "messages": [HumanMessage(content="Hermes Agent power-user workflows")],
        "force_deep_research": True,
        "skip_clarifier": True,
    }
    await agent.run(state2, thread_id="thread-1")

    assert deep.clarifier_results[1] is None


@pytest.mark.asyncio
async def test_turn_boundary_resets_checkpointed_clarifier_result():
    """A new turn never starts with the previous turn's clarifier_result."""
    deep = _DeepCapture()

    call_count = {"n": 0}

    async def clarifier_fn(_state):
        call_count["n"] += 1
        return ClarifierResult(
            clarifier_log=f"clarified-{call_count['n']}",
            plan_title=f"Plan {call_count['n']}",
            plan_sections=[f"Section {call_count['n']}"],
            plan_approved=True,
        )

    agent = _make_agent(clarifier_fn, deep, checkpointer=MemorySaver())

    for turn in (1, 2):
        await agent.run(
            {
                "messages": [HumanMessage(content=f"query {turn}")],
                "force_deep_research": True,
                "skip_clarifier": False,
            },
            thread_id="thread-2",
        )

    # Each turn's deep research only sees the plan generated for that turn.
    assert "Plan 1" in deep.clarifier_results[0]
    assert "Plan 2" not in deep.clarifier_results[0]
    assert "Plan 2" in deep.clarifier_results[1]
    assert "Plan 1" not in deep.clarifier_results[1]


@pytest.mark.asyncio
async def test_pydantic_state_input_also_resets_clarifier_result():
    """The ChatResearcherState (non-dict) input path resets clarifier_result too."""
    from aiq_agent.agents.chat_researcher.models import ChatResearcherState

    deep = _DeepCapture()

    async def clarifier_fn(_state):
        return ClarifierResult(
            clarifier_log="clarified",
            plan_title="First Plan",
            plan_sections=["S1"],
            plan_approved=True,
        )

    agent = _make_agent(clarifier_fn, deep, checkpointer=MemorySaver())

    state1 = ChatResearcherState(
        messages=[HumanMessage(content="first query")],
        force_deep_research=True,
        skip_clarifier=False,
    )
    await agent.run(state1, thread_id="thread-3")
    assert "First Plan" in deep.clarifier_results[0]

    state2 = ChatResearcherState(
        messages=[HumanMessage(content="second query")],
        force_deep_research=True,
        skip_clarifier=True,
    )
    await agent.run(state2, thread_id="thread-3")

    assert deep.clarifier_results[1] is None

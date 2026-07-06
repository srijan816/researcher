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

"""Tests for chat researcher image settings submission."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_agent.agents.chat_researcher.models import DepthDecision
from aiq_agent.agents.chat_researcher.models import IntentResult
from aiq_agent.agents.chat_researcher.register import ChatDeepResearcherConfig
from aiq_agent.agents.chat_researcher.register import chat_deepresearcher_agent


class _FakeContext:
    @staticmethod
    def get():
        return SimpleNamespace(conversation_id="test-thread")


class _FakeFunction:
    def __init__(self, fn):
        self.ainvoke = fn


class _FakeBuilder:
    async def get_function(self, name):
        async def intent_classifier(_state):
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision="shallow", raw_reasoning="test"),
            }

        async def unused_agent(_state):
            raise AssertionError(f"{name} should not run when async deep research is enabled")

        if name == "intent_classifier":
            return _FakeFunction(intent_classifier)
        return _FakeFunction(unused_agent)

    def get_function_config(self, _name):
        return SimpleNamespace(tools=[], exclude_tools=[])

    async def get_tools(self, *, tool_names, wrapper_type):
        del tool_names, wrapper_type
        tool = MagicMock()
        tool.name = "web_search"
        tool.description = "Search the web"
        return [tool]


async def _run_chat_submit(monkeypatch, payload):
    submitted_kwargs = {}

    async def fake_submit_agent_job(**kwargs):
        submitted_kwargs.update(kwargs)
        return "job-123"

    monkeypatch.setenv("NAT_DASK_SCHEDULER_ADDRESS", "tcp://scheduler:8786")
    monkeypatch.setattr(
        "aiq_agent.agents.chat_researcher.register.get_checkpointer",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr("aiq_agent.agents.chat_researcher.register.Context", _FakeContext)
    monkeypatch.setattr("aiq_api.jobs.submit.submit_agent_job", fake_submit_agent_job)
    monkeypatch.setattr("aiq_agent.auth.get_current_principal", lambda: None)

    config = ChatDeepResearcherConfig(
        enable_clarifier=False,
        use_async_deep_research=True,
    )
    async with chat_deepresearcher_agent(config, _FakeBuilder()) as function_info:
        await function_info.single_fn(payload)

    return submitted_kwargs


@pytest.mark.asyncio
async def test_chat_payload_image_settings_reach_submit_agent_job(monkeypatch):
    payload = {
        "content": {
            "messages": [
                {
                    "role": "user",
                    "content": "Research AI image generation cost and quality.",
                }
            ],
        },
        "force_deep_research": True,
        "research_depth": "deeper",
        "include_images": True,
        "image_count": 2,
    }

    submitted_kwargs = await _run_chat_submit(monkeypatch, payload)

    assert submitted_kwargs["include_images"] is True
    assert submitted_kwargs["image_count"] == 2


@pytest.mark.asyncio
async def test_chat_payload_missing_image_settings_default_in_submit_agent_job(monkeypatch):
    payload = {
        "content": {
            "messages": [
                {
                    "role": "user",
                    "content": "Research AI image generation cost and quality.",
                }
            ],
        },
        "force_deep_research": True,
        "research_depth": "deeper",
    }

    submitted_kwargs = await _run_chat_submit(monkeypatch, payload)

    assert submitted_kwargs["include_images"] is False
    assert submitted_kwargs["image_count"] is None

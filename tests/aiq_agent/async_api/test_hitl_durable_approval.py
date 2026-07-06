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

"""Tests for durable plan-approval HITL: persistence, reconnect re-delivery, and timeout."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from aiq_api import websocket_reconnect
from aiq_api.websocket_reconnect import HITL_RESPONSE_TIMEOUT_SECONDS
from aiq_api.websocket_reconnect import ReconnectableWebSocketMessageHandler
from aiq_api.websocket_reconnect import WebSocketSessionRegistry
from aiq_api.websocket_reconnect import get_hitl_response_timeout_seconds
from nat.data_models.api_server import TextContent
from nat.data_models.interactive import HumanPromptText
from nat.data_models.interactive import InteractionPrompt


class DummySocket:
    """Minimal websocket stand-in for handler testing."""

    def __init__(self, query_params: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.scope = {"headers": [], "type": "websocket"}
        self.query_params = query_params or {}

    async def receive_json(self) -> dict:
        raise WebSocketDisconnect(1000)

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class DummySessionManager:
    def get_workflow_single_output_schema(self):
        return None

    def get_workflow_streaming_output_schema(self):
        return None


class DummyStepAdaptor:
    pass


class DummyWorker:
    def set_conversation_handler(self, _conversation_id: str, _handler: object) -> None:
        return None

    def get_conversation_handler(self, _conversation_id: str) -> object | None:
        return None

    def remove_conversation_handler(self, _conversation_id: str) -> None:
        return None


class DummyPromptMessage(BaseModel):
    kind: str = "pending_prompt"
    text: str = "Approve the research plan?"


def _make_handler(socket: DummySocket) -> ReconnectableWebSocketMessageHandler:
    return ReconnectableWebSocketMessageHandler(
        socket=socket,
        session_manager=DummySessionManager(),
        step_adaptor=DummyStepAdaptor(),
        worker=DummyWorker(),
    )


class TestApprovalTimeoutConfig:
    """AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS controls the HITL auto-proceed timer."""

    def test_default_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS", raising=False)
        assert get_hitl_response_timeout_seconds() == HITL_RESPONSE_TIMEOUT_SECONDS == 300

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS", "42.5")
        assert get_hitl_response_timeout_seconds() == 42.5

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS", "not-a-number")
        assert get_hitl_response_timeout_seconds() == HITL_RESPONSE_TIMEOUT_SECONDS

    def test_negative_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS", "-5")
        assert get_hitl_response_timeout_seconds() == HITL_RESPONSE_TIMEOUT_SECONDS


class TestPendingPromptRegistry:
    """Pending HITL prompts are persisted server-side keyed by conversation."""

    @pytest.mark.asyncio
    async def test_pending_prompt_returned_only_while_interaction_pending(self):
        registry = WebSocketSessionRegistry()
        prompt = DummyPromptMessage()
        future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()

        await registry.set_pending_prompt("conv-1", prompt)
        # No pending interaction future yet -> prompt is not considered live
        assert await registry.get_pending_prompt("conv-1") is None

        await registry.register_pending_interaction("conv-1", future)
        assert await registry.get_pending_prompt("conv-1") is prompt

        # Once resolved, the prompt is no longer pending
        future.set_result(TextContent(text="approve"))
        assert await registry.get_pending_prompt("conv-1") is None

    @pytest.mark.asyncio
    async def test_clear_pending_interaction_also_clears_prompt(self):
        registry = WebSocketSessionRegistry()
        future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()
        await registry.register_pending_interaction("conv-1", future)
        await registry.set_pending_prompt("conv-1", DummyPromptMessage())

        await registry.clear_pending_interaction("conv-1")

        assert await registry.get_pending_prompt("conv-1") is None

    @pytest.mark.asyncio
    async def test_get_pending_prompt_handles_missing_conversation(self):
        registry = WebSocketSessionRegistry()
        assert await registry.get_pending_prompt(None) is None
        assert await registry.get_pending_prompt("missing") is None


class TestReconnectReattach:
    """Reconnecting sockets re-attach to the conversation and get the pending prompt back."""

    @pytest.mark.asyncio
    async def test_restore_registers_socket_and_resends_pending_prompt(self, monkeypatch):
        registry = WebSocketSessionRegistry()
        monkeypatch.setattr(websocket_reconnect, "_registry", registry)

        prompt = DummyPromptMessage()
        future: asyncio.Future[TextContent] = asyncio.get_running_loop().create_future()
        await registry.register_pending_interaction("conv-1", future)
        await registry.set_pending_prompt("conv-1", prompt)

        socket = DummySocket(query_params={"conversation_id": "conv-1"})
        handler = _make_handler(socket)

        await handler._restore_execution_state()

        # Pending approval prompt re-delivered to the reconnected client
        assert socket.sent == [prompt.model_dump()]
        # Conversation is re-attached for interaction-response routing
        assert handler._conversation_id == "conv-1"
        # Subsequent workflow output flows to the new socket
        assert await registry.send("conv-1", DummyPromptMessage(text="next")) is True
        assert len(socket.sent) == 2

    @pytest.mark.asyncio
    async def test_restore_without_pending_prompt_only_reattaches(self, monkeypatch):
        registry = WebSocketSessionRegistry()
        monkeypatch.setattr(websocket_reconnect, "_registry", registry)

        socket = DummySocket(query_params={"conversation_id": "conv-2"})
        handler = _make_handler(socket)

        await handler._restore_execution_state()

        assert socket.sent == []
        assert await registry.send("conv-2", DummyPromptMessage()) is True

    @pytest.mark.asyncio
    async def test_restore_without_conversation_id_is_noop(self, monkeypatch):
        registry = WebSocketSessionRegistry()
        monkeypatch.setattr(websocket_reconnect, "_registry", registry)

        socket = DummySocket(query_params={})
        handler = _make_handler(socket)

        await handler._restore_execution_state()

        assert socket.sent == []
        assert handler._conversation_id is None


class TestApprovalTimeoutAutoProceed:
    """No approve/reject within the timeout -> proceed exactly as if approved."""

    @pytest.mark.asyncio
    async def test_timeout_resolves_with_skip_and_clears_pending_state(self, monkeypatch):
        registry = WebSocketSessionRegistry()
        monkeypatch.setattr(websocket_reconnect, "_registry", registry)
        monkeypatch.setenv("AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS", "0.01")

        socket = DummySocket()
        handler = _make_handler(socket)
        handler._conversation_id = "conv-1"

        sent_prompts: list[object] = []

        async def fake_create_websocket_message(**kwargs):
            sent_prompts.append(kwargs.get("data_model"))

        converted: list[TextContent] = []

        async def fake_convert(text_content, _prompt_content):
            converted.append(text_content)
            return {"resolved": text_content.text}

        monkeypatch.setattr(handler, "create_websocket_message", fake_create_websocket_message)
        monkeypatch.setattr(
            handler._message_validator,
            "convert_text_content_to_human_response",
            fake_convert,
        )

        prompt = InteractionPrompt(
            id="interaction-1",
            timestamp="2026-01-01T00:00:00Z",
            content=HumanPromptText(text="Approve the plan?", required=True, placeholder="..."),
        )

        # No client response arrives; the server-side timer must fire on its own.
        result = await handler.human_interaction_callback(prompt)

        assert result == {"resolved": "skip"}
        assert converted and converted[0].text == "skip"
        # Pending interaction and prompt are cleaned up after resolution
        assert await registry.get_pending_prompt("conv-1") is None
        assert await registry.resolve_pending_interaction("conv-1", TextContent(text="late")) is False

    @pytest.mark.asyncio
    async def test_reconnected_client_response_resolves_wait_before_timeout(self, monkeypatch):
        registry = WebSocketSessionRegistry()
        monkeypatch.setattr(websocket_reconnect, "_registry", registry)
        monkeypatch.setenv("AIQ_PLAN_APPROVAL_TIMEOUT_SECONDS", "5")

        socket = DummySocket()
        handler = _make_handler(socket)
        handler._conversation_id = "conv-1"

        async def fake_create_websocket_message(**_kwargs):
            return None

        async def fake_convert(text_content, _prompt_content):
            return {"resolved": text_content.text}

        monkeypatch.setattr(handler, "create_websocket_message", fake_create_websocket_message)
        monkeypatch.setattr(
            handler._message_validator,
            "convert_text_content_to_human_response",
            fake_convert,
        )

        prompt = InteractionPrompt(
            id="interaction-2",
            timestamp="2026-01-01T00:00:00Z",
            content=HumanPromptText(text="Approve the plan?", required=True, placeholder="..."),
        )

        wait_task = asyncio.create_task(handler.human_interaction_callback(prompt))
        await asyncio.sleep(0.05)

        # A reconnected handler resolves through the registry (no local future).
        resolved = await registry.resolve_pending_interaction("conv-1", TextContent(text="approve"))
        assert resolved is True

        result = await asyncio.wait_for(wait_task, timeout=2)
        assert result == {"resolved": "approve"}

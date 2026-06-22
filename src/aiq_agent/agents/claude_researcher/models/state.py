# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State model for Claude Code driven research."""

from typing import Annotated
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from aiq_agent.common import DEFAULT_RESEARCH_DEPTH
from aiq_agent.common import ResearchDepthTier


class ClaudeResearchAgentState(BaseModel):
    """Minimal state shape used by the async job runner."""

    messages: Annotated[list[AnyMessage], add_messages]
    data_sources: list[str] | None = None
    research_depth: ResearchDepthTier = DEFAULT_RESEARCH_DEPTH
    user_info: dict[str, Any] | None = None

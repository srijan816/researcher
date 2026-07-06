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

import json
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages

from aiq_agent.common import DEFAULT_RESEARCH_DEPTH
from aiq_agent.common import ResearchDepthTier
from aiq_agent.common import normalize_research_depth
from aiq_agent.common import parse_data_sources


def trim_message_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Trim messages to a maximum number of tokens."""
    return trim_messages(
        messages=[m.model_dump() for m in messages],
        max_tokens=max_tokens,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
    )


def _normalize_enum_value(value: Any) -> str | None:
    """Extract string value from enum or return as-is if already a string."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _is_text_type(type_value: Any) -> bool:
    """Check if type value represents 'text', handling both strings and enums."""
    normalized = _normalize_enum_value(type_value)
    return normalized is not None and normalized.lower() == "text"


def _is_user_role(role_value: Any) -> bool:
    """Check if role value represents 'user', handling both strings and enums."""
    normalized = _normalize_enum_value(role_value)
    return normalized is not None and normalized.lower() == "user"


def _extract_text_from_message(message: Any) -> str | None:
    if message is None:
        return None
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        content_value = getattr(message, "content")
        if isinstance(content_value, str):
            return content_value
        if isinstance(content_value, list):
            parts = []
            for item in content_value:
                if hasattr(item, "type") and _is_text_type(getattr(item, "type")):
                    text_value = getattr(item, "text", None)
                    if text_value:
                        parts.append(str(text_value))
                elif isinstance(item, dict) and _is_text_type(item.get("type")):
                    text_value = item.get("text")
                    if text_value:
                        parts.append(str(text_value))
            if parts:
                return "\n".join(parts).strip()
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and _is_text_type(item.get("type")):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            if parts:
                return "\n".join(parts).strip()
        if isinstance(content, str):
            return content
        text_value = message.get("text")
        if isinstance(text_value, str):
            return text_value
    return None


def coerce_content_text(content: Any) -> str:
    """Convert LangChain/OpenAI/Anthropic text content variants into plain text.

    MiniMax's Anthropic-compatible endpoint returns a list of content blocks
    (for example ``thinking`` plus ``text``). Several orchestration nodes only
    need the user-visible text block for JSON parsing.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("type") in {"thinking", "reasoning"} and item.get("thinking"):
                    continue
            elif hasattr(item, "type") and _is_text_type(getattr(item, "type")):
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts).strip()
    return str(content)


def _parse_force_deep(value: Any) -> bool:
    """Parse truthy force-deep flags from JSON payloads."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _parse_image_count(value: Any) -> int | None:
    """Parse an optional image count constrained to the supported 1-4 range."""
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= count <= 4:
        return count
    return None


def _parse_agent_type(value: Any) -> str:
    """Parse the requested async research agent from JSON payloads."""
    if not isinstance(value, str):
        return "deep_researcher"
    normalized = value.strip().lower()
    if normalized in {"deep_researcher", "claude_researcher"}:
        return normalized
    return "deep_researcher"


def _extract_query_sources_force_depth_from_text(
    text: str,
) -> tuple[str, list[str] | None, bool, ResearchDepthTier, str, bool, int | None]:
    if not text:
        return ("", None, False, DEFAULT_RESEARCH_DEPTH, "deep_researcher", False, None)
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            payload = json.loads(trimmed)
        except json.JSONDecodeError:
            return (text, None, False, DEFAULT_RESEARCH_DEPTH, "deep_researcher", False, None)
        if isinstance(payload, dict):
            data_sources = parse_data_sources(payload.get("data_sources"))
            query_text = payload.get("query") or payload.get("text")
            force_deep = _parse_force_deep(payload.get("force_deep_research"))
            research_depth = normalize_research_depth(payload.get("research_depth"))
            agent_type = _parse_agent_type(payload.get("agent_type"))
            include_images = _parse_force_deep(payload.get("include_images"))
            image_count = _parse_image_count(payload.get("image_count"))
            if isinstance(query_text, str) and query_text.strip():
                return (
                    query_text.strip(),
                    data_sources,
                    force_deep,
                    research_depth,
                    agent_type,
                    include_images,
                    image_count,
                )
    return (text, None, False, DEFAULT_RESEARCH_DEPTH, "deep_researcher", False, None)


def _extract_query_sources_force_from_text(text: str) -> tuple[str, list[str] | None, bool]:
    """Backward-compatible helper returning query, sources, and force-deep."""
    query_text, data_sources, force_deep, _research_depth, _agent_type, _include_images, _image_count = (
        _extract_query_sources_force_depth_from_text(text)
    )
    return (query_text, data_sources, force_deep)


def _extract_query_sources_force_depth(
    payload: Any,
) -> tuple[str, list[str] | None, bool, ResearchDepthTier, str, bool, int | None]:
    """Extract query, sources, force-deep flag, depth tier, async agent type, and image settings."""
    if isinstance(payload, dict):
        content = payload.get("content", {}) if isinstance(payload.get("content"), dict) else {}
        data_sources = parse_data_sources(payload.get("data_sources")) or parse_data_sources(
            content.get("data_sources")
        )
        force_deep = _parse_force_deep(payload.get("force_deep_research")) or _parse_force_deep(
            content.get("force_deep_research")
        )
        research_depth = normalize_research_depth(payload.get("research_depth") or content.get("research_depth"))
        agent_type = _parse_agent_type(payload.get("agent_type") or content.get("agent_type"))
        include_images = _parse_force_deep(payload.get("include_images")) or _parse_force_deep(
            content.get("include_images")
        )
        image_count = _parse_image_count(payload.get("image_count")) or _parse_image_count(content.get("image_count"))
        messages = content.get("messages", [])
        query_text = None
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and _is_user_role(msg.get("role")):
                    query_text = _extract_text_from_message(msg)
                    if query_text:
                        break
            if not query_text:
                query_text = _extract_text_from_message(messages[-1])
        if not query_text:
            query_text = _extract_text_from_message(payload.get("message")) or _extract_text_from_message(
                payload.get("text")
            )
        if query_text:
            (
                inline_query,
                inline_sources,
                inline_force_deep,
                inline_depth,
                inline_agent_type,
                inline_include_images,
                inline_image_count,
            ) = _extract_query_sources_force_depth_from_text(query_text)
            query_text = inline_query
            data_sources = data_sources or inline_sources
            force_deep = force_deep or inline_force_deep
            research_depth = inline_depth if inline_depth != DEFAULT_RESEARCH_DEPTH else research_depth
            agent_type = inline_agent_type if inline_agent_type != "deep_researcher" else agent_type
            include_images = include_images or inline_include_images
            image_count = image_count or inline_image_count
        return (query_text or "", data_sources, force_deep, research_depth, agent_type, include_images, image_count)

    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        data_sources = parse_data_sources(getattr(payload, "data_sources", None))
        force_deep = _parse_force_deep(getattr(payload, "force_deep_research", None))
        research_depth = normalize_research_depth(getattr(payload, "research_depth", None))
        agent_type = _parse_agent_type(getattr(payload, "agent_type", None))
        include_images = _parse_force_deep(getattr(payload, "include_images", None))
        image_count = _parse_image_count(getattr(payload, "image_count", None))
        query_text = None
        for msg in reversed(messages):
            if _is_user_role(getattr(msg, "role", None)):
                query_text = _extract_text_from_message(msg)
                if query_text:
                    break
        if not query_text and messages:
            query_text = _extract_text_from_message(messages[-1])
        if query_text:
            (
                inline_query,
                inline_sources,
                inline_force_deep,
                inline_depth,
                inline_agent_type,
                inline_include_images,
                inline_image_count,
            ) = _extract_query_sources_force_depth_from_text(query_text)
            query_text = inline_query
            data_sources = data_sources or inline_sources
            force_deep = force_deep or inline_force_deep
            research_depth = inline_depth if inline_depth != DEFAULT_RESEARCH_DEPTH else research_depth
            agent_type = inline_agent_type if inline_agent_type != "deep_researcher" else agent_type
            include_images = include_images or inline_include_images
            image_count = image_count or inline_image_count
        return (query_text or "", data_sources, force_deep, research_depth, agent_type, include_images, image_count)

    query_text = str(payload)
    return _extract_query_sources_force_depth_from_text(query_text)


def _extract_query_sources_and_force_deep(payload: Any) -> tuple[str, list[str] | None, bool]:
    """Extract query text and data sources from various payload formats.

    Returns:
        Tuple of (query_text, data_sources, force_deep_research).
        - data_sources is None if not specified, meaning use all configured tools
        - data_sources is a list if explicitly specified (use only those)
    """
    query_text, data_sources, force_deep, _research_depth, _agent_type, _include_images, _image_count = (
        _extract_query_sources_force_depth(payload)
    )
    return (query_text, data_sources, force_deep)


def _extract_query_from_text(text: str) -> tuple[str, list[str] | None]:
    """Backward-compatible helper returning only query text and data sources."""
    query_text, data_sources, _force_deep = _extract_query_sources_force_from_text(text)
    return (query_text, data_sources)


def _extract_query_and_sources(payload: Any) -> tuple[str, list[str] | None]:
    """Backward-compatible helper returning only query text and data sources."""
    query_text, data_sources, _force_deep = _extract_query_sources_and_force_deep(payload)
    return (query_text, data_sources)

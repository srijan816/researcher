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

"""Custom middleware for the deep research agent."""

import asyncio
import contextvars
import hashlib
import json
import logging
import re
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage

from aiq_agent.common import get_source_id_for_tool
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.citation_verification import extract_sources_from_tool_result
from aiq_agent.common.source_classification import source_class_rank

logger = logging.getLogger(__name__)

_session_tool_counts: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_deep_research_session_tool_counts", default=None
)
_session_tool_limits: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_deep_research_session_tool_limits", default=None
)
_session_parallel_tool_limits: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_deep_research_session_parallel_tool_limits", default=None
)
_session_exhausted_tools: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "_deep_research_session_exhausted_tools", default=None
)
_session_plan_validation_failures: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_deep_research_session_plan_validation_failures", default=None
)
_session_planner_model_turns: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_deep_research_session_planner_model_turns", default=None
)
_session_recent_artifact_writes: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_deep_research_session_recent_artifact_writes", default=None
)
_session_task_search_counts: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_deep_research_session_task_search_counts", default=None
)

_SEARCH_TOOL_FAMILY = {"advanced_web_search_tool", "web_search_tool", "exa_web_search_tool"}


def set_session_tool_counts(counts: dict[str, int] | None) -> contextvars.Token:
    """Set per-run tool counts for deep research middleware."""
    return _session_tool_counts.set(counts)


def reset_session_tool_counts(token: contextvars.Token) -> None:
    """Restore previous per-run tool counts."""
    _session_tool_counts.reset(token)


def set_session_tool_limits(limits: dict[str, int] | None) -> contextvars.Token:
    """Set per-run tool limits for deep research middleware."""
    return _session_tool_limits.set(limits)


def reset_session_tool_limits(token: contextvars.Token) -> None:
    """Restore previous per-run tool limits."""
    _session_tool_limits.reset(token)


def set_session_parallel_tool_limits(limits: dict[str, int] | None) -> contextvars.Token:
    """Set per-response parallel tool limits for deep research middleware."""
    return _session_parallel_tool_limits.set(limits)


def reset_session_parallel_tool_limits(token: contextvars.Token) -> None:
    """Restore previous per-response parallel tool limits."""
    _session_parallel_tool_limits.reset(token)


def set_session_exhausted_tools(tools: set[str] | None) -> contextvars.Token:
    """Set per-run exhausted tool names for deep research middleware."""
    return _session_exhausted_tools.set(tools)


def reset_session_exhausted_tools(token: contextvars.Token) -> None:
    """Restore previous per-run exhausted tool names."""
    _session_exhausted_tools.reset(token)


def set_session_plan_validation_failures(count: int | None) -> contextvars.Token:
    """Set the per-run plan validation failure counter."""
    return _session_plan_validation_failures.set(count)


def reset_session_plan_validation_failures(token: contextvars.Token) -> None:
    """Restore the previous plan validation failure counter."""
    _session_plan_validation_failures.reset(token)


def set_session_planner_model_turns(count: int | None) -> contextvars.Token:
    """Set the per-run planner model turn counter."""
    return _session_planner_model_turns.set(count)


def reset_session_planner_model_turns(token: contextvars.Token) -> None:
    """Restore the previous planner model turn counter."""
    _session_planner_model_turns.reset(token)


def set_session_recent_artifact_writes(writes: dict[str, int] | None) -> contextvars.Token:
    """Set per-run recent artifact write tracking for readback suppression."""
    return _session_recent_artifact_writes.set(writes)


def reset_session_recent_artifact_writes(token: contextvars.Token) -> None:
    """Restore previous recent artifact write tracking."""
    _session_recent_artifact_writes.reset(token)


def set_session_task_search_counts(counts: dict[str, int] | None) -> contextvars.Token:
    """Set per-researcher-task search counts."""
    return _session_task_search_counts.set(counts)


def reset_session_task_search_counts(token: contextvars.Token) -> None:
    """Restore previous per-researcher-task search counts."""
    _session_task_search_counts.reset(token)


def _budget_key(tool_name: str, scope: str | None = None) -> str:
    """Return the counter key for a tool, optionally scoped to an agent role."""
    family = _budget_family(tool_name)
    return f"{scope}:{family}" if scope else family


def _budget_family(tool_name: str) -> str:
    """Collapse related search tools into one shared budget family."""
    if tool_name in _SEARCH_TOOL_FAMILY:
        return "search"
    return tool_name


def _mark_tool_exhausted(tool_name: str, scope: str | None = None) -> None:
    exhausted = _session_exhausted_tools.get()
    if exhausted is None:
        exhausted = set()
        _session_exhausted_tools.set(exhausted)
    exhausted.add(_budget_key(tool_name, scope))


def _scoped_limit(active_limits: dict[str, int], tool_name: str, scope: str | None = None) -> int | None:
    """Resolve a scoped budget limit, falling back to the unscoped tool limit."""
    family = _budget_family(tool_name)
    scoped = active_limits.get(_budget_key(family, scope))
    if scoped is not None:
        return scoped
    scoped = active_limits.get(_budget_key(tool_name, scope))
    if scoped is not None:
        return scoped
    family_limit = active_limits.get(family)
    if family_limit is not None:
        return family_limit
    return active_limits.get(tool_name)


def get_session_budget_snapshot() -> dict[str, object]:
    """Return a JSON-serializable view of current per-run tool budgets."""

    limits = dict(_session_tool_limits.get() or {})
    counts = dict(_session_tool_counts.get() or {})
    exhausted = sorted(_session_exhausted_tools.get() or set())
    entries: list[dict[str, object]] = []
    for key in sorted(set(limits) | set(counts)):
        limit = limits.get(key)
        used = counts.get(key, 0)
        remaining = None if limit is None else max(0, int(limit) - int(used))
        share_used = None if not limit else min(1.0, round(int(used) / max(1, int(limit)), 3))
        entries.append(
            {
                "key": key,
                "used": used,
                "limit": limit,
                "remaining": remaining,
                "share_used": share_used,
                "exhausted": key in exhausted,
            }
        )
    return {
        "budgets": entries,
        "task_search_counts": dict(_session_task_search_counts.get() or {}),
        "exhausted": exhausted,
        "planner_model_turns": _session_planner_model_turns.get() or 0,
        "plan_validation_failures": _session_plan_validation_failures.get() or 0,
    }


# Path to this agent's prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"


class EmptyContentFixMiddleware(AgentMiddleware):
    """
    Middleware that fixes empty ToolMessage content.

    Some LLM APIs (e.g., NVIDIA, OpenAI) reject messages with empty content.
    This middleware ensures all ToolMessages have non-empty content by
    replacing empty strings with a placeholder.
    """

    def __init__(self, placeholder: str = "empty content received."):
        """
        Initialize the middleware.

        Args:
            placeholder: Text to use when ToolMessage content is empty.
        """
        self.placeholder = placeholder

    async def awrap_model_call(self, request, handler):
        """Fix empty ToolMessage content before sending to the model."""
        fixed_messages = []
        for msg in request.messages:
            if isinstance(msg, ToolMessage) and not msg.content:
                # Create a new ToolMessage with placeholder content
                fixed_messages.append(
                    ToolMessage(
                        content=self.placeholder,
                        tool_call_id=msg.tool_call_id,
                        name=getattr(msg, "name", None),
                        id=msg.id,
                    )
                )
            else:
                fixed_messages.append(msg)

        return await handler(request.override(messages=fixed_messages))


class ThinkingOnlyRepairMiddleware(AgentMiddleware):
    """Repair MiniMax turns that contain provider thinking but no action.

    MiniMax's Anthropic-compatible endpoint can occasionally return a content
    payload made only of `thinking`/`signature` blocks. Those blocks are useful
    as private reasoning metadata, but they are not a valid agent step: there is
    no visible text and no tool call for DeepAgents to execute. If accepted, the
    final synthesis stage can end with an empty report even though research
    artifacts exist.
    """

    def __init__(self, max_repairs: int = 2) -> None:
        self.max_repairs = max(0, max_repairs)

    @staticmethod
    def _has_tool_calls(message: AIMessage) -> bool:
        return bool(getattr(message, "tool_calls", None))

    @staticmethod
    def _visible_text_from_blocks(content: list) -> str:
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                if str(block).strip():
                    parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type in {"text", "output_text"} and block.get("text"):
                parts.append(str(block["text"]))
            elif block_type not in {"thinking", "input_json_delta", "tool_use"}:
                value = block.get("content") or block.get("delta")
                if value:
                    parts.append(str(value))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()

    @classmethod
    def _is_thinking_only_message(cls, message: AIMessage) -> bool:
        if cls._has_tool_calls(message):
            return False

        content = message.content
        if isinstance(content, list):
            if cls._visible_text_from_blocks(content):
                return False
            return any(isinstance(block, dict) and block.get("type") == "thinking" for block in content)

        if isinstance(content, str):
            stripped = content.strip()
            return bool(
                stripped
                and (
                    stripped.startswith("[{'thinking'")
                    or stripped.startswith('[{"thinking"')
                    or stripped.startswith("{'thinking'")
                    or stripped.startswith('{"thinking"')
                )
            )

        return False

    @classmethod
    def _needs_repair(cls, response: ModelResponse) -> bool:
        return any(
            isinstance(message, AIMessage) and cls._is_thinking_only_message(message) for message in response.result
        )

    async def awrap_model_call(self, request, handler):
        """Retry with an explicit repair instruction when output has only thinking."""
        response = await handler(request)
        if not self._needs_repair(response):
            return response

        repair_messages = list(request.messages)
        for attempt in range(self.max_repairs):
            logger.warning(
                "MiniMax returned thinking-only content with no text/tool calls; requesting repair (%d/%d)",
                attempt + 1,
                self.max_repairs,
            )
            repair_messages = [
                *repair_messages,
                HumanMessage(
                    content=(
                        "Your previous assistant turn contained only provider reasoning/thinking blocks. "
                        "That is not a valid agent action. Continue from the current state now: either call "
                        "the appropriate tool, or if you are in final synthesis, call write_file with "
                        "file_path='/report.md' and return the final report prose. Do not return thinking-only "
                        "content."
                    )
                ),
            ]
            response = await handler(request.override(messages=repair_messages))
            if not self._needs_repair(response):
                return response

        return response


# Common hallucinated tool name mappings
_TOOL_NAME_ALIASES: dict[str, str] = {
    "open_file": "read_file",
    "find": "grep",
    "find_file": "glob",
}


def _sync_content_tool_use_blocks(
    content,
    kept_tool_calls: list[dict],
    name_map: dict[str, str] | None = None,
    args_map: dict[str, dict] | None = None,
):
    """Keep Anthropic-style content tool_use blocks aligned with AIMessage.tool_calls."""
    if not isinstance(content, list):
        return content

    kept_ids = {tool_call.get("id") for tool_call in kept_tool_calls if tool_call.get("id")}
    synced = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            synced.append(block)
            continue

        block_id = block.get("id")
        if block_id and block_id not in kept_ids:
            continue

        updated_block = dict(block)
        if name_map and block_id in name_map:
            updated_block["name"] = name_map[block_id]
        if args_map and block_id in args_map:
            updated_block["input"] = args_map[block_id]
        synced.append(updated_block)

    return synced


class ToolNameSanitizationMiddleware(AgentMiddleware):
    """
    Middleware that sanitizes corrupted tool names in LLM responses.

    LLMs sometimes generate malformed tool calls with suffixes like
    <|channel|>commentary or .exec, or hallucinate tool names like
    open_file or find. This middleware intercepts the model response
    and fixes tool names before the framework dispatches them.
    """

    def __init__(self, valid_tool_names: list[str]):
        self.valid_tool_names = set(valid_tool_names)

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize a potentially corrupted tool name.

        Returns the cleaned name if it maps to a valid tool,
        otherwise returns the original name unchanged.
        """
        # 1. Strip <|channel|> and everything after
        if "<|channel|>" in name:
            candidate = name.split("<|channel|>", maxsplit=1)[0]
            if candidate in self.valid_tool_names:
                logger.info("Sanitized tool name: '%s' -> '%s'", name, candidate)
                return candidate

        # 2. Strip dot suffix if base name is valid
        if "." in name:
            candidate = name.split(".", maxsplit=1)[0]
            if candidate in self.valid_tool_names:
                logger.info("Sanitized tool name: '%s' -> '%s'", name, candidate)
                return candidate

        # 3. Map common hallucinated names
        if name in _TOOL_NAME_ALIASES:
            mapped = _TOOL_NAME_ALIASES[name]
            if mapped in self.valid_tool_names:
                logger.info("Mapped tool name: '%s' -> '%s'", name, mapped)
                return mapped

        return name

    async def awrap_model_call(self, request, handler):
        """Intercept model response and sanitize tool names."""
        response = await handler(request)

        needs_fix = False
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    sanitized = self._sanitize_tool_name(tc["name"])
                    if sanitized != tc["name"]:
                        needs_fix = True
                        break
                if needs_fix:
                    break

        if not needs_fix:
            return response

        new_result = []
        for msg in response.result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                new_tool_calls = []
                name_map = {}
                for tc in msg.tool_calls:
                    sanitized = self._sanitize_tool_name(tc["name"])
                    new_tool_calls.append({**tc, "name": sanitized})
                    if sanitized != tc["name"] and tc.get("id"):
                        name_map[tc["id"]] = sanitized
                new_msg = AIMessage(
                    content=_sync_content_tool_use_blocks(msg.content, new_tool_calls, name_map),
                    tool_calls=new_tool_calls,
                    id=msg.id,
                )
                new_result.append(new_msg)
            else:
                new_result.append(msg)

        return ModelResponse(result=new_result, structured_response=response.structured_response)


class ToolArgumentNormalizationMiddleware(AgentMiddleware):
    """Repair narrow, known MiniMax structured-tool argument shape mistakes.

    This is intentionally conservative. It only normalizes aliases and simple
    string/list wrappers for tools we own or for built-in filesystem tools with
    stable argument names. Anything ambiguous is left for schema validation so
    real tool-contract bugs stay visible.
    """

    _FILE_TOOLS = {"read_file", "write_file", "edit_file"}

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def _normalize_file_tool_args(cls, args: dict) -> dict:
        normalized = dict(args)
        if "file_path" not in normalized:
            for alias in ("path", "filename", "file"):
                if alias in normalized:
                    normalized["file_path"] = normalized[alias]
                    break
        return normalized

    @classmethod
    def _normalize_plan_query(cls, value) -> dict:
        if isinstance(value, str):
            return {"query": value}
        if not isinstance(value, dict):
            return {"query": str(value)}

        query = dict(value)
        if "query" not in query:
            for alias in ("query_string", "search_query", "question", "topic"):
                if alias in query:
                    query["query"] = query[alias]
                    break
        if "rationale" not in query:
            for alias in ("purpose", "objective", "reason"):
                if alias in query:
                    query["rationale"] = query[alias]
                    break
        if "target_sections" not in query:
            for alias in ("target_section", "sections", "section"):
                if alias in query:
                    query["target_sections"] = cls._as_list(query[alias])
                    break
        if "target_claims" not in query:
            for alias in ("claims", "target_claim", "target_claim_questions"):
                if alias in query:
                    query["target_claims"] = cls._as_list(query[alias])
                    break
        if "target_claim_ids" not in query:
            for alias in ("claim_ids", "target_ids"):
                if alias in query:
                    query["target_claim_ids"] = [str(item) for item in cls._as_list(query[alias])]
                    break
        return query

    @classmethod
    def _normalize_toc_item(cls, value) -> dict:
        if isinstance(value, str):
            return {"title": value}
        if not isinstance(value, dict):
            return {"title": str(value)}
        item = dict(value)
        if "title" not in item:
            for alias in ("heading", "name", "section_title"):
                if alias in item:
                    item["title"] = item[alias]
                    break
        return item

    @classmethod
    def _normalize_write_plan_args(cls, args: dict) -> dict:
        normalized = dict(args)
        alias_map = {
            "report_title": ("title", "research_title"),
            "report_toc": ("toc", "sections", "report_sections", "outline"),
            "queries": ("research_queries", "search_queries", "query_plan"),
            "constraints": ("acceptance_criteria", "requirements", "rules"),
            "task_analysis": ("analysis",),
        }
        for canonical, aliases in alias_map.items():
            if canonical in normalized:
                continue
            for alias in aliases:
                if alias in normalized:
                    normalized[canonical] = normalized[alias]
                    break

        if "report_toc" in normalized:
            normalized["report_toc"] = [
                cls._normalize_toc_item(item) for item in cls._as_list(normalized["report_toc"])
            ]
        if "queries" in normalized:
            normalized["queries"] = [cls._normalize_plan_query(item) for item in cls._as_list(normalized["queries"])]
        if "constraints" in normalized and not isinstance(normalized["constraints"], list):
            normalized["constraints"] = cls._as_list(normalized["constraints"])
        return normalized

    @classmethod
    def _normalize_args(cls, tool_name: str, args) -> dict | None:
        if not isinstance(args, dict):
            return None
        if tool_name == "write_plan":
            return cls._normalize_write_plan_args(args)
        if tool_name in cls._FILE_TOOLS:
            return cls._normalize_file_tool_args(args)
        return args

    async def awrap_model_call(self, request, handler):
        response = await handler(request)

        needs_fix = False
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                normalized_args = self._normalize_args(tc.get("name", ""), tc.get("args"))
                if normalized_args is not None and normalized_args != tc.get("args"):
                    needs_fix = True
                    break
            if needs_fix:
                break
        if not needs_fix:
            return response

        new_result = []
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                new_result.append(msg)
                continue
            new_tool_calls = []
            args_map = {}
            for tc in msg.tool_calls:
                normalized_args = self._normalize_args(tc.get("name", ""), tc.get("args"))
                if normalized_args is None:
                    new_tool_calls.append(tc)
                    continue
                updated = {**tc, "args": normalized_args}
                new_tool_calls.append(updated)
                if normalized_args != tc.get("args") and tc.get("id"):
                    args_map[tc["id"]] = normalized_args
                    logger.info("Normalized arguments for tool %s", tc.get("name", ""))
            new_result.append(
                msg.model_copy(
                    update={
                        "content": _sync_content_tool_use_blocks(msg.content, new_tool_calls, args_map=args_map),
                        "tool_calls": new_tool_calls,
                    }
                )
            )

        return ModelResponse(result=new_result, structured_response=response.structured_response)


class ToolRetryMiddleware(AgentMiddleware):
    """Retries failed tool calls with exponential backoff.

    Provides uniform retry coverage for all tools. Some tools (e.g., Tavily)
    have their own internal retry; this middleware wraps the outer call so
    tools without retry (knowledge layer, paper search) are also covered.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay

    async def awrap_tool_call(self, request, handler):
        """Retry tool calls on failure with exponential backoff."""
        delay = self.initial_delay
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    tool_name = request.tool_call.get("name", "?") if hasattr(request, "tool_call") else "?"
                    logger.warning(
                        "Tool %s failed (attempt %d/%d): %s",
                        tool_name,
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor
        raise last_exception


class PlanFileValidationMiddleware(AgentMiddleware):
    """Reject incomplete planner writes before they poison the workflow state."""

    _PLAN_PATHS = {"/shared/plan.json", "shared/plan.json", "/plan.json", "plan.json"}
    _MAX_REPAIR_ATTEMPTS = 2

    @classmethod
    def _is_plan_write(cls, tool_name: str, args: dict) -> bool:
        if tool_name != "write_file":
            return False
        path = str(args.get("file_path") or args.get("path") or args.get("filename") or "").strip()
        return path in cls._PLAN_PATHS

    @staticmethod
    def _validate_plan_payload(content: str) -> list[str]:
        errors: list[str] = []
        if not content or not content.strip():
            return ["content is empty"]
        if "[... truncated" in content or "[omitted " in content:
            errors.append("content contains a truncation/omission marker")

        try:
            plan = json.loads(content)
        except json.JSONDecodeError as exc:
            errors.append(f"content is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}")
            return errors

        if not isinstance(plan, dict):
            return ["top-level plan must be a JSON object"]

        required_top_level = ("task_analysis", "report_title", "report_toc", "constraints", "output_style", "queries")
        for key in required_top_level:
            if key not in plan:
                errors.append(f"missing top-level field: {key}")

        if not str(plan.get("report_title") or "").strip():
            errors.append("report_title must be non-empty")

        toc = plan.get("report_toc")
        if not isinstance(toc, list) or not toc:
            errors.append("report_toc must be a non-empty list")
        elif not all(isinstance(section, dict) and str(section.get("title") or "").strip() for section in toc):
            errors.append("each report_toc item must include a non-empty title")

        constraints = plan.get("constraints")
        if not isinstance(constraints, list) or not constraints:
            errors.append("constraints must be a non-empty list")

        output_style = plan.get("output_style")
        if not isinstance(output_style, dict) or not output_style:
            errors.append("output_style must be a non-empty object")

        queries = plan.get("queries")
        if not isinstance(queries, list) or not queries:
            errors.append("queries must be a non-empty list")
        elif len(queries) < 2 and len(toc) > 2:
            errors.append("queries must cover the plan; write at least two distinct queries for multi-section plans")
        elif isinstance(queries, list):
            for index, query in enumerate(queries, start=1):
                if not isinstance(query, dict):
                    errors.append(f"queries[{index}] must be an object")
                    continue
                if len(str(query.get("query") or "").strip()) < 8:
                    errors.append(f"queries[{index}].query must be a meaningful search query")
                if not str(query.get("tool") or "").strip():
                    errors.append(f"queries[{index}].tool is required")
                target_sections = query.get("target_sections")
                target_claims = query.get("target_claims")
                if not target_sections and not target_claims and not query.get("target_claim_ids"):
                    errors.append(f"queries[{index}] must include target_sections, target_claims, or target_claim_ids")

        return errors

    async def awrap_tool_call(self, request, handler):
        tool_call = request.tool_call if hasattr(request, "tool_call") else {}
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        if not isinstance(args, dict) or not self._is_plan_write(tool_name, args):
            return await handler(request)

        content = args.get("content")
        errors = self._validate_plan_payload(str(content) if content is not None else "")
        if not errors:
            return await handler(request)

        failures = _session_plan_validation_failures.get()
        failures = 0 if failures is None else failures
        failures += 1
        _session_plan_validation_failures.set(failures)
        logger.warning(
            "Rejected incomplete /shared/plan.json write (%d/%d): %s",
            failures,
            self._MAX_REPAIR_ATTEMPTS,
            "; ".join(errors),
        )

        if failures >= self._MAX_REPAIR_ATTEMPTS:
            repair_instruction = (
                "This is the second planner JSON validation failure in this run. Stop trying to hand-write "
                "a large JSON file. Your next action must be the typed write_plan tool with a compact plan: "
                "1-5 sections, no long prose fields, and only the essential researcher queries. If write_plan "
                "is unavailable, return control to the orchestrator so it can use the deterministic compact "
                "plan builder. Do not call write_file for /shared/plan.json again."
            )
        else:
            repair_instruction = (
                "Use the typed write_plan tool now if it is available. If you must repair this manually, "
                "rewrite the plan as compact valid JSON with 1-5 sections, no long prose fields, and a "
                "non-empty queries array. Do not continue to researcher-agent until the plan write succeeds."
            )

        return ToolMessage(
            content=(
                "PLAN_FILE_VALIDATION_FAILED: /shared/plan.json was not written because it is incomplete. "
                f"Issues: {'; '.join(errors)}.\n\n"
                f"{repair_instruction}"
            ),
            tool_call_id=tool_call.get("id", ""),
            name=tool_name,
        )


class ArtifactWriteValidationMiddleware(AgentMiddleware):
    """Reject corrupted artifact writes before they enter DeepAgents state.

    MiniMax can occasionally include transport/display truncation markers in a
    long ``write_file`` or ``edit_file`` argument. If that text is accepted, the
    next model turn sees a file containing literal ``...(argument truncated)``
    and wastes more turns trying to repair a file that should never have been
    written in that form.
    """

    _FILE_TOOLS = {"write_file", "edit_file"}
    _CONTENT_KEYS = {"content", "new_string"}
    _TRUNCATION_MARKERS = (
        "(argument truncated)",
        "...(argument truncated)",
        "[omitted ",
        " chars total]",
        "[... truncated tool argument",
        "read the virtual file path if needed",
        "WRITE_FILE_CONTENT_STORED_SUCCESSFULLY",
        "TOOL_RESULT_DISPLAY_SHORTENED",
        "TOOL_ARGUMENT_DISPLAY_SHORTENED",
    )

    @staticmethod
    def _target_path(args: dict) -> str:
        return str(args.get("file_path") or args.get("path") or args.get("filename") or "")

    @classmethod
    def _should_validate(cls, tool_name: str, args: dict) -> bool:
        if tool_name not in cls._FILE_TOOLS:
            return False
        path = cls._target_path(args)
        return path.startswith("/shared/") or path.startswith("shared/") or path == "/report.md"

    @classmethod
    def _content_errors(cls, args: dict) -> list[str]:
        errors: list[str] = []
        for key in cls._CONTENT_KEYS:
            value = args.get(key)
            if not isinstance(value, str):
                continue
            for marker in cls._TRUNCATION_MARKERS:
                if marker in value:
                    errors.append(f"{key} contains truncation marker {marker!r}")
        return errors

    async def awrap_tool_call(self, request, handler):
        tool_call = request.tool_call if hasattr(request, "tool_call") else {}
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        if not isinstance(args, dict) or not self._should_validate(tool_name, args):
            return await handler(request)

        errors = self._content_errors(args)
        if not errors:
            return await handler(request)

        path = self._target_path(args) or "the target file"
        logger.warning("Rejected artifact write to %s: %s", path, "; ".join(errors))
        return ToolMessage(
            content=(
                f"ARTIFACT_WRITE_VALIDATION_FAILED: {path} was not written because the proposed file "
                f"text contains truncation/omission markers: {'; '.join(errors)}.\n\n"
                "Rewrite a shorter complete artifact now. Do not copy callback display text, "
                "partial_json text, or read_file truncation summaries into the file. Keep the file compact."
            ),
            tool_call_id=tool_call.get("id", ""),
            name=tool_name,
        )


class PostWriteReadbackGuardMiddleware(AgentMiddleware):
    """Skip immediate self-verification reads after successful artifact writes.

    M3 often writes a complete artifact, sees the callback/tool display trimmed
    in the next turn, and then spends several expensive turns reading offsets to
    prove the file was not truncated. The runtime already knows whether the
    write tool succeeded, so this middleware turns near-immediate readbacks of
    that same file into a short confirmation message. Later orchestrator reads
    still work after the small suppression window is consumed.
    """

    _WRITE_TOOLS = {"write_file", "edit_file"}
    _READ_TOOLS = {"read_file", "grep"}

    def __init__(self, suppress_read_count: int = 2) -> None:
        self.suppress_read_count = max(0, suppress_read_count)

    @staticmethod
    def _target_path(args: dict) -> str:
        return str(args.get("file_path") or args.get("path") or args.get("filename") or "")

    @staticmethod
    def _is_artifact_path(path: str) -> bool:
        return path.startswith("/shared/") or path.startswith("shared/") or path == "/report.md"

    @staticmethod
    def _write_succeeded(result: ToolMessage) -> bool:
        content = str(result.content or "").lower()
        failure_markers = (
            "failed",
            "error",
            "already exists",
            "validation_failed",
            "not written",
            "could not",
        )
        return not any(marker in content for marker in failure_markers)

    async def awrap_tool_call(self, request, handler):
        tool_call = request.tool_call if hasattr(request, "tool_call") else {}
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            return await handler(request)

        path = self._target_path(args)
        if tool_name in self._READ_TOOLS and path and self._is_artifact_path(path):
            recent_writes = _session_recent_artifact_writes.get() or {}
            remaining = recent_writes.get(path, 0)
            if remaining > 0:
                recent_writes[path] = remaining - 1
                _session_recent_artifact_writes.set(recent_writes)
                logger.info("Suppressed immediate %s readback of recently written artifact %s", tool_name, path)
                return ToolMessage(
                    content=(
                        f"READ_AFTER_WRITE_CONFIRMED: {path} was just written successfully and the runtime "
                        "has already persisted the full artifact. Treat this as verification success. "
                        "Do not call grep, read_file, edit_file, or write_file again just to prove completeness. "
                        "Move to the next workflow step or finish."
                    ),
                    tool_call_id=tool_call.get("id", ""),
                    name=tool_name,
                )

        result = await handler(request)
        if (
            tool_name in self._WRITE_TOOLS
            and path
            and self._is_artifact_path(path)
            and isinstance(result, ToolMessage)
            and self._write_succeeded(result)
            and self.suppress_read_count > 0
        ):
            recent_writes = _session_recent_artifact_writes.get()
            if recent_writes is None:
                recent_writes = {}
            recent_writes[path] = self.suppress_read_count
            _session_recent_artifact_writes.set(recent_writes)
        return result


class PlannerCommitGuardMiddleware(AgentMiddleware):
    """Force planner-agent to commit a typed plan after bounded exploration."""

    def __init__(self, *, max_model_turns: int = 4, required_tool_name: str = "write_plan", max_repairs: int = 1):
        self.max_model_turns = max(1, max_model_turns)
        self.required_tool_name = required_tool_name
        self.max_repairs = max(0, max_repairs)

    def _has_required_tool_call(self, response: ModelResponse) -> bool:
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            if any(tool_call.get("name") == self.required_tool_name for tool_call in msg.tool_calls):
                return True
        return False

    def _required_tools(self, request) -> list:
        return [tool for tool in getattr(request, "tools", []) if getattr(tool, "name", "") == self.required_tool_name]

    @staticmethod
    def _commit_instruction() -> str:
        return (
            "Planner turn budget is exhausted. Your next and only action must be the `write_plan` tool. "
            "Do not call search tools, think, defer_task, or write_file. Use the evidence already gathered, "
            "write a compact executable plan with 1-5 top-level sections and non-empty researcher queries, "
            "then return a short summary after the tool succeeds."
        )

    async def awrap_model_call(self, request, handler):
        turns = _session_planner_model_turns.get()
        turns = 0 if turns is None else turns
        turns += 1
        _session_planner_model_turns.set(turns)

        response = await handler(request)
        if self._has_required_tool_call(response) or turns < self.max_model_turns:
            return response

        required_tools = self._required_tools(request)
        if not required_tools:
            logger.warning("Planner commit guard triggered but %s tool is unavailable", self.required_tool_name)
            return response

        repair_messages = [*request.messages, HumanMessage(content=self._commit_instruction())]
        for attempt in range(self.max_repairs + 1):
            logger.warning(
                "Planner reached model turn budget without %s; forcing commit attempt %d/%d",
                self.required_tool_name,
                attempt + 1,
                self.max_repairs + 1,
            )
            forced_request = request.override(
                messages=repair_messages,
                tools=required_tools,
                tool_choice=self.required_tool_name,
            )
            forced_response = await handler(forced_request)
            if self._has_required_tool_call(forced_response):
                return forced_response
            repair_messages = [*repair_messages, HumanMessage(content=self._commit_instruction())]

        logger.warning("Planner did not call %s after forced commit attempt", self.required_tool_name)
        return response


class ToolBudgetMiddleware(AgentMiddleware):
    """Hard-cap expensive tools per deep-research run."""

    def __init__(self, limits: dict[str, int], *, scope: str | None = None):
        self.default_limits = limits
        self.scope = scope

    async def awrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name", "") if hasattr(request, "tool_call") else ""
        active_limits = _session_tool_limits.get() or self.default_limits
        limit = _scoped_limit(active_limits, tool_name, self.scope)
        if limit is None:
            return await handler(request)

        counts = _session_tool_counts.get()
        if counts is None:
            counts = {}
            _session_tool_counts.set(counts)

        counter_key = _budget_key(tool_name, self.scope)
        current_count = counts.get(counter_key, 0)
        next_count = current_count + 1
        if current_count >= limit:
            _mark_tool_exhausted(tool_name, self.scope)
            logger.info(
                "Tool budget already exhausted for %s scope=%s (%d/%d)",
                tool_name,
                self.scope or "global",
                current_count,
                limit,
            )
            tool_call = request.tool_call if hasattr(request, "tool_call") else {}
            return ToolMessage(
                content=(
                    f"GLOBAL_SEARCH_BUDGET_EXHAUSTED for {tool_name}: limit is {limit} calls. "
                    f"The budget was already exhausted before this call. Do not call {tool_name} again. "
                    "Proceed immediately to synthesis from existing evidence. If you are a researcher-agent, "
                    "write your notes with write_file now. If you are the orchestrator, read existing notes, "
                    "call get_verified_sources, and write /report.md."
                ),
                tool_call_id=tool_call.get("id", ""),
                name=tool_name,
            )

        counts[counter_key] = next_count
        if next_count <= limit:
            return await handler(request)

        _mark_tool_exhausted(tool_name, self.scope)
        logger.info(
            "Tool budget exhausted for %s scope=%s (%d/%d)",
            tool_name,
            self.scope or "global",
            next_count,
            limit,
        )
        tool_call = request.tool_call if hasattr(request, "tool_call") else {}
        return ToolMessage(
            content=(
                f"GLOBAL_SEARCH_BUDGET_EXHAUSTED for {tool_name}: limit is {limit} calls. "
                f"Do not call {tool_name} again in this job. Proceed immediately with the evidence already "
                "gathered and write the requested notes/report. If you are a researcher-agent, your next "
                "tool call must be write_file. If you are the orchestrator, move to synthesis."
            ),
            tool_call_id=tool_call.get("id", ""),
            name=tool_name,
        )


class TaskSearchBudgetMiddleware(AgentMiddleware):
    """Hard-cap search tools inside each researcher assignment.

    The global researcher budget protects total spend, but it does not stop an
    early branch from consuming searches intended for later branches. This
    middleware reads the orchestrator's `Search budget: N search calls for this
    task` line and enforces that cap per subagent task.
    """

    _BUDGET_PATTERNS = (
        re.compile(r"Search budget:\s*(?P<count>\d+)\s+search calls?", re.IGNORECASE),
        re.compile(r"['\"]search_budget['\"]\s*[:=]\s*(?P<count>\d+)", re.IGNORECASE),
        re.compile(r"\bsearch_budget\s*[:=]\s*(?P<count>\d+)", re.IGNORECASE),
    )
    _TASK_ID_PATTERNS = (
        re.compile(r"['\"]task_id['\"]\s*[:=]\s*['\"]?(?P<task_id>[A-Za-z0-9_.-]+)", re.IGNORECASE),
        re.compile(r"\btask_id\s*[:=]\s*['\"]?(?P<task_id>[A-Za-z0-9_.-]+)", re.IGNORECASE),
        re.compile(r"\bQ(?P<num>\d+)\b", re.IGNORECASE),
    )

    def __init__(self, search_tool_names: set[str]) -> None:
        self.search_tool_names = search_tool_names

    @staticmethod
    def _message_text(request) -> str:
        parts: list[str] = []
        for message in getattr(request, "messages", []) or []:
            if isinstance(message, HumanMessage):
                content = message.content
                parts.append(content if isinstance(content, str) else str(content))
        return "\n".join(parts)

    @classmethod
    def _extract_budget(cls, text: str) -> int | None:
        for pattern in cls._BUDGET_PATTERNS:
            match = pattern.search(text)
            if match:
                return max(1, int(match.group("count")))
        return None

    @classmethod
    def _extract_task_key(cls, text: str) -> str:
        for pattern in cls._TASK_ID_PATTERNS:
            match = pattern.search(text)
            if match:
                if "task_id" in match.groupdict() and match.group("task_id"):
                    return match.group("task_id")
                if "num" in match.groupdict() and match.group("num"):
                    return f"Q{match.group('num')}"
        digest = hashlib.sha1(text[:1000].encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"task:{digest}"

    async def awrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name", "") if hasattr(request, "tool_call") else ""
        if tool_name not in self.search_tool_names:
            return await handler(request)

        text = self._message_text(request)
        limit = self._extract_budget(text)
        if limit is None:
            return await handler(request)

        task_key = self._extract_task_key(text)
        counts = _session_task_search_counts.get()
        if counts is None:
            counts = {}
            _session_task_search_counts.set(counts)

        current_count = counts.get(task_key, 0)
        if current_count >= limit:
            logger.info("Task search budget exhausted for %s (%d/%d)", task_key, current_count, limit)
            tool_call = request.tool_call if hasattr(request, "tool_call") else {}
            return ToolMessage(
                content=(
                    f"TASK_SEARCH_BUDGET_EXHAUSTED for {task_key}: limit is {limit} search calls. "
                    "Do not search again for this task. Write the notes, claim fragments, and any "
                    "unverified gaps from evidence already gathered."
                ),
                tool_call_id=tool_call.get("id", ""),
                name=tool_name,
            )

        counts[task_key] = current_count + 1
        return await handler(request)


class SearchBudgetExhaustionRepairMiddleware(AgentMiddleware):
    """Prevent exhausted search budgets from becoming infinite search loops."""

    def __init__(self, search_tool_names: set[str], *, max_repairs: int = 2, scope: str | None = None):
        self.search_tool_names = search_tool_names
        self.max_repairs = max(0, max_repairs)
        self.scope = scope

    def _exhausted_search_calls(self, response: ModelResponse) -> set[str]:
        exhausted = set(_session_exhausted_tools.get() or set())
        limits = _session_tool_limits.get() or {}
        counts = _session_tool_counts.get() or {}
        names: set[str] = set()
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name not in self.search_tool_names:
                    continue
                counter_key = _budget_key(tool_name, self.scope)
                limit = _scoped_limit(limits, tool_name, self.scope)
                if counter_key in exhausted or (limit is not None and counts.get(counter_key, 0) >= limit):
                    names.add(tool_name)
        return names

    @staticmethod
    def _repair_instruction(exhausted_names: set[str]) -> str:
        exhausted_list = ", ".join(sorted(exhausted_names))
        return (
            f"The search budget is exhausted for: {exhausted_list}. "
            "You are forbidden from calling those search tools again in this job. "
            "Do not say 'let me try one more time'. Do not search for another variant. "
            "Continue from the evidence already gathered.\n\n"
            "If you are researcher-agent: your next action must be write_file with the research notes, "
            "claim fragments, or honest UNVERIFIED entries that can be supported by existing evidence. "
            "If you are the orchestrator: read existing researcher files if needed, call get_verified_sources, "
            "and write /report.md. If evidence is thin, state that limitation instead of searching again."
        )

    def _strip_exhausted_search_calls(self, response: ModelResponse, exhausted_names: set[str]) -> ModelResponse:
        new_result = []
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                new_result.append(msg)
                continue

            kept_tool_calls = [tc for tc in msg.tool_calls if tc.get("name") not in exhausted_names]
            if len(kept_tool_calls) == len(msg.tool_calls):
                new_result.append(msg)
                continue

            if kept_tool_calls:
                new_result.append(
                    msg.model_copy(
                        update={
                            "content": _sync_content_tool_use_blocks(msg.content, kept_tool_calls),
                            "tool_calls": kept_tool_calls,
                        }
                    )
                )
            else:
                new_result.append(
                    AIMessage(
                        content=(
                            "Search budget is exhausted, so I will stop searching and synthesize from existing "
                            "evidence. The next step is to write the notes or final report from available files."
                        ),
                        id=msg.id,
                    )
                )

        return ModelResponse(result=new_result, structured_response=response.structured_response)

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        exhausted_names = self._exhausted_search_calls(response)
        if not exhausted_names:
            return response

        repair_messages = list(request.messages)
        for attempt in range(self.max_repairs):
            logger.warning(
                "Model attempted exhausted search tool(s) %s; requesting synthesize/write repair (%d/%d)",
                sorted(exhausted_names),
                attempt + 1,
                self.max_repairs,
            )
            repair_messages = [
                *repair_messages,
                HumanMessage(content=self._repair_instruction(exhausted_names)),
            ]
            response = await handler(request.override(messages=repair_messages))
            exhausted_names = self._exhausted_search_calls(response)
            if not exhausted_names:
                return response

        logger.warning(
            "Model kept calling exhausted search tool(s) after repair; stripping calls: %s",
            sorted(exhausted_names),
        )
        return self._strip_exhausted_search_calls(response, exhausted_names)


class SequentialSearchMiddleware(AgentMiddleware):
    """Bound parallel web search tool calls per model response.

    Some models occasionally emit multiple expensive search calls despite
    instructions to search sequentially. Trimming extras keeps research moving
    in bounded turns and avoids large request bursts to SearXNG/Jina.
    """

    def __init__(self, search_tool_names: set[str], *, default_limit: int = 2):
        self.search_tool_names = search_tool_names
        self.default_limit = max(1, default_limit)

    async def awrap_model_call(self, request, handler):
        response = await handler(request)

        needs_fix = False
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            search_calls = [tc for tc in msg.tool_calls if tc.get("name") in self.search_tool_names]
            if len(search_calls) > self.default_limit:
                needs_fix = True
                break

        if not needs_fix:
            return response

        new_result = []
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                new_result.append(msg)
                continue

            seen_search = 0
            kept_tool_calls = []
            dropped_names = []
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name in self.search_tool_names:
                    seen_search += 1
                    if seen_search > self.default_limit:
                        dropped_names.append(tool_name)
                        continue
                kept_tool_calls.append(tool_call)

            if dropped_names:
                logger.info(
                    "Trimmed %d parallel search tool call(s) from model response: %s",
                    len(dropped_names),
                    dropped_names,
                )
                new_result.append(
                    msg.model_copy(
                        update={
                            "content": _sync_content_tool_use_blocks(msg.content, kept_tool_calls),
                            "tool_calls": kept_tool_calls,
                        }
                    )
                )
            else:
                new_result.append(msg)

        return ModelResponse(result=new_result, structured_response=response.structured_response)


class TaskBatchLimitMiddleware(AgentMiddleware):
    """Defer excess researcher tasks in one model response.

    MiniMax's Anthropic-compatible endpoint is much less reliable when several
    long researcher subagents are launched at the same instant. This middleware
    keeps the research plan intact by converting extra same-turn `task` calls
    into a lightweight `defer_task` tool call. The orchestrator receives an
    explicit reminder to launch the deferred researcher task after the current
    batch finishes, preserving coverage while avoiding provider-side bursts.
    """

    def __init__(
        self,
        *,
        task_tool_name: str = "task",
        defer_tool_name: str = "defer_task",
        default_limit: int = 1,
    ) -> None:
        self.task_tool_name = task_tool_name
        self.defer_tool_name = defer_tool_name
        self.default_limit = max(1, default_limit)

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        active_limits = _session_parallel_tool_limits.get() or {}
        limit = max(1, int(active_limits.get(self.task_tool_name, self.default_limit)))

        needs_fix = False
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            task_calls = [tc for tc in msg.tool_calls if tc.get("name") == self.task_tool_name]
            if len(task_calls) > limit:
                needs_fix = True
                break

        if not needs_fix:
            return response

        new_result = []
        for msg in response.result:
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                new_result.append(msg)
                continue

            seen_tasks = 0
            transformed_tool_calls = []
            name_map: dict[str, str] = {}
            args_map: dict[str, dict] = {}
            deferred_count = 0
            for tool_call in msg.tool_calls:
                if tool_call.get("name") != self.task_tool_name:
                    transformed_tool_calls.append(tool_call)
                    continue

                seen_tasks += 1
                if seen_tasks <= limit:
                    transformed_tool_calls.append(tool_call)
                    continue

                original_args = tool_call.get("args") or {}
                if not isinstance(original_args, dict):
                    original_args = {"description": str(original_args)}
                description = str(original_args.get("description") or original_args)
                reason = (
                    f"Deferred because only {limit} researcher-agent task(s) may run in one batch for "
                    "MiniMax connection reliability. After reviewing the completed task result, call task() "
                    "for this description if the section is still needed."
                )
                defer_args = {"description": description, "reason": reason}
                new_call = {**tool_call, "name": self.defer_tool_name, "args": defer_args}
                transformed_tool_calls.append(new_call)
                if tool_call.get("id"):
                    name_map[tool_call["id"]] = self.defer_tool_name
                    args_map[tool_call["id"]] = defer_args
                deferred_count += 1

            if deferred_count:
                logger.info(
                    "Deferred %d parallel researcher task call(s); running %d this batch",
                    deferred_count,
                    limit,
                )
                new_result.append(
                    msg.model_copy(
                        update={
                            "content": _sync_content_tool_use_blocks(
                                msg.content,
                                transformed_tool_calls,
                                name_map,
                                args_map,
                            ),
                            "tool_calls": transformed_tool_calls,
                        }
                    )
                )
            else:
                new_result.append(msg)

        return ModelResponse(result=new_result, structured_response=response.structured_response)


class SourceRegistryMiddleware(AgentMiddleware):
    """Intercepts tool call results to build a registry of actual sources.

    Two responsibilities:
    1. awrap_tool_call: Capture URLs/citation keys from tool results
    2. awrap_model_call: Inject a consolidated source list into the LLM context
       so the orchestrator has a single, authoritative reference list when
       writing the final report (no manual reconciliation across subagent files)

    Source capture is gated only by the agent's loaded tool set
    (``source_tool_names``). Internal scratchpad/runtime tools (think,
    write_file, read_file, etc.) are added by deepagents itself and never
    appear in that set, so they are implicitly excluded. Tools registered as
    configured data sources additionally carry a ``source_id`` label, but a
    tool does *not* have to be declared under ``data_sources`` to contribute
    sources — agents can be passed citable tools directly.

    The registry is also used by verify_citations() to strip fabricated,
    stale, or intermediate-artifact citations from the final report.
    """

    def __init__(self, source_tool_names: set[str] | None = None, max_sources_for_prompt: int = 120) -> None:
        self.registry = SourceRegistry()
        self._source_tool_names = source_tool_names or set()
        self.max_sources_for_prompt = max_sources_for_prompt

    def _get_registry(self) -> SourceRegistry:
        """Return the session-scoped registry if set, otherwise the instance registry."""
        from aiq_agent.common.citation_verification import get_session_registry

        return get_session_registry() or self.registry

    async def awrap_tool_call(self, request, handler):
        """Capture sources from tool results after execution.

        Capture is gated only by the agent's loaded tool set
        (``source_tool_names``). Internal scratchpad/runtime tools (think,
        write_file, read_file, etc.) are added by deepagents itself and never
        appear in that set, so they are implicitly excluded.

        Tools that resolve to a configured data source via
        :func:`get_source_id_for_tool` get a ``source_id`` label. Tools passed
        directly to the agent without a data-source declaration are still
        captured — their results are real, citable evidence even when
        ``data_source_registry`` does not know about them — but their entries
        carry no ``source_id``.
        """
        result = await handler(request)
        if isinstance(result, ToolMessage) and result.content:
            tool_name = ""
            if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", "")
            if tool_name not in self._source_tool_names:
                return result
            source_id = get_source_id_for_tool(tool_name)
            sources = extract_sources_from_tool_result(tool_name, str(result.content), source_id=source_id)
            active_registry = self._get_registry()
            for source in sources:
                active_registry.add(source)
            if sources:
                logger.info(
                    "[CitationRegistry] Captured %d source(s) from %s: %s",
                    len(sources),
                    tool_name,
                    [s.url or s.citation_key for s in sources],
                )
        return result

    def get_source_list_text(self) -> str | None:
        """Build a consolidated source list for injection into retry feedback.

        Returns rendered template text, or None if no sources captured.
        Used by agent.run() to include the source list in retry messages
        when citation quality is poor.
        """
        from urllib.parse import urlparse

        from aiq_agent.common.citation_verification import _normalize_url

        sources = self._get_registry().all_sources()
        if not sources:
            return None

        seen: set[str] = set()
        template_sources = []
        for entry in sources:
            if len(template_sources) >= self.max_sources_for_prompt:
                break
            if entry.url:
                normalized = _normalize_url(entry.url)
                if normalized in seen:
                    continue
                seen.add(normalized)
                if entry.title:
                    title = entry.title
                else:
                    try:
                        title = urlparse(entry.url).netloc.replace("www.", "")
                    except Exception:
                        title = entry.url
                template_sources.append({"title": title, "url": entry.url, "source_class": entry.source_class})
            elif entry.citation_key:
                key = entry.citation_key
                if key in seen:
                    continue
                seen.add(key)
                template_sources.append({"title": key, "url": key, "source_class": entry.source_class})

        if not template_sources:
            return None

        try:
            template = load_prompt(_PROMPTS_DIR, "source_registry")
            rendered = render_prompt_template(template, sources=template_sources)
            omitted = max(0, len(sources) - len(template_sources))
            if omitted:
                rendered += (
                    f"\n\nNote: {omitted} additional captured sources were omitted from this prompt for brevity."
                )
            return rendered
        except Exception:
            logger.warning("Failed to load source_registry prompt template", exc_info=True)
            return None


class ToolResultPruningMiddleware(AgentMiddleware):
    """Truncates tool results before model calls to keep context manageable.

    Keeps the most recent N tool results more detailed, but still applies a
    hard cap to every tool result. Large search and subagent file payloads are
    preserved in the virtual filesystem; sending the full serialized payload
    back to MiniMax can exceed practical streaming/context limits.
    """

    def __init__(
        self,
        keep_last_n: int = 3,
        max_chars: int = 500,
        recent_max_chars: int = 12000,
        max_tool_call_arg_chars: int = 2000,
    ):
        self.keep_last_n = keep_last_n
        self.max_chars = max_chars
        self.recent_max_chars = max(recent_max_chars, max_chars)
        self.max_tool_call_arg_chars = max(max_tool_call_arg_chars, 200)

    async def awrap_model_call(self, request, handler):
        """Truncate older ToolMessage content before sending to the model."""
        # Find all ToolMessage indices
        tool_indices = [i for i, msg in enumerate(request.messages) if isinstance(msg, ToolMessage)]

        # Older tool results get an aggressive cap; recent results get a
        # larger cap but are still bounded to protect the next model call.
        older_indices = set(tool_indices[: -self.keep_last_n])

        pruned_messages = []
        for i, msg in enumerate(request.messages):
            if isinstance(msg, ToolMessage) and msg.content:
                content = str(msg.content)
                limit = self._limit_for_tool_message(msg, content, i in older_indices)
                if len(content) > limit:
                    truncated_content = content[:limit] + (
                        f"\n\nTOOL_RESULT_DISPLAY_SHORTENED: this tool result originally had {len(content)} "
                        "characters and was shortened only in prompt history to save tokens. Do not copy this "
                        "notice into files, and do not treat it as evidence that the underlying artifact was "
                        "truncated."
                    )
                    pruned_messages.append(
                        ToolMessage(
                            content=truncated_content,
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", None),
                            id=msg.id,
                        )
                    )
                else:
                    pruned_messages.append(msg)
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                pruned_messages.append(self._prune_ai_tool_call_args(msg))
            else:
                pruned_messages.append(msg)

        return await handler(request.override(messages=pruned_messages))

    def _limit_for_tool_message(self, msg: ToolMessage, content: str, is_older: bool) -> int:
        """Preserve high-value source results longer than generic tool chatter."""
        if not is_older:
            return self.recent_max_chars

        tool_name = str(getattr(msg, "name", "") or "")
        if tool_name:
            sources = extract_sources_from_tool_result(tool_name, content)
            best_rank = max((source_class_rank(source.source_class) for source in sources), default=0)
            if best_rank >= source_class_rank("first_party"):
                return self.recent_max_chars
            if best_rank >= source_class_rank("third_party_authoritative"):
                return max(self.max_chars, self.recent_max_chars // 2)
        return self.max_chars

    def _prune_ai_tool_call_args(self, msg: AIMessage) -> AIMessage:
        """Trim historical tool-call arguments before the next model call.

        The tool has already executed by the time this middleware sees the
        message in request history. Keeping full ``write_file.content`` or large
        search arguments in old AIMessage tool calls can explode the next prompt
        even when ToolMessage results are capped.
        """
        changed = False
        pruned_tool_calls: list[dict] = []
        args_map: dict[str, dict] = {}

        for tool_call in msg.tool_calls:
            pruned_call = dict(tool_call)
            args = tool_call.get("args")
            if isinstance(args, dict):
                pruned_args, args_changed = self._prune_arg_dict(args, tool_call.get("name"))
                if args_changed:
                    pruned_call["args"] = pruned_args
                    if tool_call.get("id"):
                        args_map[tool_call["id"]] = pruned_args
                    changed = True
            pruned_tool_calls.append(pruned_call)

        if not changed:
            return msg

        return msg.model_copy(
            update={
                "content": _sync_content_tool_use_blocks(msg.content, pruned_tool_calls, args_map=args_map),
                "tool_calls": pruned_tool_calls,
            }
        )

    def _prune_arg_dict(self, args: dict, tool_name: str | None = None) -> tuple[dict, bool]:
        changed = False
        pruned: dict = {}
        for key, value in args.items():
            if isinstance(value, str) and len(value) > self.max_tool_call_arg_chars:
                changed = True
                path_hint = args.get("file_path") or args.get("path") or args.get("filename")
                if tool_name == "write_file" and key == "content":
                    location = f" at {path_hint}" if path_hint else ""
                    pruned[key] = (
                        f"WRITE_FILE_CONTENT_STORED_SUCCESSFULLY{location}; {len(value)} characters are hidden "
                        "from prompt history to save tokens. The write already executed. Do not rewrite or read "
                        "the file merely to verify this hidden argument."
                    )
                else:
                    head = value[: self.max_tool_call_arg_chars].rstrip()
                    pruned[key] = (
                        f"{head}\n\nTOOL_ARGUMENT_DISPLAY_SHORTENED: original argument had {len(value)} "
                        "characters and was shortened only in prompt history."
                    )
            elif isinstance(value, dict):
                nested, nested_changed = self._prune_arg_dict(value, tool_name)
                pruned[key] = nested
                changed = changed or nested_changed
            else:
                pruned[key] = value
        return pruned, changed

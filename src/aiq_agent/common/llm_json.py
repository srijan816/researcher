# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate-repair-retry helpers for LLM JSON output."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable
from typing import Any
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


async def parse_with_repair(
    raw_output: str,
    schema: type[ModelT],
    repair_llm: Callable[[str], Any] | None = None,
    max_repair_attempts: int = 2,
) -> ModelT | None:
    """Parse and validate model JSON, optionally asking an LLM to repair it.

    The function never raises for JSON/validation failures. This is deliberate:
    malformed structured output should degrade the current research attempt,
    not crash the whole job.
    """
    candidates = _local_candidates(raw_output)
    for label, candidate in candidates:
        parsed = _validate_candidate(candidate, schema)
        if parsed is not None:
            if label != "direct":
                logger.info("Parsed %s after %s cleanup", schema.__name__, label)
            return parsed

    if repair_llm is None or max_repair_attempts <= 0:
        logger.warning("Could not parse %s and no repair LLM was provided", schema.__name__)
        return None

    last_raw = raw_output
    for attempt in range(1, max_repair_attempts + 1):
        prompt = _repair_prompt(last_raw, schema)
        try:
            repaired = repair_llm(prompt)
            if inspect.isawaitable(repaired):
                repaired = await repaired
        except Exception:
            logger.warning("JSON repair attempt %d for %s failed", attempt, schema.__name__, exc_info=True)
            return None

        repaired_text = _coerce_text(repaired)
        for label, candidate in _local_candidates(repaired_text):
            parsed = _validate_candidate(candidate, schema)
            if parsed is not None:
                logger.info("Parsed %s after repair attempt %d (%s)", schema.__name__, attempt, label)
                return parsed
        last_raw = repaired_text

    logger.warning("Could not parse %s after %d repair attempt(s)", schema.__name__, max_repair_attempts)
    return None


def _local_candidates(raw_output: str) -> list[tuple[str, str]]:
    text = _coerce_text(raw_output).strip()
    candidates = [("direct", text)]

    fence_match = _CODE_FENCE_RE.match(text)
    if fence_match:
        candidates.append(("code_fence", fence_match.group(1).strip()))

    extracted = _extract_json_substring(text)
    if extracted and extracted != text:
        candidates.append(("json_substring", extracted))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for label, candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append((label, candidate))
    return unique


def _validate_candidate(candidate: str, schema: type[ModelT]) -> ModelT | None:
    try:
        payload = json.loads(candidate)
        return schema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return None


def _extract_json_substring(text: str) -> str | None:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _repair_prompt(raw_output: str, schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "Fix the following JSON so it validates against this Pydantic JSON schema. "
        "Return only valid JSON and do not add commentary.\n\n"
        f"Schema:\n{schema_json}\n\n"
        f"Invalid JSON:\n{raw_output}"
    )


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    return str(value)

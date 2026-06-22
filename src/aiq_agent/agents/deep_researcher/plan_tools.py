# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed plan-writing tools for the DeepAgents virtual filesystem."""

from __future__ import annotations

import json
from typing import Annotated
from typing import Any

from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool

from .plan_schema import PlanConstraint
from .plan_schema import PlanOutputStyle
from .plan_schema import PlanQuery
from .plan_schema import PlanTaskAnalysis
from .plan_schema import PlanTocItem
from .plan_schema import WritePlanInput
from .plan_schema import build_plan_payload

PLAN_PATHS = ("/shared/plan.json", "/plan.json")


def plan_json_from_tool_args(
    *,
    report_title: str,
    report_toc: list[PlanTocItem | dict[str, Any]],
    queries: list[PlanQuery | dict[str, Any]],
    constraints: list[PlanConstraint | str | dict[str, Any]] | None = None,
    output_style: PlanOutputStyle | dict[str, Any] | None = None,
    task_analysis: PlanTaskAnalysis | dict[str, Any] | None = None,
    fact_ledger_targets: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Build canonical plan JSON from typed tool arguments."""

    payload: dict[str, Any] = {
        "report_title": report_title,
        "report_toc": report_toc,
        "queries": queries,
        "output_style": output_style,
        "task_analysis": task_analysis,
        "fact_ledger_targets": fact_ledger_targets,
    }
    if constraints is not None:
        payload["constraints"] = constraints

    input_data = WritePlanInput.model_validate(payload)
    return json.dumps(build_plan_payload(input_data), indent=2, ensure_ascii=False)


def _upsert_virtual_file(backend: StateBackend, file_path: str, content: str) -> str:
    """Write or replace a DeepAgents virtual file using the active graph state."""

    existing = backend.read(file_path)
    if existing.error or existing.file_data is None:
        result = backend.write(file_path, content)
        if result.error:
            return f"Error writing {file_path}: {result.error}"
        return f"Updated file {file_path}"

    # Avoid edit_file-style exact replacement for potentially large plans.
    # StateBackend queues this update into the active LangGraph files channel,
    # which is the same mechanism used by DeepAgents' built-in write_file.
    backend._send_files_update({file_path: backend._prepare_for_storage(create_file_data(content))})  # noqa: SLF001
    return f"Updated file {file_path}"


def create_write_plan_tool() -> BaseTool:
    """Create a typed tool that writes `/shared/plan.json` without raw JSON emission."""

    def sync_write_plan(
        report_title: Annotated[str, "Concise report title."],
        report_toc: Annotated[list[dict[str, Any]], "Final report sections. Each item needs a title."],
        queries: Annotated[list[dict[str, Any]], "Self-contained researcher assignments."],
        constraints: Annotated[
            list[dict[str, Any] | str] | None,
            "Acceptance criteria for the final report. If omitted, the runtime will add a source-backed default.",
        ] = None,
        output_style: Annotated[dict[str, Any] | None, "Optional final report style hints."] = None,
        task_analysis: Annotated[dict[str, Any] | None, "Optional compact task analysis."] = None,
        fact_ledger_targets: Annotated[
            dict[str, list[dict[str, Any]]] | None,
            "Optional entity fact targets for fact-ledger population.",
        ] = None,
    ) -> str:
        plan_json = plan_json_from_tool_args(
            report_title=report_title,
            report_toc=report_toc,
            queries=queries,
            constraints=constraints,
            output_style=output_style,
            task_analysis=task_analysis,
            fact_ledger_targets=fact_ledger_targets,
        )
        plan_payload = json.loads(plan_json)
        backend = StateBackend()
        results = [_upsert_virtual_file(backend, path, plan_json) for path in PLAN_PATHS]
        if any(result.startswith("Error") for result in results):
            return "\n".join(results)
        return (
            "Plan successfully written and validated at /shared/plan.json. "
            f"Sections: {len(plan_payload.get('report_toc', []))}. "
            f"Researcher queries: {len(plan_payload.get('queries', []))}."
        )

    async def async_write_plan(
        report_title: Annotated[str, "Concise report title."],
        report_toc: Annotated[list[dict[str, Any]], "Final report sections. Each item needs a title."],
        queries: Annotated[list[dict[str, Any]], "Self-contained researcher assignments."],
        constraints: Annotated[
            list[dict[str, Any] | str] | None,
            "Acceptance criteria for the final report. If omitted, the runtime will add a source-backed default.",
        ] = None,
        output_style: Annotated[dict[str, Any] | None, "Optional final report style hints."] = None,
        task_analysis: Annotated[dict[str, Any] | None, "Optional compact task analysis."] = None,
        fact_ledger_targets: Annotated[
            dict[str, list[dict[str, Any]]] | None,
            "Optional entity fact targets for fact-ledger population.",
        ] = None,
    ) -> str:
        return sync_write_plan(
            report_title=report_title,
            report_toc=report_toc,
            queries=queries,
            constraints=constraints,
            output_style=output_style,
            task_analysis=task_analysis,
            fact_ledger_targets=fact_ledger_targets,
        )

    return StructuredTool.from_function(
        name="write_plan",
        description=(
            "Create or replace the canonical /shared/plan.json research plan using typed fields. "
            "Use this instead of write_file for plan.json. Do not hand-write JSON."
        ),
        func=sync_write_plan,
        coroutine=async_write_plan,
        infer_schema=True,
    )

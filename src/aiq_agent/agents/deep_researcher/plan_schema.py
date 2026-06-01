# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed planner schema used by the deep research `write_plan` tool."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


def _unwrap_item(value: Any) -> Any:
    """Normalize Anthropic/MiniMax tool-call wrappers like {"item": [...]}.

    MiniMax M3 sometimes serializes JSON-schema arrays through an object with an
    `item` key. Accepting that shape keeps a valid plan from becoming a costly
    retry loop while preserving strict validation of the normalized payload.
    """

    if isinstance(value, dict) and set(value) == {"item"}:
        return _unwrap_item(value["item"])
    if isinstance(value, list):
        return [_unwrap_item(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap_item(item) for key, item in value.items()}
    return value


def _textish(value: Any, *keys: str) -> str | None:
    """Extract a useful string from provider-specific text wrappers."""

    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    for key in (*keys, "claim", "text", "$text", "value", "title", "constraint"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item
    return None


class PlanTargetClaim(BaseModel):
    """A compact claim or question a researcher should resolve."""

    claim_id: str = Field(default="")
    claim_type: str = Field(default="discovery")
    claim: str
    required_source_class: str = Field(default="mixed")

    @model_validator(mode="before")
    @classmethod
    def _normalize_claim_shape(cls, value: Any) -> Any:
        value = _unwrap_item(value)
        text = _textish(value)
        if text is not None:
            if isinstance(value, dict):
                normalized = dict(value)
                normalized["claim"] = text
                return normalized
            return {"claim": text}
        return value

    @field_validator("claim")
    @classmethod
    def _claim_is_meaningful(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 6:
            raise ValueError("claim must be meaningful")
        return value


class PlanTocItem(BaseModel):
    """A section or subsection in the final report."""

    id: str | None = None
    title: str
    per_entity: str | None = None
    subsections: list[PlanTocItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_wrapped_lists(cls, value: Any) -> Any:
        return _unwrap_item(value)

    @field_validator("title")
    @classmethod
    def _title_is_meaningful(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("TOC title must be meaningful")
        return value[:180].rstrip(" .,:;")


class PlanConstraint(BaseModel):
    """Acceptance criterion for the final report."""

    category: str = Field(default="content")
    constraint: str
    rationale: str = Field(default="")
    verification: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize_constraint_shape(cls, value: Any) -> Any:
        value = _unwrap_item(value)
        text = _textish(value)
        if text is not None:
            if isinstance(value, dict):
                normalized = dict(value)
                normalized["constraint"] = text
                return normalized
            return {"constraint": text}
        return value

    @field_validator("constraint")
    @classmethod
    def _constraint_is_meaningful(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 8:
            raise ValueError("constraint must be meaningful")
        return value


class PlanOutputStyle(BaseModel):
    """Instructions that shape the final report without replacing the TOC."""

    mode: Literal["standard_report", "focused_screen", "lesson_first"] = "standard_report"
    target: str = Field(default="Comprehensive source-grounded report matching the approved scope.")
    avoid: list[str] = Field(default_factory=list)
    topic_anchor: str | None = None
    motion_anchor: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_wrapped_lists(cls, value: Any) -> Any:
        return _unwrap_item(value)


class PlanTaskAnalysis(BaseModel):
    """Compact analysis fields needed by downstream orchestration."""

    user_intent: str = Field(default="")
    claim_profile: dict[str, Any] = Field(default_factory=dict)
    source_strategy: dict[str, Any] = Field(default_factory=dict)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    explicit_requirements: list[str] = Field(default_factory=list)
    implicit_requirements: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    language: str = Field(default="English")

    @model_validator(mode="before")
    @classmethod
    def _normalize_wrapped_lists(cls, value: Any) -> Any:
        return _unwrap_item(value)


class PlanQuery(BaseModel):
    """A self-contained researcher assignment."""

    query: str
    tool: str = Field(default="advanced_web_search_tool")
    task_id: str | None = None
    task_category: str = Field(default="evidence")
    relevance_weight: int = Field(default=1, ge=1, le=5)
    budget_percent: float | None = None
    search_budget: int | None = Field(default=None, ge=1)
    target_claims: list[PlanTargetClaim] = Field(default_factory=list)
    target_claim_ids: list[str] = Field(default_factory=list)
    target_sections: list[str] = Field(default_factory=list)
    target_entity: str | None = None
    target_class: str = Field(default="mixed")
    rationale: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize_wrapped_lists(cls, value: Any) -> Any:
        return _unwrap_item(value)

    @field_validator("query")
    @classmethod
    def _query_is_meaningful(cls, value: str) -> str:
        value = " ".join(value.split()).strip()
        if len(value) < 8:
            raise ValueError("query must be meaningful")
        return value


class WritePlanInput(BaseModel):
    """Input accepted from the model by the typed `write_plan` tool."""

    report_title: str
    report_toc: list[PlanTocItem]
    queries: list[PlanQuery]
    constraints: list[PlanConstraint | str]
    output_style: PlanOutputStyle | dict[str, Any] | None = None
    task_analysis: PlanTaskAnalysis | dict[str, Any] | None = None
    fact_ledger_targets: dict[str, Any] | list[dict[str, Any]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_wrapped_lists(cls, value: Any) -> Any:
        return _unwrap_item(value)

    @field_validator("report_title")
    @classmethod
    def _report_title_is_meaningful(cls, value: str) -> str:
        value = " ".join(value.split()).strip()
        if len(value) < 4:
            raise ValueError("report_title must be meaningful")
        return value[:180].rstrip(" .,:;")

    @field_validator("report_toc")
    @classmethod
    def _toc_is_non_empty(cls, value: list[PlanTocItem]) -> list[PlanTocItem]:
        if not value:
            raise ValueError("report_toc must contain at least one section")
        return value

    @field_validator("queries")
    @classmethod
    def _queries_are_non_empty(cls, value: list[PlanQuery]) -> list[PlanQuery]:
        if not value:
            raise ValueError("queries must contain at least one researcher assignment")
        return value

    @field_validator("fact_ledger_targets", mode="before")
    @classmethod
    def _normalize_fact_ledger_targets(cls, value: Any) -> Any:
        value = _unwrap_item(value)
        if isinstance(value, list):
            return {"entities": value}
        return value


def _constraint_to_dict(value: PlanConstraint | str) -> dict[str, str]:
    if isinstance(value, PlanConstraint):
        return value.model_dump()
    return {
        "category": "content",
        "constraint": value.strip(),
        "rationale": "User or planner requirement.",
        "verification": "Check the final report against this criterion.",
    }


def _toc_to_dict(item: PlanTocItem, index: int, prefix: str = "") -> dict[str, Any]:
    section_id = item.id or f"{prefix}{index}"
    return {
        "id": section_id,
        "title": item.title,
        "per_entity": item.per_entity,
        "subsections": [
            _toc_to_dict(subsection, sub_index, f"{section_id}.")
            for sub_index, subsection in enumerate(item.subsections, start=1)
        ],
    }


def _source_strategy_for_queries() -> dict[str, Any]:
    return {
        "required_source_classes": [
            "first_party",
            "primary_issuer",
            "academic",
            "authoritative_third_party",
            "trade_press",
            "forum",
            "mixed",
        ],
        "diversity_floor_domains": 4,
        "max_single_domain_share": 0.4,
        "numeric_claim_rule": (
            "Use primary or authoritative sources for numbers; if unavailable, label "
            "secondary-source numbers as partially verified."
        ),
    }


def build_plan_payload(input_data: WritePlanInput) -> dict[str, Any]:
    """Expand compact tool arguments into the canonical `/shared/plan.json` shape."""

    toc = [_toc_to_dict(item, index) for index, item in enumerate(input_data.report_toc, start=1)]
    section_titles = [section["title"] for section in toc]

    task_analysis = (
        input_data.task_analysis
        if isinstance(input_data.task_analysis, PlanTaskAnalysis)
        else PlanTaskAnalysis.model_validate(input_data.task_analysis or {})
    )
    task_analysis_dict = task_analysis.model_dump()
    task_analysis_dict.setdefault("source_strategy", {})
    if not task_analysis_dict["source_strategy"]:
        task_analysis_dict["source_strategy"] = _source_strategy_for_queries()
    task_analysis_dict.setdefault("claim_profile", {})
    if not task_analysis_dict["claim_profile"]:
        task_analysis_dict["claim_profile"] = {
            "claim_density": "medium",
            "verifiability": "mixed",
            "claims": [],
        }

    output_style = (
        input_data.output_style
        if isinstance(input_data.output_style, PlanOutputStyle)
        else PlanOutputStyle.model_validate(input_data.output_style or {})
    )

    queries: list[dict[str, Any]] = []
    raw_queries = list(input_data.queries)
    budget_percents = [query.budget_percent for query in raw_queries]
    needs_budget_normalization = (
        not raw_queries
        or any(percent is None or percent <= 0 for percent in budget_percents)
        or abs(sum(float(percent or 0) for percent in budget_percents) - 100.0) > 1.0
    )
    if needs_budget_normalization:
        weights = [max(1, int(query.relevance_weight or 1)) for query in raw_queries] or [1]
        weight_total = sum(weights) or 1
        normalized_budget_percents = [round(weight / weight_total * 100.0, 1) for weight in weights]
        if normalized_budget_percents:
            normalized_budget_percents[-1] = round(
                normalized_budget_percents[-1] + (100.0 - sum(normalized_budget_percents)), 1
            )
    else:
        normalized_budget_percents = [round(float(percent or 0), 1) for percent in budget_percents]
        if normalized_budget_percents:
            normalized_budget_percents[-1] = round(
                normalized_budget_percents[-1] + (100.0 - sum(normalized_budget_percents)), 1
            )

    for index, query in enumerate(input_data.queries, start=1):
        query_dict = query.model_dump()
        query_dict["task_id"] = query_dict.get("task_id") or f"Q{index}"
        query_dict["budget_percent"] = normalized_budget_percents[index - 1]
        if not query_dict.get("target_sections"):
            query_dict["target_sections"] = section_titles[:]
        if not query_dict.get("target_claims"):
            claim_id = f"C{index}"
            query_dict["target_claims"] = [
                {
                    "claim_id": claim_id,
                    "claim_type": "discovery",
                    "claim": f"Evidence needed for: {query.query}",
                    "required_source_class": "mixed",
                }
            ]
            query_dict["target_claim_ids"] = [claim_id]
        elif not query_dict.get("target_claim_ids"):
            for claim_index, claim in enumerate(query_dict["target_claims"], start=1):
                if isinstance(claim, dict) and not claim.get("claim_id"):
                    claim["claim_id"] = f"C{index}.{claim_index}"
            query_dict["target_claim_ids"] = [
                claim["claim_id"]
                for claim in query_dict["target_claims"]
                if isinstance(claim, dict) and claim.get("claim_id")
            ]
        queries.append(query_dict)

    plan = {
        "task_analysis": task_analysis_dict,
        "report_title": input_data.report_title,
        "report_toc": toc,
        "constraints": [_constraint_to_dict(constraint) for constraint in input_data.constraints],
        "output_style": output_style.model_dump(),
        "queries": queries,
    }
    if input_data.fact_ledger_targets:
        plan["fact_ledger_targets"] = input_data.fact_ledger_targets
    return plan

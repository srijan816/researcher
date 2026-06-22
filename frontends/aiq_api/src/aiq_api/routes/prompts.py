# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routes for inspecting agent prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field


@dataclass(frozen=True)
class PromptSpec:
    """Static metadata for one prompt template used by the research workflow."""

    id: str
    agent: str
    role: str
    package: str
    resource_path: str
    relative_path: str
    description: str


PROMPT_SPECS: tuple[PromptSpec, ...] = (
    PromptSpec(
        id="chat_researcher.intent_classification",
        agent="chat_researcher",
        role="intent_classifier",
        package="aiq_agent.agents.chat_researcher",
        resource_path="prompts/intent_classification.j2",
        relative_path="src/aiq_agent/agents/chat_researcher/prompts/intent_classification.j2",
        description="Classifies a chat turn as meta, shallow research, or deep research.",
    ),
    PromptSpec(
        id="clarifier.research_clarification",
        agent="clarifier",
        role="clarifier",
        package="aiq_agent.agents.clarifier",
        resource_path="prompts/research_clarification.j2",
        relative_path="src/aiq_agent/agents/clarifier/prompts/research_clarification.j2",
        description="Decides whether the research request needs clarification and asks focused follow-ups.",
    ),
    PromptSpec(
        id="clarifier.plan_generation",
        agent="clarifier",
        role="plan_generator",
        package="aiq_agent.agents.clarifier",
        resource_path="prompts/plan_generation.j2",
        relative_path="src/aiq_agent/agents/clarifier/prompts/plan_generation.j2",
        description="Builds the human-reviewable research plan preview.",
    ),
    PromptSpec(
        id="shallow_researcher.researcher",
        agent="shallow_researcher",
        role="researcher",
        package="aiq_agent.agents.shallow_researcher",
        resource_path="prompts/researcher.j2",
        relative_path="src/aiq_agent/agents/shallow_researcher/prompts/researcher.j2",
        description="Guides the bounded shallow research tool loop and citation behavior.",
    ),
    PromptSpec(
        id="deep_researcher.planner",
        agent="deep_researcher",
        role="planner",
        package="aiq_agent.agents.deep_researcher",
        resource_path="prompts/planner.j2",
        relative_path="src/aiq_agent/agents/deep_researcher/prompts/planner.j2",
        description="Creates /shared/plan.json with task analysis, TOC, constraints, and research queries.",
    ),
    PromptSpec(
        id="deep_researcher.researcher",
        agent="deep_researcher",
        role="researcher",
        package="aiq_agent.agents.deep_researcher",
        resource_path="prompts/researcher.j2",
        relative_path="src/aiq_agent/agents/deep_researcher/prompts/researcher.j2",
        description="Guides researcher-agent source gathering and note writing.",
    ),
    PromptSpec(
        id="deep_researcher.orchestrator",
        agent="deep_researcher",
        role="orchestrator",
        package="aiq_agent.agents.deep_researcher",
        resource_path="prompts/orchestrator.j2",
        relative_path="src/aiq_agent/agents/deep_researcher/prompts/orchestrator.j2",
        description="Coordinates planning, researcher delegation, synthesis, citations, and final report writing.",
    ),
    PromptSpec(
        id="deep_researcher.source_registry",
        agent="deep_researcher",
        role="source_registry",
        package="aiq_agent.agents.deep_researcher",
        resource_path="prompts/source_registry.j2",
        relative_path="src/aiq_agent/agents/deep_researcher/prompts/source_registry.j2",
        description="Formats captured verified sources for prompt injection before final citation use.",
    ),
)

_KNOWN_PROMPT_VARIABLES = {
    "available_documents",
    "clarifier_result",
    "current_datetime",
    "research_depth",
    "sandbox_enabled",
    "sandbox_python_packages",
    "skill_sources",
    "skills_enabled",
    "sources",
    "tools",
    "user_info",
}


class PromptTemplate(BaseModel):
    """One prompt template exposed for inspection."""

    id: str = Field(..., description="Stable prompt identifier.")
    agent: str = Field(..., description="Agent package that owns the prompt.")
    role: str = Field(..., description="Runtime role or prompt purpose.")
    description: str = Field(..., description="Human-readable prompt purpose.")
    path: str = Field(..., description="Repository-relative source path.")
    template: str = Field(..., description="Raw Jinja2 prompt template.")
    variables: list[str] = Field(default_factory=list, description="Known runtime variables referenced by template.")
    character_count: int = Field(..., description="Length of the raw prompt template.")


class MissingPrompt(BaseModel):
    """Metadata for a prompt that could not be read."""

    id: str
    path: str
    error: str


class PromptListResponse(BaseModel):
    """List of prompt templates used by the research workflow."""

    count: int
    prompts: list[PromptTemplate]
    missing: list[MissingPrompt] = Field(default_factory=list)


def _repo_root() -> Path:
    """Return the repository root when running from the source tree."""
    return Path(__file__).resolve().parents[5]


def _read_prompt(spec: PromptSpec) -> str:
    """Read a prompt from installed package resources, with a source-tree fallback."""
    try:
        return resources.files(spec.package).joinpath(spec.resource_path).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return (_repo_root() / spec.relative_path).read_text(encoding="utf-8")


def _extract_known_variables(template: str) -> list[str]:
    """Return known Jinja runtime variables referenced by a template."""
    return sorted(variable for variable in _KNOWN_PROMPT_VARIABLES if variable in template)


def get_prompt_templates() -> PromptListResponse:
    """Load all known prompt templates."""
    prompts: list[PromptTemplate] = []
    missing: list[MissingPrompt] = []
    for spec in PROMPT_SPECS:
        try:
            template = _read_prompt(spec)
        except Exception as exc:
            missing.append(MissingPrompt(id=spec.id, path=spec.relative_path, error=str(exc)))
            continue
        prompts.append(
            PromptTemplate(
                id=spec.id,
                agent=spec.agent,
                role=spec.role,
                description=spec.description,
                path=spec.relative_path,
                template=template,
                variables=_extract_known_variables(template),
                character_count=len(template),
            )
        )
    return PromptListResponse(count=len(prompts), prompts=prompts, missing=missing)


def register_prompt_routes(app: FastAPI) -> None:
    """Register prompt-inspection endpoints."""

    @app.get(
        "/prompts",
        response_model=PromptListResponse,
        tags=["prompts"],
        summary="List research prompt templates",
        description="Returns the raw Jinja2 prompt templates used by chat, clarifier, shallow, and deep research.",
    )
    async def list_prompts() -> PromptListResponse:
        return get_prompt_templates()

    @app.get(
        "/v1/prompts",
        response_model=PromptListResponse,
        tags=["prompts"],
        summary="List research prompt templates",
        description="Versioned alias for /prompts.",
    )
    async def list_prompts_v1() -> PromptListResponse:
        return get_prompt_templates()

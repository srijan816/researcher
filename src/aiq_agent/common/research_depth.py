# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Research-depth tiers shared by API, chat routing, and agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal

ResearchDepthTier = Literal["shallow", "medium", "deeper", "deep"]
DEFAULT_RESEARCH_DEPTH: ResearchDepthTier = "deeper"


@dataclass(frozen=True)
class ResearchDepthConfig:
    tier: ResearchDepthTier
    label: str
    source_target: str
    max_researcher_tasks: int
    max_parallel_researcher_tasks: int
    search_calls_per_task: int
    planner_search_limit: int
    advanced_web_search_limit: int
    web_search_limit: int
    stock_quote_limit: int
    planner_guidance: str

    def budget_profile(
        self,
        *,
        mode: Literal["lesson_first", "symmetric", "focused_screen"] = "symmetric",
        section_count: int | None = None,
    ) -> dict[str, Any]:
        """Return a plain dict that describes how the search budget should be spent.

        The deep-research workflow still enforces a single shared search-family
        cap, but the prompts need a clearer spending plan so lesson prompts can
        preserve topic breadth and broad reports can keep their branches more
        evenly distributed.
        """
        total_search_calls = self.advanced_web_search_limit
        if mode == "lesson_first":
            reserve = max(8, round(total_search_calls * 0.20))
            active = max(1, total_search_calls - reserve)
            distribution = {
                "broad_topic_foundations": 0.55,
                "examples_and_cases": 0.25,
                "motion_bridge": 0.15,
                "gap_fill_and_synthesis": 0.05,
            }
            notes = (
                "Spend most search effort on broad-topic foundations and examples first, keep the motion "
                "bridge bounded, and leave a reserve for gap-filling and synthesis."
            )
        elif mode == "focused_screen":
            reserve = max(2, round(total_search_calls * 0.10))
            active = max(1, total_search_calls - reserve)
            distribution = {
                "candidate_screening": 0.70,
                "evidence_and_caveats": 0.20,
                "gap_fill_and_synthesis": 0.10,
            }
            notes = (
                "Use a compact evidence budget: identify candidates quickly, verify current evidence, and "
                "keep a small reserve for caveats or one targeted follow-up."
            )
        else:
            reserve = max(6, round(total_search_calls * 0.15))
            active = max(1, total_search_calls - reserve)
            branch_count = max(3, min(5, section_count or 4))
            even_share = round(1.0 / branch_count, 2)
            distribution = {f"branch_{index + 1}": even_share for index in range(branch_count)}
            distribution["gap_fill_and_synthesis"] = round(reserve / max(1, total_search_calls), 2)
            notes = (
                "Spread research roughly evenly across the main branches. No single branch should consume "
                "most of the search budget while enough reserve remains for gap-filling and synthesis."
            )

        return {
            "mode": mode,
            "total_search_calls": total_search_calls,
            "active_search_calls": active,
            "reserve_search_calls": reserve,
            "planner_search_calls": self.planner_search_limit,
            "search_calls_per_task": self.search_calls_per_task,
            "max_researcher_tasks": self.max_researcher_tasks,
            "max_parallel_researcher_tasks": self.max_parallel_researcher_tasks,
            "distribution": distribution,
            "notes": notes,
        }

    @property
    def prompt_summary(self) -> str:
        return (
            f"{self.label}: target {self.source_target} total sources, use up to "
            f"{self.max_researcher_tasks} researcher tasks, batch at most "
            f"{self.max_parallel_researcher_tasks} at a time, and allow up to "
            f"{self.search_calls_per_task} search calls per researcher task. "
            f"Planner may use up to {self.planner_search_limit} search call(s) only when ambiguity "
            "blocks decomposition. "
            "Keep an explicit reserve for gap-filling and synthesis instead of spending the entire budget "
            "in the early search phase."
        )


RESEARCH_DEPTH_CONFIGS: dict[ResearchDepthTier, ResearchDepthConfig] = {
    "shallow": ResearchDepthConfig(
        tier="shallow",
        label="Shallow",
        source_target="10-20",
        max_researcher_tasks=2,
        max_parallel_researcher_tasks=1,
        search_calls_per_task=8,
        planner_search_limit=1,
        advanced_web_search_limit=20,
        web_search_limit=12,
        stock_quote_limit=4,
        planner_guidance=(
            "Keep the plan compact. Prefer 1-2 packed researcher tasks covering the highest-value angles."
        ),
    ),
    "deeper": ResearchDepthConfig(
        tier="deeper",
        label="Deeper",
        source_target="40-77",
        max_researcher_tasks=5,
        max_parallel_researcher_tasks=3,
        search_calls_per_task=14,
        planner_search_limit=1,
        advanced_web_search_limit=77,
        web_search_limit=24,
        stock_quote_limit=8,
        planner_guidance=(
            "Build a multi-angle plan, but pack related sections into 3-4 researcher tasks. "
            "Use the larger per-task search budget for current evidence, primary sources, "
            "independent analysis, comparisons, and caveats."
        ),
    ),
    "medium": ResearchDepthConfig(
        tier="medium",
        label="Medium",
        source_target="32-64",
        max_researcher_tasks=5,
        max_parallel_researcher_tasks=3,
        search_calls_per_task=12,
        planner_search_limit=1,
        advanced_web_search_limit=64,
        web_search_limit=20,
        stock_quote_limit=8,
        planner_guidance=(
            "Use the deeper evidence budget with a latency-first execution style: keep planning compact, "
            "run bounded researcher tasks in parallel, and reserve the synthesis step for heavier reasoning."
        ),
    ),
    "deep": ResearchDepthConfig(
        tier="deep",
        label="Deep",
        source_target="90-150+",
        max_researcher_tasks=10,
        max_parallel_researcher_tasks=3,
        search_calls_per_task=14,
        planner_search_limit=2,
        advanced_web_search_limit=140,
        web_search_limit=48,
        stock_quote_limit=16,
        planner_guidance=(
            "Build an exhaustive plan, but consolidate related queries into up to 8 broad researcher "
            "tasks. Use the larger per-task search budget to cover primary sources, recent updates, "
            "benchmarks, pricing, counterarguments, history, and edge cases."
        ),
    ),
}


def normalize_research_depth(value: Any) -> ResearchDepthTier:
    """Return a supported research depth tier, defaulting to the balanced tier."""
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "quick": "shallow",
            "light": "shallow",
            "standard": "medium",
            "balanced": "deeper",
            "comprehensive": "deep",
            "maximum": "deep",
            "max": "deep",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in RESEARCH_DEPTH_CONFIGS:
            return normalized  # type: ignore[return-value]
    return DEFAULT_RESEARCH_DEPTH


def get_research_depth_config(value: Any) -> ResearchDepthConfig:
    """Return the tier configuration for arbitrary user/API input."""
    return RESEARCH_DEPTH_CONFIGS[normalize_research_depth(value)]

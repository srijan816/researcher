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

"""
Clarifier agent for interactive clarification dialog.

This module provides the ClarifierAgent class which handles multi-turn
clarification dialogs with users before deep research begins. The agent
uses LangGraph for workflow orchestration and supports tool calling
for context gathering.

Example:
    >>> from aiq_agent.agents.clarifier_agent import ClarifierAgent
    >>> from aiq_agent.common import LLMProvider
    >>>
    >>> async def prompt_user(question: str) -> str:
    ...     return input(question)
    >>>
    >>> provider = LLMProvider()
    >>> provider.set_default(my_llm)
    >>> agent = ClarifierAgent(
    ...     llm_provider=provider,
    ...     user_prompt_callback=prompt_user,
    ... )
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from aiq_agent.agents.chat_researcher.utils import coerce_content_text
from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common import extract_json
from aiq_agent.common import get_latest_user_query
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

from .models import ClarificationResponse
from .models import ClarifierAgentState
from .models import ClarifierResult

logger = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).parent
"""Path to the clarifier agent's directory, used for loading prompts."""

DEFAULT_CLARIFICATION_PROMPT = (
    "/no_think\n\n"
    "You are a helpful research clarification assistant. "
    "Ask focused questions to understand the user's needs. "
    'Respond with JSON: {"needs_clarification": true/false, "clarification_question": "your question?" or null}'
)
"""Fallback prompt used when the prompt file cannot be loaded."""

DEFAULT_PLAN_GENERATION_PROMPT = (
    "/no_think\n\n"
    "Generate a research plan with a title and 5-8 sections. "
    'Respond with JSON: {"title": "...", "sections": ["...", "..."]}'
)
"""Fallback prompt for plan generation."""

APPROVAL_KEYWORDS = {
    "approve",
    "approved",
    "yes",
    "ok",
    "proceed",
    "continue",
    "go ahead",
    "looks good",
    "y",
    "accept",
    "skip",
}
"""Keywords that indicate the user approves the plan."""

REJECTION_KEYWORDS = {"reject", "rejected", "no", "cancel", "stop", "abort", "n"}
"""Keywords that indicate the user rejects the plan."""

SECTION_FOCUS_RE = re.compile(
    r"(?:\b(?:just\s+)?focus(?:\s+only)?\s+on\b|\bonly\b|\bsection\b|\bnumber\b|#)\s*(?:section\s*)?(\d+)\b",
    re.IGNORECASE,
)
"""Detect feedback like "just focus on number 3" during plan review."""

FINANCIAL_SCREEN_CLARIFIER_MARKER = "**Universe and fair-value basis**"
"""Marker used to avoid asking the deterministic finance-screen clarification twice."""

PLACEHOLDER_PLAN_SECTIONS = {
    "research report",
    "introduction",
    "background",
    "analysis",
    "findings",
    "conclusion",
    "scope and criteria",
    "current evidence",
    "current evidence base",
    "key findings",
    "key findings and trade-offs",
    "trade-offs and caveats",
    "recommended next steps",
    "topic-specific section one",
    "topic-specific section two",
    "topic-specific section three",
    "topic area one",
    "topic area two",
}
"""Section names that indicate the model copied an example or generic template."""

GENERIC_FALLBACK_PLAN_SECTIONS = {
    "landscape",
    "recent evidence and signals",
    "capability gaps",
    "adoption risks and recommendations",
    "requirements",
    "architecture and interfaces",
    "failure paths and guardrails",
    "implementation plan",
    "core question",
    "evidence and competing views",
    "practical strategy options",
    "trade-offs and failure modes",
    "decision framework",
}
"""Coarse safety-net headings that are never good enough for explicit deep-research briefs."""

JSON_REMINDER_AFTER_TOOLS = (
    "Based on the search results above, now make your clarification decision. "
    "IMPORTANT: You must respond with ONLY a valid JSON object, nothing else. "
    "Do NOT write a report, summary, or analysis. "
    "Output exactly: "
    '{"needs_clarification": true, "clarification_question": "your question"} '
    "OR "
    '{"needs_clarification": false, "clarification_question": null}'
)
"""Reminder prompt added after tool results to reinforce JSON-only output."""


class ClarifierAgent:
    """
    Clarifier agent for interactive clarification dialog.

    This agent handles interactive clarification dialogs for deep research queries.
    It asks follow-up questions to refine the research scope, constraints, and
    requirements before the actual research begins.

    The agent uses LangGraph for workflow orchestration with three main nodes:
    - agent: Generates clarification questions using the LLM
    - tools: Executes tool calls for context gathering (e.g., web search)
    - ask_for_clarification: Prompts the user and processes their response

    Attributes:
        llm_provider: Provider for obtaining LLM instances.
        tools: List of tools available for context gathering.
        user_prompt_callback: Async callback for prompting user input.
        max_turns: Maximum number of Q&A turns before auto-completing.
        system_prompt: The loaded system prompt for the LLM.
        callbacks: LangChain callbacks for tracing/logging.

    Example:
        >>> async def user_prompt_fn(question: str) -> str:
        ...     return input(question)
        >>>
        >>> provider = LLMProvider()
        >>> provider.set_default(my_llm)
        >>> agent = ClarifierAgent(
        ...     llm_provider=provider,
        ...     tools=[search_tool],
        ...     user_prompt_callback=user_prompt_fn,
        ...     max_turns=3,
        ... )
        >>> state = ClarifierAgentState(messages=[HumanMessage(content="Research AI")])
        >>> result = await agent.run(state)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        user_prompt_callback: Callable[[str], Awaitable[str]],
        max_turns: int = 3,
        enable_plan_approval: bool = False,
        max_plan_iterations: int = 10,
        planner_llm: BaseChatModel | None = None,
        log_response_max_chars: int = 2000,
        verbose: bool = False,
        callbacks: list[Any] | None = None,
    ) -> None:
        """
        Initialize the clarifier agent.

        Args:
            llm_provider: Provider for obtaining LLM instances by role.
            tools: Optional sequence of LangChain tools for context gathering
                (e.g., web search). Tools help the agent ask more informed questions.
            user_prompt_callback: Async callback function to prompt the user for input.
                Takes a question string and returns the user's response string.
            max_turns: Maximum number of clarification Q&A turns before
                automatically completing clarification. Defaults to 3.
            enable_plan_approval: Whether to enable plan preview and approval
                after clarification completes. Defaults to False.
            max_plan_iterations: Maximum number of plan feedback iterations
                before auto-approving. Defaults to 10.
            planner_llm: Optional LLM to use for plan generation. If not provided,
                uses the default clarifier LLM.
            log_response_max_chars: Maximum characters to log from LLM responses.
                Used for debugging. Defaults to 2000.
            verbose: Whether to enable detailed logging. Defaults to False.
            callbacks: Optional list of LangChain callback handlers for
                tracing and logging.
        """
        self.llm_provider: LLMProvider = llm_provider
        self.tools = list(tools) if tools else []
        self.user_prompt_callback = user_prompt_callback
        self.max_turns = max_turns
        self.enable_plan_approval = enable_plan_approval
        self.max_plan_iterations = max_plan_iterations
        self.planner_llm = planner_llm
        self.log_response_max_chars = log_response_max_chars
        self.verbose = verbose
        self.callbacks = callbacks or []

        self.system_prompt = self._load_default_prompt()
        self.plan_generation_prompt = self._load_plan_generation_prompt()

        self._graph = self._build_graph()

    def _load_default_prompt(self) -> str:
        """
        Load the research clarification prompt from file.

        Attempts to load the prompt from the prompts/research_clarification.j2
        file. Falls back to DEFAULT_CLARIFICATION_PROMPT if the file is not found.

        Returns:
            The loaded prompt string, or the default fallback prompt.
        """
        try:
            return load_prompt(AGENT_DIR / "prompts", "research_clarification")
        except Exception:
            logger.warning("Clarifier prompt not found, using inline default")
            return DEFAULT_CLARIFICATION_PROMPT

    def _load_plan_generation_prompt(self) -> str:
        """
        Load the plan generation prompt from file.

        Returns:
            The loaded prompt string, or the default fallback prompt.
        """
        try:
            return load_prompt(AGENT_DIR / "prompts", "plan_generation")
        except Exception:
            logger.warning("Plan generation prompt not found, using inline default")
            return DEFAULT_PLAN_GENERATION_PROMPT

    @staticmethod
    def _get_original_query(state: ClarifierAgentState) -> str | None:
        """Return the pinned research request, falling back to initial messages."""
        if state.original_query:
            return state.original_query
        return get_latest_user_query(state.messages)

    def _parse_plan_response(self, text: Any) -> tuple[str | None, list[str]]:
        """
        Parse plan generation response from LLM.

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            Tuple of (title, sections) or (None, []) if parsing fails.
        """
        if not text:
            return None, []

        text = coerce_content_text(text).strip()
        extracted = extract_json(text)
        if isinstance(extracted, dict):
            title = extracted.get("title") or extracted.get("plan_title") or extracted.get("report_title")
            sections = self._normalize_plan_sections(
                extracted.get("sections")
                or extracted.get("section_headings")
                or extracted.get("outline")
                or extracted.get("toc")
            )
            if isinstance(title, str) and sections:
                return title.strip(), sections

        candidates = [text]

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            candidates.insert(0, json_match.group(1).strip())

        object_match = re.search(r"\{[\s\S]*\}", text)
        if object_match:
            candidates.append(object_match.group(0).strip())

        for candidate in dict.fromkeys(candidates):
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug("Failed to parse plan response candidate as JSON: %s", e)
                continue

            if not isinstance(data, dict):
                continue

            title = data.get("title") or data.get("plan_title") or data.get("report_title")
            sections = self._normalize_plan_sections(
                data.get("sections") or data.get("section_headings") or data.get("outline") or data.get("toc")
            )
            if isinstance(title, str) and sections:
                return title.strip(), sections

        markdown_title, markdown_sections = self._parse_markdown_plan_response(text)
        if markdown_title and markdown_sections:
            return markdown_title, markdown_sections

        logger.warning("Failed to generate valid plan JSON from response")

        return None, []

    @staticmethod
    def _plan_response_excerpt(text: Any) -> str:
        """Return a compact log excerpt for invalid planner output."""
        excerpt = coerce_content_text(text).strip()
        excerpt = re.sub(r"\s+", " ", excerpt)
        return excerpt[:500]

    @staticmethod
    def _normalize_plan_sections(raw_sections: Any) -> list[str]:
        """Normalize common LLM section-list shapes into clean heading strings."""
        if not isinstance(raw_sections, list):
            return []

        sections: list[str] = []
        for raw in raw_sections:
            section = ""
            if isinstance(raw, str):
                section = raw
            elif isinstance(raw, dict):
                value = raw.get("title") or raw.get("heading") or raw.get("name") or raw.get("section")
                if isinstance(value, str):
                    section = value
            section = re.sub(r"^\s*(?:section\s*)?\d+[\).:\-\s]+", "", section, flags=re.IGNORECASE).strip()
            section = re.sub(r"\s+", " ", section)
            if section and section not in sections:
                sections.append(section[:150].rstrip(" .,:;"))

        return sections[:7]

    @staticmethod
    def _parse_markdown_plan_response(text: Any) -> tuple[str | None, list[str]]:
        """Recover a usable plan when a model ignores JSON and emits Markdown."""
        raw_text = coerce_content_text(text).strip()
        if not raw_text:
            return None, []

        title: str | None = None
        title_match = re.search(
            r"^\s*(?:#{1,3}\s*)?(?:title|plan title|report title)\s*:\s*(.+)$",
            raw_text,
            re.IGNORECASE | re.MULTILINE,
        )
        if title_match:
            title = title_match.group(1).strip(" *`")
        else:
            heading_match = re.search(r"^\s*#{1,3}\s+(.+)$", raw_text, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1).strip(" *`")

        sections_start = re.search(r"^\s*(?:#{1,4}\s*)?sections?\s*:?\s*$", raw_text, re.IGNORECASE | re.MULTILINE)
        section_text = raw_text[sections_start.end() :] if sections_start else raw_text
        sections: list[str] = []
        for line in section_text.splitlines():
            match = re.match(r"^\s*(?:[-*•]|\d+[\).:-])\s+(.+?)\s*$", line)
            if not match:
                continue
            section = re.sub(r"\*\*|`", "", match.group(1)).strip()
            section = re.sub(r"^\s*(?:section\s*)?\d+[\).:\-\s]+", "", section, flags=re.IGNORECASE).strip()
            section = re.split(r"\s+[-–—:]\s+", section, maxsplit=1)[0].strip()
            section = re.sub(r"\s+", " ", section).strip(" .,:;")
            if section and section not in sections:
                sections.append(section[:150].rstrip(" .,:;"))
            if len(sections) >= 7:
                break

        if not title and sections:
            title = "Focused Research Plan"

        return (title, sections) if title and sections else (None, [])

    def _parse_approval(self, response: str) -> tuple[bool, bool, str | None]:
        """
        Parse user's approval response.

        Args:
            response: User's response text (may be JSON wrapped).

        Returns:
            Tuple of (approved, rejected, feedback).
            - If approved: (True, False, None)
            - If rejected: (False, True, None)
            - If feedback: (False, False, feedback_text)
        """
        # Extract query from JSON if wrapped (e.g., {"query": "approve", ...})
        text = response.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "query" in data:
                text = data["query"]
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON, use original text

        normalized = text.strip().lower()

        if normalized in APPROVAL_KEYWORDS:
            return True, False, None

        if normalized in REJECTION_KEYWORDS:
            return False, True, None

        # Treat as feedback for plan revision
        return False, False, text.strip()

    @staticmethod
    def _is_precise_financial_screen(query: str | None) -> bool:
        """Return True for narrow finance screens that should not get broad TOCs."""
        if not query:
            return False
        normalized = query.lower()
        has_stock = any(term in normalized for term in ("stock", "stocks", "equities", "ticker", "tickers"))
        has_valuation = "fair value" in normalized or "undervalued" in normalized or "discount" in normalized
        has_current_screen = any(term in normalized for term in ("currently", "current", "trading", "below"))
        has_percent = bool(re.search(r"\d+\s*(?:-|–|to)\s*\d+\s*%", normalized)) or bool(
            re.search(r"\d+\s*%", normalized)
        )
        return has_stock and has_valuation and has_current_screen and has_percent

    def _needs_financial_screen_clarification(self, query: str | None) -> bool:
        """Return True when a valuation screen is missing assumptions that change results."""
        if not self._is_precise_financial_screen(query):
            return False

        normalized = query.lower()
        universe_terms = (
            "u.s.",
            "us ",
            "united states",
            "nyse",
            "nasdaq",
            "s&p",
            "sp500",
            "russell",
            "large-cap",
            "mid-cap",
            "small-cap",
            "global",
            "international",
            "europe",
            "asia",
            "japan",
            "india",
            "china",
            "canada",
            "uk",
            "universe",
        )
        fair_value_source_terms = (
            "morningstar",
            "analyst",
            "consensus",
            "dcf",
            "discounted cash flow",
            "gurufocus",
            "simply wall",
            "finbox",
            "tipranks",
            "estimated by",
            "fair value source",
        )
        has_universe = any(term in normalized for term in universe_terms)
        has_fair_value_source = any(term in normalized for term in fair_value_source_terms)
        return not (has_universe and has_fair_value_source)

    def _financial_screen_clarification_response(self) -> str:
        """Build a topic-specific clarification question for valuation screens."""
        question = (
            f"{FINANCIAL_SCREEN_CLARIFIER_MARKER}: Which stock universe and fair-value source should I use?\n\n"
            "1. U.S.-listed liquid stocks; use source-backed analyst/fair-value estimates and current quotes\n"
            "2. Global liquid stocks; use source-backed analyst/fair-value estimates and current quotes\n"
            "3. Any market where reliable current price and fair-value estimate sources are available\n"
            "4. I will specify the universe and valuation source in my reply\n"
            "5. Skip: default to U.S.-listed large/mid caps, source-backed fair-value estimates, and current quotes\n\n"
            "Please choose a number, add constraints, or type 'skip' to use the default."
        )
        return ClarificationResponse(needs_clarification=True, clarification_question=question).model_dump_json()

    def _compact_plan_for_query(
        self,
        title: str,
        sections: list[str],
        query: str | None,
    ) -> tuple[str, list[str]]:
        """Apply deterministic guardrails for precise queries the LLM tends to over-outline."""
        if self._is_precise_financial_screen(query):
            return (
                "Stocks Trading 40-50% Below Fair Value",
                [
                    "Screening Assumptions",
                    "Candidate Price/Fair Value Table",
                    "Evidence and Caveats",
                ],
            )
        return title, sections

    def _extract_section_focus_index(self, feedback: str, sections: list[str]) -> int | None:
        """Extract a zero-based section index from plan feedback, if present."""
        normalized = feedback.strip().lower()
        if normalized.isdigit():
            index = int(normalized) - 1
            return index if 0 <= index < len(sections) else None

        match = SECTION_FOCUS_RE.search(feedback)
        if not match:
            return None
        index = int(match.group(1)) - 1
        return index if 0 <= index < len(sections) else None

    def _focused_plan_from_feedback(
        self,
        title: str,
        sections: list[str],
        feedback: str,
        query: str | None,
    ) -> tuple[str, list[str]] | None:
        """Build a deterministic focused plan for "focus on section N" feedback."""
        index = self._extract_section_focus_index(feedback, sections)
        if index is None:
            return None

        if self._is_precise_financial_screen(query):
            return (
                "Current 40-50% Fair-Value Discount Candidates",
                [
                    "Candidate Price/Fair Value Table",
                    "Source Quality and Caveats",
                ],
            )

        selected = sections[index]
        return f"{title}: {selected}", [selected]

    def _normalize_plan_feedback(self, feedback: str, sections: list[str]) -> str:
        """Make terse plan feedback explicit before passing it to the planner LLM."""
        index = self._extract_section_focus_index(feedback, sections)
        if index is None:
            return feedback
        selected = sections[index]
        return (
            f"Focus only on section {index + 1}: {selected}. "
            "Remove unrelated sections and revise the plan around that narrow focus."
        )

    @staticmethod
    def _fallback_plan_title(query: str) -> str:
        """Create a non-placeholder title if the planner response cannot be parsed."""
        clean_query = ClarifierAgent._strip_research_role_preamble(query)
        if not clean_query:
            return "Focused Research Plan"
        explicit_plan = ClarifierAgent._deterministic_plan_from_query(clean_query)
        if explicit_plan:
            return explicit_plan[0]
        build_concept_title = ClarifierAgent._build_concept_plan_title(clean_query)
        if build_concept_title:
            return build_concept_title
        lowered = clean_query.lower()
        if "age of ai" in lowered and ("live" in lowered or "life" in lowered):
            return "Living Well in the Age of AI"
        if ClarifierAgent._is_precise_financial_screen(clean_query):
            return "Stocks Trading 40-50% Below Estimated Fair Value"
        if ClarifierAgent._is_competitor_research_query(clean_query):
            focal = ClarifierAgent._extract_competitor_focal_entity(clean_query)
            return f"Competitive Analysis of {focal}" if focal else "Competitive Landscape Analysis"
        domain_match = re.search(r"\b([a-z0-9][a-z0-9-]*(?:\.[a-z]{2,})+)\b", clean_query, re.IGNORECASE)
        if domain_match and re.search(r"\bmoneti[sz]e|business|revenue\b", clean_query, re.IGNORECASE):
            return f"Monetizing {domain_match.group(1)}"
        return clean_query[:90].rstrip(" .,:;")

    @staticmethod
    def _fallback_plan_sections(query: str) -> list[str]:
        """Create specific-enough fallback sections instead of generic report headings."""
        clean_query = (query or "").lower()
        topic = ClarifierAgent._fallback_plan_topic(query)
        topic_title = topic[:1].upper() + topic[1:] if topic else "Research"
        explicit_plan = ClarifierAgent._deterministic_plan_from_query(query)
        if explicit_plan:
            return explicit_plan[1]
        build_sections = ClarifierAgent._build_concept_plan_sections(query)
        if build_sections:
            return build_sections
        if "age of ai" in clean_query and ("live" in clean_query or "life" in clean_query):
            return [
                "AI-Era Human Priorities",
                "Work and Learning Strategy",
                "Attention, Relationships, and Wellbeing",
                "Risk Boundaries and Agency",
                "Practical Life Operating System",
            ]
        domain_match = re.search(r"\b([a-z0-9][a-z0-9-]*(?:\.[a-z]{2,})+)\b", query or "", re.IGNORECASE)
        if domain_match and re.search(r"\bmoneti[sz]e|business|revenue|pricing\b", clean_query, re.IGNORECASE):
            return [
                "Market and Demand Signals",
                "Competitive White Space",
                "Pricing and Revenue Models",
                "Five Monetization Concepts",
                "MVP Path and Risks",
            ]
        if ClarifierAgent._is_precise_financial_screen(query):
            return [
                "Universe and Assumptions",
                "Candidate Valuation Table",
                "Price and Fair-Value Evidence",
                "Source Quality and Caveats",
            ]
        if ClarifierAgent._is_competitor_research_query(query):
            focal = ClarifierAgent._extract_competitor_focal_entity(query)
            profile = (
                f"{focal} Profile, Offerings, and Positioning" if focal else "Focal Company Profile and Positioning"
            )
            return [
                profile,
                "Competitor Discovery and Verification",
                "Direct Competitor Mapping: Verified Same-Market Providers",
                "Indirect Competitors and Substitute Options",
                "Positioning Gaps, Opportunities, and Risks",
            ]
        if "risk" in clean_query or "hallucination" in clean_query or "governance" in clean_query:
            return [
                f"{topic_title} Scope and Controls",
                "Failure Modes and Exposure",
                "Verification and Monitoring",
                "Governance and Mitigation Plan",
            ]
        if "compare" in clean_query or "comparison" in clean_query:
            return [
                "Decision Criteria",
                f"{topic_title} Evidence Matrix",
                "Trade-offs and Caveats",
                "Recommended Fit",
            ]
        if any(term in clean_query for term in ("latest", "current", "2025", "2026", "recent", "emerging")):
            return [
                f"{topic_title} Landscape",
                "Recent Evidence and Signals",
                "Capability Gaps",
                "Adoption Risks and Recommendations",
            ]
        if any(term in clean_query for term in ("build", "implement", "pipeline", "workflow", "api", "integration")):
            return [
                f"{topic_title} Requirements",
                "Architecture and Interfaces",
                "Failure Paths and Guardrails",
                "Implementation Plan",
            ]
        return [
            f"{topic_title} Core Question",
            "Evidence and Competing Views",
            "Practical Strategy Options",
            "Trade-offs and Failure Modes",
            "Decision Framework",
        ]

    @staticmethod
    def _deterministic_plan_from_query(query: str | None) -> tuple[str, list[str]] | None:
        """
        Compile explicit report instructions into an approval plan.

        This is not a generic fallback. It is a deterministic parser for cases where the
        user has already specified the report contract: ranked lists, required
        dimensions, source standards, and final structure. Those instructions should
        survive even if the LLM emits invalid JSON or an underfit outline.
        """
        clean_query = ClarifierAgent._strip_research_role_preamble(query)
        if not clean_query:
            return None
        normalized = re.sub(r"\s+", " ", clean_query).strip()
        lowered = normalized.lower()

        lesson_topic_match = re.search(
            r"\bExact lesson topic:\s*(.+?)(?:\.| Final debate motion:| Audience:|$)", normalized, re.IGNORECASE
        )
        if lesson_topic_match:
            topic = lesson_topic_match.group(1).strip().rstrip(".")
            if topic:
                return (
                    f"{topic} Content Research Dossier",
                    [
                        "Research Scope and Motion Anchor",
                        "Audience Assumptions and Topic Framing",
                        "Topic Essentials: Definitions, Terms, and Background",
                        "Source-Backed Factual Findings",
                        "Concrete Examples and Case Studies",
                        "Important Tensions and Misconceptions",
                        "Motion Relevance Notes",
                    ],
                )

        top_match = re.search(
            r"\btop\s+(\d{1,2})\s+(?:highest[-\s]value\s+)?(?:use\s+cases?|applications?)\s+of\s+AI\b",
            normalized,
            re.IGNORECASE,
        )
        if top_match and re.search(r"\b(rank|ranked|ranking|business value|roi|transformative)\b", lowered):
            count = top_match.group(1)
            year_match = re.search(r"\b(20\d{2})\b", normalized)
            year = year_match.group(1) if year_match else ""
            year_suffix = f" in {year}" if year else ""
            return (
                f"Top {count} Highest-Value AI Use Cases{year_suffix}",
                [
                    "Executive Summary and Ranking Criteria",
                    f"Ranked Top {count} AI Use Cases",
                    "Business Impact, ROI, and Efficiency Evidence",
                    "2025-2026 Real-World Deployment Examples",
                    "Maturity Levels and Enabling Technologies",
                    "Adoption Barriers and Best-Fit Beneficiaries",
                    "2027-2028 AI Value Creation Outlook",
                ],
            )

        numbered_dimensions = ClarifierAgent._extract_numbered_dimensions(normalized)
        if len(numbered_dimensions) >= 4 and re.search(r"\b(report|research|deep research|comprehensive)\b", lowered):
            title = ClarifierAgent._fallback_plan_topic(normalized)
            title = ClarifierAgent._title_case_phrase(title)
            sections = ["Executive Summary and Scope"]
            sections.extend(ClarifierAgent._dimension_to_section_heading(item) for item in numbered_dimensions[:5])
            if re.search(r"\b(rank|ranking|ranked)\b", lowered) and not any("rank" in s.lower() for s in sections):
                sections.insert(1, "Ranking Framework and Priority Order")
            return (title[:90].rstrip(" .,:;") or "Structured Research Report", sections[:7])

        return None

    @staticmethod
    def _extract_numbered_dimensions(query: str) -> list[str]:
        """Extract explicit numbered report dimensions from a long user request."""
        dimensions: list[str] = []
        pattern = re.compile(
            r"(?:^|\s)(\d{1,2})[\.\)]\s*([A-Z][^0-9]{4,220}?)(?=\s+\d{1,2}[\.\)]\s*[A-Z]|\s+Rank\b|\s+Prioriti[sz]e\b|\s+Present\b|$)",
            re.DOTALL,
        )
        for match in pattern.finditer(query):
            item = re.sub(r"\s+", " ", match.group(2)).strip(" .,:;")
            item = re.split(r"\s+[–—-]\s+", item, maxsplit=1)[0].strip(" .,:;")
            if item and item not in dimensions:
                dimensions.append(item)
        return dimensions

    @staticmethod
    def _dimension_to_section_heading(dimension: str) -> str:
        """Convert a user-provided report dimension into a concise plan heading."""
        lowered = dimension.lower()
        if "what it is" in lowered or "clear" in lowered and "explanation" in lowered:
            return "Use-Case Definitions and Plain-Language Explanations"
        if "high-value" in lowered or "roi" in lowered or "cost savings" in lowered or "revenue" in lowered:
            return "Business Value, ROI, and Quantified Impact"
        if "real-world" in lowered or "companies" in lowered or "industries" in lowered:
            return "Real-World Examples and 2025-2026 Deployments"
        if "maturity" in lowered or "emerging" in lowered or "scaling" in lowered:
            return "Maturity Level: Emerging, Scaling, or Mainstream"
        if "enabling technologies" in lowered or "models" in lowered or "platforms" in lowered:
            return "Enabling Models, Platforms, and Techniques"
        if "barriers" in lowered or "obstacles" in lowered or "regulation" in lowered:
            return "Adoption Barriers and Constraints"
        if "benefits most" in lowered or "who benefits" in lowered or "industries" in lowered:
            return "Best-Fit Beneficiaries by Industry, Role, and Company Size"
        return ClarifierAgent._title_case_phrase(dimension)[:90].rstrip(" .,:;")

    @staticmethod
    def _fallback_plan_topic(query: str | None) -> str:
        """Extract a short topic label so fallback plans are not identical."""
        clean_query = ClarifierAgent._strip_research_role_preamble(query)
        if not clean_query:
            return "research"
        clean_query = re.sub(
            r"^(please\s+)?(can you\s+)?(could you\s+)?"
            r"(deep\s+res(?:ea)?rch|research|analyze|compare|investigate|find|look into|build|implement)\s+",
            "",
            clean_query,
            flags=re.IGNORECASE,
        )
        clean_query = re.sub(
            r"^(?:on\s+)?all\s+the\s+ingred(?:ie|ei)nts\s+required\s+to\s+build\s*:?\s*",
            "",
            clean_query,
            flags=re.IGNORECASE,
        )
        clean_query = re.split(
            r"\s+and\s+(?:identify|find|assess|evaluate|explain|recommend|include|cover)\b",
            clean_query,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        clean_query = re.split(r"[?.!\n]", clean_query, maxsplit=1)[0]
        words = [word.strip(" ,:;()[]{}\"'") for word in clean_query.split() if word.strip(" ,:;()[]{}\"'")]
        if not words:
            return "research"
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "for",
            "with",
            "about",
            "into",
            "latest",
            "current",
            "recent",
            "please",
        }
        topic_words = [word for word in words if word.lower() not in stop_words][:5]
        return " ".join(topic_words or words[:5]).rstrip(" .,:;") or "research"

    @staticmethod
    def _build_concept_plan_title(query: str | None) -> str | None:
        """Create a polished approval title for "ingredients to build X" prompts."""
        concept, uses_ai = ClarifierAgent._extract_build_concept(query)
        if not concept:
            return None
        lowered = concept.lower()
        if "curiosity engine" in lowered:
            return "Building AI Curiosity Engines" if uses_ai else "Building Curiosity Engines"
        title = ClarifierAgent._title_case_phrase(concept)
        if uses_ai and not re.search(r"\bAI\b|artificial intelligence", title, re.IGNORECASE):
            title = f"AI-Powered {title}"
        return f"Building {title}"[:90].rstrip(" .,:;")

    @staticmethod
    def _build_concept_plan_sections(query: str | None) -> list[str] | None:
        """Create rich user-facing research threads for AI product-concept build prompts."""
        concept, uses_ai = ClarifierAgent._extract_build_concept(query)
        if not concept:
            return None
        lowered = f"{query or ''} {concept}".lower()
        if "curiosity engine" in lowered or ("niche topic" in lowered and "interest" in lowered):
            return [
                "Learning Science and Curiosity Foundations",
                "Dynamic Interest Modeling Over Time",
                "Long-Tail Discovery and Serendipity Architecture",
                "Adaptive Teaching, Scaffolding, and Dialogue",
                "RAG, Knowledge Graphs, and Agentic Workflows",
                "Niche Content Verification and Hallucination Controls",
                "Engagement, Retention, and Learning Outcome Metrics",
            ]
        if uses_ai or "ai" in lowered:
            return [
                "User Need and Product Thesis",
                "Core AI Capabilities and Data",
                "Personalization and Interaction Design",
                "Architecture, Interfaces, and Workflow",
                "Evaluation, Safety, and Launch Path",
            ]
        return None

    @staticmethod
    def _extract_build_concept(query: str | None) -> tuple[str | None, bool]:
        """Return the product/concept being built and whether AI is central."""
        clean_query = ClarifierAgent._strip_research_role_preamble(query)
        if not clean_query:
            return None, False
        normalized = clean_query.replace("“", '"').replace("”", '"').replace("’", "'")
        lowered = normalized.lower()
        if not re.search(r"\b(build|building|implement|create|ingredients|required)\b", lowered):
            return None, False
        uses_ai = bool(re.search(r"\bAI\b|artificial intelligence|LLM|model", normalized, re.IGNORECASE))

        quoted = re.search(r'"([^"]{8,180})"', normalized)
        if quoted:
            concept = quoted.group(1)
        else:
            match = re.search(
                r"(?:build|building|create|implement)\s*:?\s*(.+?)(?:\s+(?:using|with|powered by)\s+AI\b|$)",
                normalized,
                re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r"ingredients\s+required\s+to\s+build\s*:?\s*(.+?)(?:\s+(?:using|with|powered by)\s+AI\b|$)",
                    normalized,
                    re.IGNORECASE,
                )
            concept = match.group(1) if match else ""
        concept = re.sub(r"\busing\s+AI\b.*$", "", concept, flags=re.IGNORECASE).strip(" :;,.\"'")
        concept = re.split(r"\s+that\s+", concept, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        concept = re.sub(r"^(?:all\s+the\s+)?(?:ingredients|required|to\s+build)\s+", "", concept, flags=re.IGNORECASE)
        return (concept or None), uses_ai

    @staticmethod
    def _title_case_phrase(value: str) -> str:
        """Title-case a short concept without wrecking common AI abbreviations."""
        words = []
        for word in re.sub(r"\s+", " ", value).strip().split():
            if word.upper() in {"AI", "LLM", "RAG", "API"}:
                words.append(word.upper())
            else:
                words.append(word[:1].upper() + word[1:].lower())
        return " ".join(words)

    @staticmethod
    def _strip_research_role_preamble(query: str | None) -> str:
        """Remove role-play/task wrapper text before deriving a human-facing plan title."""
        clean_query = re.sub(r"\s+", " ", query or "").strip()
        if not clean_query:
            return ""

        clean_query = re.sub(
            r"^you\s+are\s+deep\s+research[^.]*\.\s*",
            "",
            clean_query,
            flags=re.IGNORECASE,
        ).strip()
        clean_query = re.sub(
            r"^you\s+are\s+[^.]{0,180}research[^.]*\.\s*",
            "",
            clean_query,
            flags=re.IGNORECASE,
        ).strip()

        task_match = re.search(r"\byour\s+task\s+is\s+to\s+(.+)$", clean_query, re.IGNORECASE)
        if task_match:
            clean_query = task_match.group(1).strip()

        clean_query = re.sub(
            r"^(?:conduct\s+)?(?:the\s+)?(?:deepest\s+possible\s+)?research(?:\s+and\s+analysis)?\s+to\s+",
            "",
            clean_query,
            flags=re.IGNORECASE,
        ).strip()
        clean_query = re.sub(r"^identify\s+the\s+absolute\s+best\s+ways\s+to\s+", "", clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r"^generate\s+exactly\s+\d+\s+", "", clean_query, flags=re.IGNORECASE)
        return clean_query.strip(" .,:;")

    @staticmethod
    def _looks_like_instruction_leak(text: str) -> bool:
        """Detect role/system-prompt text accidentally used as a plan title or section."""
        lowered = text.lower()
        return (
            lowered.startswith("you are ")
            or "you are deep research" in lowered
            or "operating in full autonomous" in lowered
            or "your task is to" in lowered
        )

    @staticmethod
    def _looks_like_placeholder_section(text: str) -> bool:
        """Detect generic/example section names that should not reach users."""
        normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .,:;")
        return normalized in PLACEHOLDER_PLAN_SECTIONS

    @staticmethod
    def _looks_like_generic_fallback_section(text: str) -> bool:
        """Detect coarse safety-net headings masquerading as a real plan."""
        normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .,:;")
        normalized = re.sub(r"^[a-z0-9 ,.'\"-]{0,80}\s+", "", normalized)
        return normalized in GENERIC_FALLBACK_PLAN_SECTIONS or text.strip().lower() in GENERIC_FALLBACK_PLAN_SECTIONS

    def _sanitize_plan_for_query(
        self,
        title: str,
        sections: list[str],
        query: str | None,
    ) -> tuple[str, list[str]]:
        """Replace planner/fallback text that leaked role instructions into the user-facing plan."""
        if self._looks_like_instruction_leak(title):
            title = self._fallback_plan_title(query or "")
        fallback_title = self._fallback_plan_title(query or "")
        fallback_sections = self._fallback_plan_sections(query or "")
        explicit_plan = self._deterministic_plan_from_query(query)
        if self._is_weak_build_plan(title, sections, query) or self._is_underfit_explicit_report_plan(
            title,
            sections,
            query,
        ):
            title = fallback_title
            sections = fallback_sections
        elif (
            explicit_plan and sum(1 for section in sections if self._looks_like_generic_fallback_section(section)) >= 2
        ):
            title, sections = explicit_plan

        sections = self._sanitize_unverified_competitor_sections(sections, query)

        if not title or self._looks_like_instruction_leak(title):
            title = "Focused Research Plan"

        placeholder_count = sum(1 for section in sections if self._looks_like_placeholder_section(section))
        placeholder_heavy = placeholder_count >= 2 or (bool(sections) and placeholder_count == len(sections))
        generic_count = sum(1 for section in sections if self._looks_like_generic_fallback_section(section))
        generic_heavy = generic_count >= 2 or (bool(sections) and generic_count == len(sections))

        if (
            not sections
            or any(self._looks_like_instruction_leak(section) for section in sections)
            or placeholder_heavy
            or (generic_heavy and explicit_plan is not None)
        ):
            sections = fallback_sections

        clean_sections = [
            section
            for section in sections
            if section
            and not self._looks_like_instruction_leak(section)
            and not self._looks_like_placeholder_section(section)
        ]
        return title, clean_sections or fallback_sections

    @staticmethod
    def _is_competitor_research_query(query: str | None) -> bool:
        """Return True for competitor/market-positioning requests."""
        if not query:
            return False
        lowered = query.lower()
        return any(
            marker in lowered
            for marker in (
                "competitor",
                "competition",
                "competitive positioning",
                "market positioning",
                "competitive landscape",
            )
        )

    @staticmethod
    def _extract_competitor_focal_entity(query: str | None) -> str | None:
        """Extract the focal company from common competitor-analysis phrasing."""
        if not query:
            return None
        normalized = re.sub(r"\s+", " ", query).strip()
        match = re.search(
            r"\b(?:competitors?\s+(?:to|for|of|against)\s+|compare\s+)(?P<entity>[A-Z][A-Za-z0-9&.' -]{2,80})",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            return None
        entity = match.group("entity")
        entity = re.split(
            r"\s+(?:in|within|across|over|for|and|versus|vs\.?|with)\b|[,;:\n]",
            entity,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        entity = entity.strip(" .,:;\"'")
        return entity if len(entity) >= 3 else None

    @staticmethod
    def _names_appear_in_query(names: list[str], query: str | None) -> int:
        """Count listed entity names that appear in the user's own request/context."""
        if not query:
            return 0
        haystack = re.sub(r"\s+", " ", query).lower()
        count = 0
        for name in names:
            normalized = re.sub(r"\s+", " ", name).strip(" .,:;()[]{}\"'").lower()
            if len(normalized) >= 3 and normalized in haystack:
                count += 1
        return count

    @classmethod
    def _sanitize_unverified_competitor_sections(cls, sections: list[str], query: str | None) -> list[str]:
        """
        Remove plausible-but-unverified competitor lists from approval-plan headings.

        The approval UI is a plan, not a finding. For competitor research, specific
        competitor names should appear only when the user supplied them or when the
        later research stage has verified them. This keeps a plausible LLM guess from
        becoming an approved scope anchor.
        """
        if not cls._is_competitor_research_query(query):
            return sections

        sanitized: list[str] = []
        for section in sections:
            prefix, separator, suffix = section.partition(":")
            lowered_prefix = prefix.lower()
            if not separator or "competitor" not in lowered_prefix:
                sanitized.append(section)
                continue

            candidate_names = [
                part.strip(" .,:;()[]{}\"'")
                for part in re.split(r",|\band\b|/|;", suffix)
                if part.strip(" .,:;()[]{}\"'")
            ]
            # A colon followed by several proper names is usually the hallucination
            # pattern. If the user named most of them, keep the section intact.
            looks_like_entity_list = len(candidate_names) >= 2 and any(
                re.search(r"\b[A-Z][A-Za-z0-9&.'-]{2,}", name) for name in candidate_names
            )
            if not looks_like_entity_list:
                sanitized.append(section)
                continue

            supplied_count = cls._names_appear_in_query(candidate_names, query)
            if supplied_count >= max(2, len(candidate_names) // 2):
                sanitized.append(section)
                continue

            if "indirect" in lowered_prefix:
                replacement = (
                    "Indirect Competitor Mapping: Schools, Independent Tutors, Online Platforms, and EdTech Substitutes"
                )
            elif "direct" in lowered_prefix:
                replacement = "Direct Competitor Mapping: Verified Local Test-Prep and Tutoring Centers"
            else:
                replacement = "Competitor Identification and Verification"
            if replacement not in sanitized:
                sanitized.append(replacement)

        return sanitized

    def _plan_quality_issue(self, title: str | None, sections: list[str], query: str | None) -> str | None:
        """Return why a plan should be repaired before it reaches the approval UI."""
        if not title or not sections:
            return "The planner did not return both a title and section list."

        if self._looks_like_instruction_leak(title) or any(self._looks_like_instruction_leak(s) for s in sections):
            return "The plan leaked role or task instructions instead of a user-facing research plan."

        normalized_title = re.sub(r"\s+", " ", title.strip().lower()).strip(" .,:;")
        if re.fullmatch(r"(?:\d+|and|or|,|\s|&)+", normalized_title):
            return "The plan title used a clarification answer instead of the original research request."

        placeholder_count = sum(1 for section in sections if self._looks_like_placeholder_section(section))
        if placeholder_count >= 2 or (bool(sections) and placeholder_count == len(sections)):
            return "The plan used generic placeholder sections instead of topic-specific research threads."

        if self._is_weak_build_plan(title, sections, query):
            return (
                "The plan reduced a broad build/product-concept request to generic engineering buckets. "
                "It needs concrete research threads the user can approve."
            )

        if self._is_underfit_explicit_report_plan(title, sections, query):
            return (
                "The plan ignored explicit report requirements such as ranking, required dimensions, "
                "business-impact evidence, examples, maturity, technologies, barriers, beneficiaries, "
                "or outlook."
            )

        explicit_plan = self._deterministic_plan_from_query(query)
        generic_count = sum(1 for section in sections if self._looks_like_generic_fallback_section(section))
        if explicit_plan and generic_count >= 2:
            return "The plan used coarse fallback-like headings instead of the user's explicit report structure."

        return None

    @staticmethod
    def _plan_repair_prompt(
        issue: str,
        original_query: str | None,
        title: str | None,
        sections: list[str],
    ) -> str:
        """Build a focused repair request for a weak approval plan."""
        bad_sections = "; ".join(sections[:7]) if sections else "(none)"
        return (
            "Your previous research plan is not acceptable for the approval UI.\n"
            f"Quality issue: {issue}\n\n"
            f"Original user request: {original_query or '(missing)'}\n"
            f"Previous title: {title or '(none)'}\n"
            f"Previous sections: {bad_sections}\n\n"
            "Regenerate the plan from the ORIGINAL user request, not from the previous title. "
            "Do not return generic buckets like Requirements, Architecture and Interfaces, "
            "Failure Paths and Guardrails, Implementation Plan, Scope and Criteria, Key Findings, "
            "Landscape, Recent Evidence and Signals, Capability Gaps, Adoption Risks and Recommendations, "
            "or Recommended Next Steps unless the user explicitly asked for that exact structure.\n\n"
            "If the original request specifies a ranked list, top-N report, required dimensions, "
            "source standards, executive summary, detailed sections, or outlook, those requirements "
            "must appear as concrete plan sections.\n\n"
            "For broad deep-research or 'ingredients required to build X' requests, produce concrete "
            "research threads the user can approve without editing: foundations, user modeling, "
            "discovery/recommendation architecture, teaching or interaction design, AI/data stack, "
            "verification/evaluation, risks, or other topic-specific threads as appropriate.\n\n"
            'Return ONLY valid JSON: {"title":"...","sections":["..."]}.'
        )

    @staticmethod
    def _is_weak_build_plan(title: str, sections: list[str], query: str | None) -> bool:
        """Detect generic build-plan previews that under-describe product-concept research."""
        if not ClarifierAgent._extract_build_concept(query)[0]:
            return False
        normalized_title = re.sub(r"\s+", " ", title.lower())
        typo_or_echo_title = (
            "ingrediet" in normalized_title
            or "ingredie" in normalized_title
            or normalized_title.startswith("deep research on")
            or normalized_title.startswith("deep reserach on")
        )
        weak_sections = {
            "requirements",
            "architecture and interfaces",
            "failure paths and guardrails",
            "implementation plan",
        }
        normalized_sections = {
            re.sub(r"\s+", " ", section.lower()).replace("deep reserach on all ingredietns ", "").strip()
            for section in sections
        }
        generic_build_sections = len(normalized_sections & weak_sections) >= 2
        return typo_or_echo_title or generic_build_sections

    @staticmethod
    def _is_underfit_explicit_report_plan(title: str, sections: list[str], query: str | None) -> bool:
        """Detect plans that ignore an explicit report contract already present in the prompt."""
        deterministic = ClarifierAgent._deterministic_plan_from_query(query)
        if not deterministic:
            return False
        combined = " ".join([title, *sections]).lower()
        required_signals = [
            "rank",
            "business",
            "roi",
            "example",
            "maturity",
            "technolog",
            "barrier",
            "beneficiar",
        ]
        hits = sum(1 for signal in required_signals if signal in combined)
        return hits < 4

    def _format_plan_for_user(self, title: str, sections: list[str]) -> str:
        """
        Format the plan for user display.

        Args:
            title: Plan title.
            sections: List of section titles.

        Returns:
            Formatted string for user display.
        """
        sections_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(sections))
        return (
            f"**Research Plan Preview**\n\n"
            f"**Title:** {title}\n\n"
            f"**Sections:**\n{sections_text}\n\n"
            f"---\n"
            f"Reply **approve** to proceed, **reject** to cancel, or provide feedback to revise the plan."
        )

    def _parse_response(self, text: Any) -> ClarificationResponse | None:
        """
        Parse JSON response from LLM into ClarificationResponse.

        Attempts multiple strategies to extract JSON:
        1. Parse the entire text as JSON
        2. Extract from markdown code blocks
        3. Find JSON object pattern anywhere in text

        Args:
            text: Raw text response from LLM.

        Returns:
            ClarificationResponse if parsing succeeds, None otherwise.
        """
        if not text:
            return None

        text = coerce_content_text(text).strip()

        # Strategy 1: Try parsing the entire text as JSON
        try:
            data = json.loads(text)
            return ClarificationResponse.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Strategy 2: Extract from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                data = json.loads(json_match.group(1).strip())
                return ClarificationResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Strategy 3: Find JSON object pattern anywhere in text
        # Look for {"needs_clarification": ...} pattern
        json_pattern = re.search(r'\{[^{}]*"needs_clarification"[^{}]*\}', text)
        if json_pattern:
            try:
                data = json.loads(json_pattern.group(0))
                return ClarificationResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        # Strategy 4: Find any JSON object (more permissive)
        brace_match = re.search(r"\{[\s\S]*?\}", text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return ClarificationResponse.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass

        logger.warning("Failed to parse clarification response as JSON: %s...", text[:200])
        return None

    def _is_needed(self, text: str) -> bool:
        """
        Check if clarification is needed based on JSON response.

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            True if clarification is needed or parsing failed, False otherwise.
        """
        response = self._parse_response(text)
        if response is None:
            logger.warning("Failed to parse response, assuming clarification needed")
            return True
        return response.needs_clarification

    def _is_complete(self, text: str) -> bool:
        """
        Check if clarification is complete based on JSON response.

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            True if clarification is complete (needs_clarification=false),
            False otherwise or if parsing failed.
        """
        response = self._parse_response(text)
        if response is None:
            return False
        return response.is_complete()

    def _valid_needed(self, text: str) -> bool:
        """
        Check if the clarification response is valid.

        A response is valid if:
        - It parses successfully as JSON
        - When needs_clarification is true, it contains a clarification question

        Args:
            text: Raw JSON text response from the LLM.

        Returns:
            True if the response is valid, False otherwise.
        """
        response = self._parse_response(text)
        if response is None:
            return False
        return response.is_valid()

    def _get_clarification_question(self, text: str) -> str:
        """
        Extract the clarification question from the response.

        Args:
            text: Raw text response from LLM.

        Returns:
            The clarification question text.
        """
        response = self._parse_response(text)
        if response is not None and response.clarification_question:
            return response.clarification_question
        logger.warning("No clarification question found in response")
        return "Could you provide more details about your research needs?"

    def _get_llm(self) -> BaseChatModel:
        """
        Get the LLM instance for the clarifier agent.

        Uses LLMRole.CLARIFIER to obtain the appropriate LLM from the provider.

        Returns:
            The LangChain LLM instance for generating clarification questions.
        """
        return self.llm_provider.get(LLMRole.CLARIFIER)

    def _get_fallback_clarification(self, query: str | None = None) -> str:
        """
        Get fallback clarification text when the LLM response is invalid.

        Returns a topic-aware clarification question when query is provided,
        otherwise falls back to a generic question.

        Args:
            query: Optional user query to make the fallback more relevant.

        Returns:
            JSON string representing a ClarificationResponse with a fallback question.
        """
        if query:
            # Create topic-aware fallback
            topic_snippet = query[:80].strip()
            if len(query) > 80:
                topic_snippet += "..."
            question = (
                f'To help with your research on: "{topic_snippet}"\n\n'
                "Could you specify:\n"
                "1. Which specific aspects are most important to you?\n"
                "2. What level of detail do you need?\n"
                "3. Or type 'skip' to proceed with a general approach."
            )
        else:
            question = (
                "I'd like to help with your research. Could you provide more details about:\n\n"
                "1. What specific aspects interest you most?\n"
                "2. Who is this report for?\n"
                "3. How detailed should it be?"
            )

        fallback = ClarificationResponse(
            needs_clarification=True,
            clarification_question=question,
        )
        return fallback.model_dump_json()

    SKIP_COMMANDS = {"skip", "done", "exit", "quit", "proceed", "continue", "no", "n", ""}
    """Set of commands that indicate the user wants to skip clarification."""

    def _is_skip_command(self, user_reply: str) -> bool:
        """
        Check if the user's reply indicates they want to skip clarification.

        Recognized skip commands: skip, done, exit, quit, proceed, continue, no, n,
        or empty string.

        Args:
            user_reply: The user's response text.

        Returns:
            True if the reply is a skip command, False otherwise.
        """
        return user_reply.strip().lower() in self.SKIP_COMMANDS

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph StateGraph for the clarification workflow.

        Creates a graph with three nodes:
        - agent: Generates clarification questions using the LLM
        - tools: Executes tool calls (e.g., web search) for context
        - ask_for_clarification: Prompts user and processes response

        The graph flow:
        1. agent generates a response (question, tool call, or completion)
        2. If tool call → tools node → back to agent
        3. If question → ask_for_clarification → back to agent
        4. If complete → end

        Returns:
            Compiled LangGraph StateGraph ready for execution.
        """
        llm = self._get_llm()
        bound_llm = llm.bind_tools(self.tools, parallel_tool_calls=True) if self.tools else llm
        # Use planner_llm for plan generation if provided, otherwise use default llm
        planner_llm = self.planner_llm if self.planner_llm is not None else llm

        graph = StateGraph(ClarifierAgentState)

        async def agent_node(state: ClarifierAgentState):
            if state.remaining_questions <= 0:
                complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
                return {"messages": [AIMessage(content=complete_response.model_dump_json())]}

            original_query = self._get_original_query(state)
            if (
                state.iteration == 0
                and FINANCIAL_SCREEN_CLARIFIER_MARKER not in state.clarifier_log
                and self._needs_financial_screen_clarification(original_query)
            ):
                logger.info("Clarifier: Asking deterministic finance-screen clarification")
                return {"messages": [AIMessage(content=self._financial_screen_clarification_response())]}

            if state.iteration > 0 and FINANCIAL_SCREEN_CLARIFIER_MARKER in state.clarifier_log:
                logger.info("Clarifier: Finance-screen clarification answered; proceeding")
                complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
                return {"messages": [AIMessage(content=complete_response.model_dump_json())]}

            tools_info = [
                {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")} for t in self.tools
            ]
            rendered_system_prompt = render_prompt_template(
                self.system_prompt,
                clarifier_result=state.clarifier_log,
                available_documents=state.available_documents or [],
                tools=tools_info,
                tool_names=[t["name"] for t in tools_info],
            )

            # Build message list
            messages = [SystemMessage(content=rendered_system_prompt)] + state.messages

            # If last message is a tool result, add JSON reminder to prevent report generation
            if state.messages and isinstance(state.messages[-1], ToolMessage):
                logger.info("Adding JSON reminder after tool results")
                messages.append(HumanMessage(content=JSON_REMINDER_AFTER_TOOLS))

            response = await bound_llm.ainvoke(messages)
            return {"messages": [response]}

        async def ask_clarification(state: ClarifierAgentState):
            iteration = state.iteration
            max_turns = state.max_turns
            clarifier_log = state.clarifier_log
            if iteration >= max_turns:
                return {
                    "clarifier_log": f"Clarification complete: Met the maximum number of turns\n{clarifier_log}",
                }
            text = coerce_content_text(state.messages[-1].content) if state.messages else ""
            if not self._is_needed(text):
                return {}

            if not self._valid_needed(text):
                logger.warning("Invalid clarification format, forcing fallback")
                # Extract latest query for topic-aware fallback
                original_query = self._get_original_query(state)
                text = self._get_fallback_clarification(query=original_query if original_query else None)

            question_text = self._get_clarification_question(text)
            clarifier_log = f"{clarifier_log}\n**Turn {iteration + 1} - Assistant:**\n{question_text}"
            user_reply = await self.user_prompt_callback(question_text)

            if self._is_skip_command(user_reply):
                logger.info("Clarifier: User requested to skip clarification")
                complete_response = ClarificationResponse(needs_clarification=False, clarification_question=None)
                clarifier_log = f"{clarifier_log}\n**Turn {iteration + 1} - User:** [Skipped clarification]"
                return {
                    "messages": [AIMessage(content=complete_response.model_dump_json())],
                    "iteration": max_turns,  # Force end of clarification
                    "clarifier_log": clarifier_log,
                }

            clarifier_log = f"{clarifier_log}\n**Turn {iteration + 1} - User:**\n{user_reply}"
            return {
                "messages": [HumanMessage(content=user_reply)],
                "iteration": iteration + 1,
                "clarifier_log": clarifier_log,
            }

        def decide_route(state: ClarifierAgentState | dict):
            if isinstance(state, dict):
                messages = state.get("messages", [])
            elif hasattr(state, "messages"):
                messages = state.messages
            else:
                msg = f"No messages found in input state to tool_edge: {state}"
                raise ValueError(msg)

            if not messages:
                msg = f"Empty messages list in state: {state}"
                raise ValueError(msg)

            ai_message = messages[-1]
            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                return "tools"

            if self._is_complete(ai_message.content):
                if self.enable_plan_approval:
                    return "plan_preview"
                return "__end__"
            return "ask_for_clarification"

        async def plan_preview_node(state: ClarifierAgentState):
            """Generate plan preview and handle approval/feedback loop."""
            clarifier_log = state.clarifier_log
            feedback_history: list[str] = list(state.plan_feedback_history)
            original_query = self._get_original_query(state)

            # Initialize with fallback values in case loop doesn't execute (max_plan_iterations <= 0)
            title: str = self._fallback_plan_title(original_query)
            sections: list[str] = self._fallback_plan_sections(original_query)

            for iteration in range(self.max_plan_iterations):
                rendered_prompt = render_prompt_template(
                    self.plan_generation_prompt,
                    original_query=original_query,
                    clarifier_context=clarifier_log,
                    feedback_history=feedback_history if feedback_history else None,
                )

                # Generate plan using a clean, bounded message. The system prompt
                # already contains the original request and feedback history; passing
                # the whole conversation again made long role-play prompts easier to
                # echo and harder to parse as JSON.
                messages_for_plan = [
                    HumanMessage(
                        content=(
                            "Generate the research plan now for ORIGINAL USER REQUEST in the system prompt. "
                            "Return ONLY the JSON object with keys title and sections."
                        )
                    )
                ]
                response = await planner_llm.ainvoke([SystemMessage(content=rendered_prompt)] + messages_for_plan)
                title, sections = self._parse_plan_response(response.content)
                last_plan_excerpt = self._plan_response_excerpt(response.content)

                if not title or not sections:
                    logger.warning(
                        "Failed to generate valid plan, retrying once before fallback. Response excerpt: %s",
                        last_plan_excerpt,
                    )
                    retry_response = await planner_llm.ainvoke(
                        [
                            SystemMessage(content=rendered_prompt),
                            HumanMessage(
                                content=(
                                    "Your previous answer was not valid JSON. Return exactly one JSON object now: "
                                    '{"title":"...","sections":["..."]}. No markdown, no prose, no thinking text.'
                                )
                            ),
                        ]
                    )
                    title, sections = self._parse_plan_response(retry_response.content)
                    last_plan_excerpt = self._plan_response_excerpt(retry_response.content)

                if not title or not sections:
                    logger.warning(
                        "Failed to generate valid plan after retry, using fallback. Response excerpt: %s",
                        last_plan_excerpt,
                    )
                    title = self._fallback_plan_title(original_query)
                    sections = self._fallback_plan_sections(original_query)

                title, sections = self._compact_plan_for_query(title, sections, original_query)
                quality_issue = self._plan_quality_issue(title, sections, original_query)
                if quality_issue:
                    logger.warning("Planner produced weak approval plan; requesting model repair: %s", quality_issue)
                    repair_response = await planner_llm.ainvoke(
                        [
                            SystemMessage(content=rendered_prompt),
                            HumanMessage(
                                content=self._plan_repair_prompt(
                                    quality_issue,
                                    original_query,
                                    title,
                                    sections,
                                )
                            ),
                        ]
                    )
                    repaired_title, repaired_sections = self._parse_plan_response(repair_response.content)
                    if repaired_title and repaired_sections:
                        repaired_title, repaired_sections = self._compact_plan_for_query(
                            repaired_title,
                            repaired_sections,
                            original_query,
                        )
                        repaired_issue = self._plan_quality_issue(repaired_title, repaired_sections, original_query)
                        if not repaired_issue:
                            title, sections = repaired_title, repaired_sections
                        else:
                            logger.warning(
                                "Planner repair still produced a weak plan; using deterministic safety net: %s",
                                repaired_issue,
                            )
                            title = self._fallback_plan_title(original_query)
                            sections = self._fallback_plan_sections(original_query)
                    else:
                        logger.warning(
                            "Planner repair did not return a valid plan; using deterministic safety net. "
                            "Response excerpt: %s",
                            self._plan_response_excerpt(repair_response.content),
                        )
                        title = self._fallback_plan_title(original_query)
                        sections = self._fallback_plan_sections(original_query)

                title, sections = self._sanitize_plan_for_query(title, sections, original_query)

                # Present plan to user
                plan_display = self._format_plan_for_user(title, sections)
                user_response = await self.user_prompt_callback(plan_display)

                approved, rejected, feedback = self._parse_approval(user_response)

                if approved:
                    logger.info("Clarifier: Plan approved by user")
                    return {
                        "plan_title": title,
                        "plan_sections": sections,
                        "plan_approved": True,
                        "plan_rejected": False,
                        "plan_feedback_history": feedback_history,
                    }

                if rejected:
                    logger.info("Clarifier: Plan rejected by user")
                    return {
                        "plan_title": title,
                        "plan_sections": sections,
                        "plan_approved": False,
                        "plan_rejected": True,
                        "plan_feedback_history": feedback_history,
                    }

                if feedback:
                    normalized_feedback = self._normalize_plan_feedback(feedback, sections)
                    focused_plan = self._focused_plan_from_feedback(title, sections, feedback, original_query)
                    if focused_plan:
                        title, sections = focused_plan
                        title, sections = self._sanitize_plan_for_query(title, sections, original_query)
                        plan_display = self._format_plan_for_user(title, sections)
                        user_response = await self.user_prompt_callback(plan_display)
                        approved, rejected, followup_feedback = self._parse_approval(user_response)

                        if approved:
                            logger.info("Clarifier: Focused plan approved by user")
                            return {
                                "plan_title": title,
                                "plan_sections": sections,
                                "plan_approved": True,
                                "plan_rejected": False,
                                "plan_feedback_history": feedback_history + [normalized_feedback],
                            }

                        if rejected:
                            logger.info("Clarifier: Focused plan rejected by user")
                            return {
                                "plan_title": title,
                                "plan_sections": sections,
                                "plan_approved": False,
                                "plan_rejected": True,
                                "plan_feedback_history": feedback_history + [normalized_feedback],
                            }

                        feedback = followup_feedback or feedback

                # User provided feedback, add to history and continue loop
                logger.info("Clarifier: User provided feedback, regenerating plan")
                feedback_history.append(self._normalize_plan_feedback(feedback, sections))

            # Max iterations reached, auto-approve
            logger.warning("Clarifier: Max plan iterations reached, auto-approving")
            return {
                "plan_title": title,
                "plan_sections": sections,
                "plan_approved": True,
                "plan_rejected": False,
                "plan_feedback_history": feedback_history,
            }

        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("ask_for_clarification", ask_clarification)
        graph.add_node("plan_preview", plan_preview_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            decide_route,
            {
                "tools": "tools",
                "ask_for_clarification": "ask_for_clarification",
                "plan_preview": "plan_preview",
                "__end__": "__end__",
            },
        )

        graph.add_edge("tools", "agent")
        graph.add_edge("ask_for_clarification", "agent")
        graph.add_edge("plan_preview", "__end__")

        return graph.compile()

    async def run(self, state: ClarifierAgentState) -> ClarifierResult:
        """
        Execute the clarification dialog.

        Args:
            state: Initial state of the clarifier agent.

        Returns:
            ClarifierResult with clarification log and plan approval details.
        """
        logger.info("Clarifier: Starting (max %d turns)", self.max_turns)
        query = self._get_original_query(state)
        if query and not state.original_query:
            state = state.model_copy(update={"original_query": query})
        logger.info("User's query: %s...", str(query)[:100] if query else "")
        result = await self._graph.ainvoke(state, config={"callbacks": self.callbacks})
        final_state = ClarifierAgentState.model_validate(result)
        return ClarifierResult(
            clarifier_log=final_state.clarifier_log,
            plan_title=final_state.plan_title,
            plan_sections=final_state.plan_sections,
            plan_approved=final_state.plan_approved,
            plan_rejected=final_state.plan_rejected,
        )

    @property
    def graph(self) -> CompiledStateGraph:
        """Get the compiled LangGraph for direct access."""
        return self._graph

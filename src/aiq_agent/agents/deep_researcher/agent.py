# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deep research agent using deepagents library for multi-phase workflow."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore

from aiq_agent.common import LLMProvider
from aiq_agent.common import LLMRole
from aiq_agent.common import current_datetime_context
from aiq_agent.common import evaluate_report_source_quality
from aiq_agent.common import get_research_depth_config
from aiq_agent.common import is_model_failure_report
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template
from aiq_agent.common import report_matches_request_scope
from aiq_agent.common import sanitize_report_structure
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import rebuild_references
from aiq_agent.common.citation_verification import sanitize_report
from aiq_agent.common.citation_verification import verify_citations
from aiq_agent.common.claim_table import merge_claim_tables_json
from aiq_agent.common.claim_table import validate_claim_table_json
from aiq_agent.common.claude_code_specialist import run_claude_code_specialist
from aiq_agent.common.claude_code_specialist import should_use_claude_code
from aiq_agent.common.evidence_packet import build_evidence_packet
from aiq_agent.common.fact_ledger import live_fact_conflicts
from aiq_agent.common.fact_ledger import merge_fact_ledgers_json
from aiq_agent.common.fact_ledger import validate_fact_ledger_json
from aiq_agent.common.report_fact_audit import evaluate_report_fact_audit
from aiq_agent.common.report_fact_audit import fact_audit_note
from aiq_agent.common.research_artifacts import build_virtual_research_artifacts
from aiq_agent.common.research_artifacts import mirror_run_artifacts
from aiq_agent.common.source_quality_gates import evaluate_source_quality as evaluate_url_source_quality

from .custom_middleware import ArtifactWriteValidationMiddleware
from .custom_middleware import EmptyContentFixMiddleware
from .custom_middleware import PlanFileValidationMiddleware
from .custom_middleware import PlannerCommitGuardMiddleware
from .custom_middleware import PostWriteReadbackGuardMiddleware
from .custom_middleware import ReportEditCircuitBreakerMiddleware
from .custom_middleware import SearchBudgetExhaustionRepairMiddleware
from .custom_middleware import SequentialSearchMiddleware
from .custom_middleware import SourceRegistryMiddleware
from .custom_middleware import TaskBatchLimitMiddleware
from .custom_middleware import TaskSearchBudgetMiddleware
from .custom_middleware import ThinkingOnlyRepairMiddleware
from .custom_middleware import ToolArgumentNormalizationMiddleware
from .custom_middleware import ToolBudgetMiddleware
from .custom_middleware import ToolNameSanitizationMiddleware
from .custom_middleware import ToolResultPruningMiddleware
from .custom_middleware import ToolRetryMiddleware
from .custom_middleware import get_session_budget_snapshot
from .custom_middleware import reset_session_exhausted_tools
from .custom_middleware import reset_session_parallel_tool_limits
from .custom_middleware import reset_session_plan_validation_failures
from .custom_middleware import reset_session_planner_model_turns
from .custom_middleware import reset_session_recent_artifact_writes
from .custom_middleware import reset_session_report_edit_failures
from .custom_middleware import reset_session_task_search_counts
from .custom_middleware import reset_session_tool_counts
from .custom_middleware import reset_session_tool_limits
from .custom_middleware import set_session_exhausted_tools
from .custom_middleware import set_session_parallel_tool_limits
from .custom_middleware import set_session_plan_validation_failures
from .custom_middleware import set_session_planner_model_turns
from .custom_middleware import set_session_recent_artifact_writes
from .custom_middleware import set_session_report_edit_failures
from .custom_middleware import set_session_task_search_counts
from .custom_middleware import set_session_tool_counts
from .custom_middleware import set_session_tool_limits
from .deepagents_runtime import DeepAgentsRuntime
from .deepagents_runtime import SandboxConfig
from .deepagents_runtime import SkillsConfig
from .models import DeepResearchAgentState
from .plan_tools import create_write_plan_tool

try:
    from aiq_api.auth.errors import AuthError as _AuthError
except ImportError:
    _AuthError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Minimum character count for a report to be considered substantive.
# Used by both _extract_report_content (to decide if write_file fallback is needed)
# and _is_report_complete (to reject too-short reports).
_MIN_REPORT_LENGTH = 1500
_REPORT_COMPILER_MAX_INPUT_CHARS = 80000
_REPORT_COMPILER_MAX_ARTIFACT_CHARS = 24000

# Path to this agent's directory (for loading prompts)
AGENT_DIR = Path(__file__).parent

APPROVED_PLAN_RE = re.compile(
    r"\*\*Approved Research Plan\*\*\s*Title:\s*(?P<title>.+?)\s*Sections:\s*(?P<sections>(?:\s*-\s*.+(?:\n|$))+)",
    re.IGNORECASE,
)
"""Extract approved plan context created by the clarifier."""

GENERIC_APPROVED_PLAN_SECTIONS = {
    "introduction",
    "background",
    "analysis",
    "findings",
    "conclusion",
}

GENERIC_APPROVED_PLAN_SECTION_MARKERS = {
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

EXACT_LESSON_TOPIC_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:Exact lesson topic|Broad topic)(?:\*\*)?\s*:\s*(?P<topic>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FINAL_DEBATE_MOTION_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?Final debate motion(?:\*\*)?\s*:\s*(?P<motion>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
REPORT_TYPE_RE = re.compile(r"^\s*Report type:\s*(?P<report_type>.+?)\s*$", re.IGNORECASE | re.MULTILINE)
AUDIENCE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:Audience|Student tier)(?:\*\*)?\s*:\s*(?P<audience>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
MARKDOWN_SECTION_LIST_RE = re.compile(
    r"The Markdown must include these sections:\s*(?P<sections>(?:\s*\d+\.\s*.+(?:\n|$))+)",
    re.IGNORECASE,
)
"""Extract structured lesson/report prompts generated by the content API."""

TRAINING_CONTENT_KEYWORDS_RE = re.compile(
    r"\b(?:WSDC|BP|debate training|competitive debate|training session|slide[- ]deck|speaker notes|debater|coach)\b",
    re.IGNORECASE,
)
TRAINING_TOPIC_PATTERNS = (
    re.compile(r"\bcovering\s+[\"“](?P<topic>[^\"”\n]{8,220})[\"”]", re.IGNORECASE),
    re.compile(
        r"\b(?:research|report|material|session|specialist)\s+on\s+(?P<topic>[^.\n]{8,220}?)\s+to\s+"
        r"(?:create|build|produce|develop|turn)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bon\s+(?P<topic>[^.\n]{8,220}?)\s+(?:to\s+(?:create|build|produce|develop|turn)|for\s+"
        r"(?:a\s+)?(?:world-class\s+)?(?:WSDC|BP|competitive|debate|training|slide))",
        re.IGNORECASE,
    ),
)


@tool
def think(thought: str) -> str:
    """Use this tool to reason through complex decisions, verify constraints, or
    plan next steps before acting. The tool records your thought without taking
    any action or retrieving new information.

    When to use:
    - Before making a decision: reason through options and trade-offs
    - After receiving information: analyze findings and identify gaps
    - For constraint verification: check if a constraint is satisfied and note PASS/FAIL
    - When planning: outline your approach before executing

    Args:
        thought: Your reasoning, analysis, or verification to record.
    """
    logger.info("Thinking: %s", thought)
    return "Thought recorded."


@tool
def defer_task(description: str, reason: str) -> str:
    """Record a researcher task that should be launched after the current batch.

    Args:
        description: Full researcher-agent task description to run later.
        reason: Why the task was deferred.
    """
    return (
        "Deferred researcher task for a later batch.\n\n"
        f"Reason: {reason}\n\n"
        f"Task to launch later if still needed:\n{description}"
    )


class DeepResearcherAgent:
    """
    Deep research agent using deepagents library for multi-phase workflow.

    This agent produces publication-ready research reports through an iterative process:

    1. **Planning Phase**: Generate a structured research plan with queries and report
       organization (planner subagent)
    2. **Research Loops**: Execute queries via web search (researcher subagent), then
       synthesize drafts directly in the orchestrator
    3. **Iteration**: Repeat research and synthesis loops to fill gaps
    4. **Citation Management**: Catalog and number sources in the orchestrator
    5. **Finalization**: Produce a polished report with inline citations and references
       directly in the orchestrator

    The agent is NAT-independent and receives all dependencies via constructor.

    Example:
        >>> from aiq_agent.common import LLMProvider, LLMRole
        >>> provider = LLMProvider()
        >>> provider.set_default(my_llm)
        >>> provider.configure(LLMRole.ORCHESTRATOR, orchestrator_llm)
        >>> provider.configure(LLMRole.RESEARCHER, researcher_llm)
        >>> provider.configure(LLMRole.PLANNER, planner_llm)
        >>>
        >>> from aiq_agent.agents.deep_researcher.models import DeepResearchAgentState
        >>> agent = DeepResearcherAgent(
        ...     llm_provider=provider,
        ...     tools=[search_tool_a, search_tool_b],
        ... )
        >>> state = DeepResearchAgentState(messages=[HumanMessage(content="Compare CUDA vs OpenCL")])
        >>> result = await agent.run(state)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[BaseTool] | None = None,
        *,
        max_loops: int = 2,
        verbose: bool = True,
        callbacks: list[Any] | None = None,
        skills: SkillsConfig | None = None,
        sandbox: SandboxConfig | None = None,
        config: Any | None = None,
        job_id: str | None = None,
    ) -> None:
        """
        Initialize the deep researcher subagent.

        Args:
            llm_provider: LLMProvider for role-based LLM access.
            tools: Optional sequence of LangChain tools for research.
            max_loops: Maximum number of research loops (default 2).
            verbose: Enable detailed logging.
            callbacks: Optional list of callbacks.
            skills: Optional DeepAgents skills config.
            sandbox: Optional DeepAgents sandbox config.
            config: Optional agent config. Used by async workers to pass function config generically.
            job_id: Optional async job identifier used to scope sandbox backends.
        """
        self.llm_provider = llm_provider
        self.tools = list(tools) if tools else []
        self.max_loops = max_loops
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.job_id = job_id
        self._active_request_text = ""
        self._active_research_depth = "deeper"

        if self.verbose:
            logger.info("Tools configured: %d", len(self.tools))
        if config is not None:
            skills = skills or getattr(config, "skills", None)
            sandbox = sandbox if sandbox is not None else getattr(config, "sandbox", None)
        self.deepagents_runtime = DeepAgentsRuntime(skills=skills, sandbox=sandbox, job_id=job_id)

        self._prompts = self._load_prompts()
        self.tools_info = []
        for t in self.tools:
            self.tools_info.append({"name": t.name, "description": t.description})

        self.source_registry_middleware = SourceRegistryMiddleware(
            source_tool_names={t.name for t in self.tools},
        )

        # Create a tool that gives the orchestrator access to verified sources
        registry_middleware = self.source_registry_middleware

        @tool
        def get_verified_sources() -> str:
            """Returns the list of all verified source URLs captured from search tool calls.

            Call this tool during the Synthesize step (Step 5) BEFORE writing the
            final report. It returns every URL and citation key that was returned
            by search tools during research. Use ONLY these sources in your report
            — any other URL will be automatically removed.

            Returns:
                A numbered list of verified sources with titles and URLs.
            """
            source_list = registry_middleware.get_source_list_text()
            if source_list:
                return source_list
            return "No sources captured yet. Run research queries first."

        @tool
        def get_source_quality_snapshot() -> str:
            """Return the current captured-source diversity and source-class mix.

            Call this after each researcher batch and before final synthesis. If
            status is warn/fail, steer the next available researcher task toward
            stronger or more diverse sources instead of waiting for post-run gates.
            """
            sources = registry_middleware._get_registry().all_sources()
            urls = [source.url for source in sources if source.url]
            if not urls:
                return json.dumps(
                    {
                        "status": "fail",
                        "message": "No source URLs captured yet. Researchers must use search/source tools.",
                    },
                    indent=2,
                )
            report = evaluate_url_source_quality(list(dict.fromkeys(urls)), tier="deep")
            return report.model_dump_json(indent=2)

        @tool
        def get_research_progress_snapshot() -> str:
            """Return live budget, source, and structured-artifact progress.

            Call this between researcher batches and before final synthesis.
            Use it to decide whether to launch a gap-filling task, move to
            synthesis, or acknowledge source/claim limitations.
            """
            backend = StateBackend()

            def _glob_count(pattern: str) -> int:
                try:
                    result = backend.glob(pattern)
                    if isinstance(result, list):
                        return len(result)
                    matches = getattr(result, "matches", None)
                    if isinstance(matches, list):
                        return len(matches)
                    paths = getattr(result, "paths", None)
                    if isinstance(paths, list):
                        return len(paths)
                except Exception:
                    logger.debug("Unable to glob progress pattern %s", pattern, exc_info=True)
                return 0

            def _read_present(path: str) -> bool:
                try:
                    result = backend.read(path)
                    if getattr(result, "error", None):
                        return False
                    content = getattr(getattr(result, "file_data", None), "content", None)
                    if isinstance(content, list):
                        return bool("".join(str(part) for part in content).strip())
                    return bool(str(content or "").strip())
                except Exception:
                    logger.debug("Unable to read progress path %s", path, exc_info=True)
                    return False

            sources = registry_middleware._get_registry().all_sources()
            urls = [source.url for source in sources if source.url]
            source_quality = (
                evaluate_url_source_quality(list(dict.fromkeys(urls)), tier="deep").model_dump()
                if urls
                else {"overall_status": "fail", "failure_reasons": ["no_sources_captured"]}
            )
            snapshot = {
                "budget": get_session_budget_snapshot(),
                "sources": {
                    "captured_urls": len(set(urls)),
                    "quality": source_quality,
                },
                "artifacts": {
                    "claim_fragments": _glob_count("/shared/claims/claims_*.json"),
                    "extract_fragments": _glob_count("/shared/extracts/*.json"),
                    "section_briefs": _glob_count("/shared/section_briefs/*.md"),
                    "claim_table_present": _read_present("/shared/claim_table.json"),
                    "evidence_packet_present": _read_present("/shared/evidence_packet.json"),
                },
            }
            return json.dumps(snapshot, indent=2, ensure_ascii=False)

        @tool
        def compile_research_dossier() -> str:
            """Compile `/shared/research.md`, `/shared/sources.json`, gaps, and contradictions.

            Call this after researcher batches and before final report writing.
            It creates the canonical evidence brief that final synthesis should
            use as the truth substrate.
            """
            files = self._collect_live_virtual_files()
            if not files:
                return "No virtual research artifacts were available to compile."
            artifacts = build_virtual_research_artifacts(
                files=files,
                registry_sources=self.source_registry_middleware._get_registry().all_sources(),
                job_id=self.job_id,
                request_text=self._active_request_text,
            )
            if "/shared/research.md" in artifacts:
                artifacts["/shared/research.md"] = self._append_claude_code_memos(
                    artifacts["/shared/research.md"],
                    files,
                )
            for path, content in artifacts.items():
                self._upsert_live_virtual_file(path, content)
            research = artifacts.get("/shared/research.md", "")
            sources = artifacts.get("/shared/sources.json", "[]")
            try:
                source_count = len(json.loads(sources))
            except Exception:
                source_count = 0
            return (
                "Compiled research dossier artifacts: /shared/research.md, /shared/sources.json, "
                "/shared/gaps.md, /shared/contradictions.md. "
                f"Scored sources: {source_count}. "
                f"Research brief chars: {len(research)}."
            )

        async def async_ask_claude_code_specialist(stage: str, task: str) -> str:
            """Ask Claude Code for bounded plan-direction or synthesis-review help."""
            files = self._collect_live_virtual_files()
            context = self._claude_code_context(files=files, stage=stage)
            source_count = len(self.source_registry_middleware._get_registry().all_sources())
            safe_stage = re.sub(r"[^a-zA-Z0-9_-]+", "_", stage or "specialist")
            safe_job_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.job_id or "job")
            local_artifact_path = Path("/tmp") / "aiq_claude_code" / safe_job_id / f"{safe_stage}.md"
            decision = should_use_claude_code(
                task,
                current_context_size=len(context),
                num_sources=source_count,
                stage=stage,
                tier=self._active_research_depth,
            )
            if not decision.should_use:
                self._record_claude_code_usage(
                    stage,
                    used=False,
                    reason=", ".join(decision.reasons) or "routing_score_below_threshold",
                )
                return (
                    "Claude Code specialist skipped by routing policy. "
                    f"Score: {decision.score}; reasons: {', '.join(decision.reasons) or 'none'}."
                )
            result = await run_claude_code_specialist(
                task=task,
                context=context,
                stage=stage,
                tier=self._active_research_depth,
                cwd=Path.cwd(),
                output_path=local_artifact_path,
            )
            if not result.ok:
                self._record_claude_code_usage(stage, used=False, reason=result.error or "no_output")
                return (
                    f"Claude Code specialist unavailable: {result.error or 'no output'}. "
                    f"Expected local artifact: {result.artifact_path or local_artifact_path}."
                )
            artifact_path = f"/shared/claude_code/{safe_stage}.md"
            self._upsert_live_virtual_file(artifact_path, result.output)
            self._record_claude_code_usage(
                stage,
                used=True,
                reason=", ".join(decision.reasons) or "used",
                artifact_path=artifact_path,
                artifact_chars=len(result.output),
            )
            return (
                f"Claude Code specialist wrote {artifact_path} from local artifact {result.artifact_path}.\n"
                "REQUIRED HANDOFF: read and apply the artifact's Handoff Notes in the next workflow step; "
                "do not treat this as optional commentary.\n\n"
                f"{self._truncate_artifact(result.output, limit=6000)}"
            )

        ask_claude_code_specialist = StructuredTool.from_function(
            name="ask_claude_code_specialist",
            description=(
                "Optional Claude Code + M3 specialist handoff for plan direction, synthesis review, "
                "structured tables, gap analysis, or artifact shaping. Do not use for simple search/scrape."
            ),
            coroutine=async_ask_claude_code_specialist,
            infer_schema=True,
        )

        write_plan_tool = create_write_plan_tool()
        # M3 has native thinking; exposing an additional planner `think` tool
        # caused costly detours after "I have enough grounding" instead of a
        # committed plan. Keep `think` for orchestration/synthesis, but make
        # planner-agent commit through write_plan or search tools only.
        self.planner_tools = [write_plan_tool, *self.tools]
        self.orchestrator_tools = [
            think,
            defer_task,
            write_plan_tool,
            get_verified_sources,
            get_source_quality_snapshot,
            get_research_progress_snapshot,
            compile_research_dossier,
            ask_claude_code_specialist,
        ]
        self.all_tools = [*self.orchestrator_tools, *self.tools]

        self.middleware = self._build_middleware_for_scope("orchestrator")

    def _build_middleware_for_scope(self, budget_scope: str):
        """Build middleware with role-scoped search budgets and shared source registry."""
        middleware = [
            EmptyContentFixMiddleware(),
            ToolNameSanitizationMiddleware(valid_tool_names=[t.name for t in self.all_tools]),
            ToolArgumentNormalizationMiddleware(),
            PlanFileValidationMiddleware(),
            ArtifactWriteValidationMiddleware(),
            ReportEditCircuitBreakerMiddleware(),
            PostWriteReadbackGuardMiddleware(),
            TaskBatchLimitMiddleware(task_tool_name="task", defer_tool_name="defer_task"),
            SequentialSearchMiddleware(
                search_tool_names={"advanced_web_search_tool", "web_search_tool", "exa_web_search_tool"}
            ),
            SearchBudgetExhaustionRepairMiddleware(
                search_tool_names={"advanced_web_search_tool", "web_search_tool", "exa_web_search_tool"},
                max_repairs=0,
                scope=budget_scope,
            ),
            TaskSearchBudgetMiddleware(
                search_tool_names={"advanced_web_search_tool", "web_search_tool", "exa_web_search_tool"}
            ),
            ToolBudgetMiddleware(
                limits={
                    "exa_web_search_tool": 6,
                    "advanced_web_search_tool": 8,
                    "web_search_tool": 4,
                    "stock_quote_tool": 4,
                },
                scope=budget_scope,
            ),
            ToolRetryMiddleware(max_retries=1, backoff_factor=1.5, initial_delay=0.5),
            self.source_registry_middleware,
            # Keep planner/researcher turns compact. M3's large context is most
            # valuable for final synthesis over structured evidence packets,
            # not for every search/write loop inside subagents.
            self._tool_result_pruning_middleware_for_scope(budget_scope),
            ThinkingOnlyRepairMiddleware(max_repairs=2),
            ModelRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
        ]
        if budget_scope == "planner":
            middleware.insert(3, PlannerCommitGuardMiddleware(max_model_turns=2, max_repairs=0))
        return middleware

    @staticmethod
    def _tool_result_pruning_middleware_for_scope(scope: str) -> ToolResultPruningMiddleware:
        """Return role-specific pruning tuned for M3 stability.

        Researcher and planner agents should work from compact current evidence
        plus virtual files. The orchestrator gets a wider window because final
        synthesis is where M3's long context is useful.
        """
        if scope == "orchestrator":
            return ToolResultPruningMiddleware(
                keep_last_n=8,
                max_chars=1200,
                recent_max_chars=40000,
                max_tool_call_arg_chars=5000,
            )
        if scope == "planner":
            return ToolResultPruningMiddleware(
                keep_last_n=4,
                max_chars=500,
                recent_max_chars=10000,
                max_tool_call_arg_chars=1600,
            )
        return ToolResultPruningMiddleware(
            keep_last_n=4,
            max_chars=700,
            recent_max_chars=12000,
            max_tool_call_arg_chars=1800,
        )

    def _load_prompts(self) -> dict[str, str]:
        """Load all prompts for subagents."""
        prompts = {}
        prompt_names = ["planner", "researcher", "orchestrator"]

        for name in prompt_names:
            try:
                prompts[name] = load_prompt(AGENT_DIR / "prompts", name)
            except Exception as e:
                logger.warning("Failed to load prompt %s: %s, using inline default", name, e)
                prompts[name] = self._get_inline_default(name)

        return prompts

    def _get_inline_default(self, name: str) -> str:
        """Get inline default prompt for fallback."""
        defaults = {
            "planner": "You are a research planning strategist. Create a structured research plan.",
            "researcher": "You are a research investigator. Gather information from available sources.",
            "orchestrator": (
                "You are a research orchestrator. Coordinate the research process and produce a polished report."
            ),
        }
        return defaults.get(name, f"You are a {name} agent.")

    def _deepagents_prompt_context(self) -> dict[str, Any]:
        """Return prompt variables for optional DeepAgents skills and sandbox support."""
        skill_sources = self.deepagents_runtime.skill_sources
        sandbox = self.deepagents_runtime.sandbox
        return {
            "skills_enabled": skill_sources is not None,
            "skill_sources": skill_sources or [],
            "sandbox_enabled": sandbox is not None,
            "sandbox_python_packages": tuple(sandbox.python_packages) if sandbox is not None else (),
        }

    def _get_subagents(self, state: DeepResearchAgentState) -> list[dict[str, Any]]:
        """Build subagent configs with state-dependent prompts (e.g. available_documents)."""
        available_docs = [doc.model_dump() for doc in (state.available_documents or [])]
        deepagents_prompt_context = self._deepagents_prompt_context()
        skill_sources = self.deepagents_runtime.skill_sources
        current_datetime = current_datetime_context()
        depth_config = get_research_depth_config(state.research_depth)
        budget_profile = self._budget_profile_for_state(state)
        planner_agent: dict[str, Any] = {
            "name": "planner-agent",
            "description": (
                "Content-driven research planning - iteratively builds evidence-grounded "
                "outlines through interleaved search and outline optimization"
            ),
            "system_prompt": render_prompt_template(
                self._prompts["planner"],
                current_datetime=current_datetime,
                user_info=state.user_info,
                tools=self.tools_info,
                available_documents=available_docs,
                research_depth=depth_config,
                budget_profile=budget_profile,
                **deepagents_prompt_context,
            ),
            "tools": self.planner_tools,
            "model": self._planner_llm_for_state(state),
            "middleware": self._build_middleware_for_scope("planner"),
        }
        researcher_agent: dict[str, Any] = {
            "name": "researcher-agent",
            "description": (
                "Information gathering - executes search queries and synthesizes "
                "relevant content from available sources"
            ),
            "system_prompt": render_prompt_template(
                self._prompts["researcher"],
                current_datetime=current_datetime,
                user_info=state.user_info,
                tools=self.tools_info,
                available_documents=available_docs,
                research_depth=depth_config,
                budget_profile=budget_profile,
                **deepagents_prompt_context,
            ),
            "tools": self.all_tools,
            "model": self._researcher_llm_for_state(state),
            "middleware": self._build_middleware_for_scope("researcher"),
        }
        if skill_sources is not None:
            planner_agent["skills"] = skill_sources
            researcher_agent["skills"] = skill_sources
        return [planner_agent, researcher_agent]

    def _build_orchestrator_agent(self, state: DeepResearchAgentState) -> str:
        """Get the orchestrator instructions for the deep research agent."""

        available_docs = [doc.model_dump() for doc in (state.available_documents or [])]
        deepagents_prompt_context = self._deepagents_prompt_context()
        depth_config = get_research_depth_config(state.research_depth)
        budget_profile = self._budget_profile_for_state(state)
        orchestrator_instructions = render_prompt_template(
            self._prompts["orchestrator"],
            current_datetime=current_datetime_context(),
            user_info=state.user_info,
            clarifier_result=state.clarifier_result,
            available_documents=available_docs,
            tools=self.tools_info,
            research_depth=depth_config,
            budget_profile=budget_profile,
            **deepagents_prompt_context,
        )

        deepagents_kwargs = self.deepagents_runtime.create_agent_kwargs

        agent = create_deep_agent(
            model=self._orchestrator_llm_for_state(state),
            tools=self.orchestrator_tools,
            system_prompt=orchestrator_instructions,
            subagents=self._get_subagents(state),
            store=InMemoryStore(),
            context_schema=DeepResearchAgentState,
            middleware=self.middleware,
            **deepagents_kwargs,
        )
        return agent.with_config({"recursion_limit": 1000})

    def _orchestrator_llm_for_state(self, state: DeepResearchAgentState) -> Any:
        """Use a thinking orchestrator only for tasks that benefit from it."""
        if self._needs_thinking_orchestrator(state):
            return self.llm_provider.get(LLMRole.ORCHESTRATOR)
        if state.research_depth == "medium" and self.llm_provider.has_role(LLMRole.MEDIUM_ORCHESTRATOR):
            return self.llm_provider.get(LLMRole.MEDIUM_ORCHESTRATOR)
        if state.research_depth == "deeper" and self.llm_provider.has_role(LLMRole.DEEPER_ORCHESTRATOR):
            return self.llm_provider.get(LLMRole.DEEPER_ORCHESTRATOR)
        return self.llm_provider.get(LLMRole.ORCHESTRATOR)

    @classmethod
    def _needs_thinking_orchestrator(cls, state: DeepResearchAgentState) -> bool:
        """Return true only for the highest-leverage synthesis path."""
        return state.research_depth == "deep"

    @staticmethod
    def _has_large_document_context(state: DeepResearchAgentState) -> bool:
        docs = state.available_documents or []
        if len(docs) >= 3:
            return True
        summary_chars = 0
        for doc in docs:
            dump = doc.model_dump() if hasattr(doc, "model_dump") else {}
            summary_chars += len(str(dump.get("summary") or ""))
            summary_chars += len(str(dump.get("file_name") or ""))
        return summary_chars >= 8000

    def _planner_llm_for_state(self, state: DeepResearchAgentState) -> Any:
        """Use tier-specific planner models when configured."""
        if state.research_depth == "deep" and self.llm_provider.has_role(LLMRole.DEEP_PLANNER):
            return self.llm_provider.get(LLMRole.DEEP_PLANNER)
        if state.research_depth == "deeper" and self.llm_provider.has_role(LLMRole.DEEPER_PLANNER):
            return self.llm_provider.get(LLMRole.DEEPER_PLANNER)
        return self.llm_provider.get(LLMRole.PLANNER)

    def _researcher_llm_for_state(self, state: DeepResearchAgentState) -> Any:
        """Use tier-specific researcher models when configured."""
        if state.research_depth == "deep" and self.llm_provider.has_role(LLMRole.DEEP_RESEARCHER):
            return self.llm_provider.get(LLMRole.DEEP_RESEARCHER)
        if state.research_depth == "deeper" and self.llm_provider.has_role(LLMRole.DEEPER_RESEARCHER):
            return self.llm_provider.get(LLMRole.DEEPER_RESEARCHER)
        return self.llm_provider.get(LLMRole.RESEARCHER)

    @staticmethod
    def _latest_user_text(state: DeepResearchAgentState) -> str:
        """Return the latest human message as text."""
        for message in reversed(state.messages or []):
            if isinstance(message, HumanMessage):
                content = message.content
                return content if isinstance(content, str) else str(content)
        return ""

    @staticmethod
    def _extract_approved_plan(text: str | None) -> tuple[str, list[str]] | None:
        """Extract an approved plan title/sections from clarification context text."""
        if not text:
            return None
        match = APPROVED_PLAN_RE.search(text)
        if not match:
            return None

        title = match.group("title").strip()
        raw_sections = match.group("sections")
        sections = []
        for line in raw_sections.splitlines():
            normalized = line.strip()
            if normalized.startswith("-"):
                section = normalized[1:].strip()
                if section:
                    sections.append(section)

        return (title, sections) if title and sections else None

    @staticmethod
    def _query_without_context(query: str) -> str:
        """Strip appended clarification context from API/direct deep-research input."""
        markers = ("## Clarification Context", "**Approved Research Plan**")
        clean = query
        for marker in markers:
            if marker in clean:
                clean = clean.split(marker, 1)[0]
        return clean.strip() or query.strip()

    @staticmethod
    def _is_financial_screen_query(query: str) -> bool:
        """Return true for precise stock/fair-value discount screens."""
        clean_query = query.lower()
        return all(term in clean_query for term in ("stock", "fair value")) and bool(
            re.search(r"\d+\s*(?:-|–|to)\s*\d+\s*%", clean_query)
        )

    @staticmethod
    def _format_approved_plan_context(title: str, sections: list[str]) -> str:
        """Format plan scope so the orchestrator prompt takes the approved-plan fast path."""
        section_lines = "\n".join(f"- {section}" for section in sections)
        return f"**Approved Research Plan**\n\nTitle: {title}\n\nSections:\n{section_lines}"

    @staticmethod
    def _format_structured_lesson_context(scope: dict[str, Any]) -> str:
        """Format structured lesson scope with glossary notes for planner-agent."""

        sections = DeepResearcherAgent._structured_lesson_section_titles(scope)
        context = DeepResearcherAgent._format_approved_plan_context(
            f"{scope['topic']} Content Research Dossier",
            sections,
        )
        glossary = scope.get("glossary") or {}
        if glossary:
            glossary_lines = "\n".join(f"- {key} = {value}" for key, value in glossary.items())
            context += (
                "\n\nPlanner Context Notes:\n"
                f"- Broad topic: {scope['topic']}\n"
                f"- Final motion/application focus: {scope['motion']}\n"
                "- Abbreviation glossary:\n"
                f"{glossary_lines}\n"
                "- Use glossary expansions when interpreting curriculum shorthand; do not invent alternate meanings."
            )
        return context

    @staticmethod
    def _is_generic_approved_plan(title: str, sections: list[str]) -> bool:
        """Detect placeholder plans that should be replaced by real planning."""
        normalized_title = title.strip().lower()
        normalized_sections = {section.strip().lower() for section in sections}
        if normalized_title == "research report" and normalized_sections == GENERIC_APPROVED_PLAN_SECTIONS:
            return True
        generic_marker_count = 0
        for section in normalized_sections:
            section_tail = re.sub(r"^[a-z0-9 ,.'\"-]{0,80}\s+", "", section)
            if (
                section in GENERIC_APPROVED_PLAN_SECTION_MARKERS
                or section_tail in GENERIC_APPROVED_PLAN_SECTION_MARKERS
            ):
                generic_marker_count += 1
        if generic_marker_count >= 2:
            return True
        if normalized_title in {"approve", "approved", "continue", "skip"}:
            return True
        instruction_leak_markers = (
            "you are deep research",
            "operating in full autonomous",
            "your task is to",
        )
        if any(marker in normalized_title for marker in instruction_leak_markers):
            return True
        return any(any(marker in section for marker in instruction_leak_markers) for section in normalized_sections)

    @staticmethod
    def _strip_approved_plan_context(text: str | None) -> str | None:
        """Remove a placeholder approved-plan block so planner-agent can rebuild it."""
        if not text:
            return text
        stripped = APPROVED_PLAN_RE.sub("", text).strip()
        if stripped:
            return (
                f"{stripped}\n\n"
                "The previous plan preview was a generic placeholder. Generate a fresh, specific research plan."
            )
        return "The previous plan preview was a generic placeholder. Generate a fresh, specific research plan."

    @staticmethod
    def _title_case_phrase(text: str) -> str:
        """Title-case a short topic phrase while preserving AI casing."""
        words = []
        for word in re.sub(r"\s+", " ", text.strip()).split(" "):
            if word.lower() == "ai":
                words.append("AI")
            elif word.isupper() and len(word) <= 5:
                words.append(word)
            else:
                words.append(word[:1].upper() + word[1:])
        return " ".join(words).strip()

    @staticmethod
    def _fallback_plan_topic(query: str | None) -> str:
        """Extract a compact topic label from an explicit prompt."""
        if not query:
            return "Research Report"
        clean_query = DeepResearcherAgent._query_without_context(query)
        clean_query = re.sub(
            r"^(?:conduct\s+)?(?:a\s+)?(?:comprehensive\s+)?(?:deep\s+)?research(?:\s+report)?\s+(?:on|about)?\s*",
            "",
            clean_query,
            flags=re.IGNORECASE,
        ).strip()
        clean_query = re.sub(
            r"^(?:on\s+)?all\s+the\s+ingred(?:ie|ei)nts\s+required\s+to\s+build\s*:?\s*",
            "",
            clean_query,
            flags=re.IGNORECASE,
        )
        clean_query = re.split(r"[?.!\n]", clean_query, maxsplit=1)[0]
        words = [word.strip(" ,:;()[]{}\"'“”") for word in clean_query.split()]
        words = [word for word in words if word]
        return DeepResearcherAgent._title_case_phrase(" ".join(words[:8]))[:120].rstrip(" .,:;") or "Research Report"

    @staticmethod
    def _explicit_plan_from_rich_query(query: str | None) -> tuple[str, list[str]] | None:
        """Recover a specific plan from rich prompts when the UI approval plan was generic."""
        if not query:
            return None
        clean_query = DeepResearcherAgent._query_without_context(query)
        normalized = re.sub(r"\s+", " ", clean_query).strip()
        lowered = normalized.lower()

        top_match = re.search(
            r"\btop\s+(\d{1,2})\s+(?:highest[-\s]value\s+)?(?:use\s+cases?|applications?)\s+of\s+AI\b",
            normalized,
            re.IGNORECASE,
        )
        if top_match and re.search(r"\b(rank|ranked|ranking|business value|roi|transformative)\b", lowered):
            count = top_match.group(1)
            year_match = re.search(r"\b(20\d{2})\b", normalized)
            year_suffix = f" in {year_match.group(1)}" if year_match else ""
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

        if re.search(r"\bingred(?:ie|ei)nts\s+required\s+to\s+build\b", lowered) and "curiosity" in lowered:
            return (
                "Building AI Curiosity Engines",
                [
                    "Learning Science and Curiosity Foundations",
                    "Dynamic Interest Modeling Over Time",
                    "Long-Tail Discovery and Serendipity Architecture",
                    "Adaptive Teaching, Scaffolding, and Dialogue",
                    "RAG, Knowledge Graphs, and Agentic Workflows",
                    "Niche Content Verification and Hallucination Controls",
                    "Engagement, Retention, and Learning Outcome Metrics",
                ],
            )

        numbered = re.findall(
            r"(?:^|\s)\d{1,2}[\.\)]\s*([A-Z][^0-9]{4,160}?)(?=\s+\d{1,2}[\.\)]\s*[A-Z]|\s+Rank\b|\s+Present\b|$)",
            normalized,
        )
        if len(numbered) >= 4 and re.search(r"\b(report|research|comprehensive|deep research)\b", lowered):
            sections = ["Executive Summary and Scope"]
            if re.search(r"\b(rank|ranking|ranked)\b", lowered):
                sections.append("Ranking Framework and Priority Order")
            for item in numbered[:5]:
                title = re.split(r"\s+[–—-]\s+", re.sub(r"\s+", " ", item).strip(), maxsplit=1)[0]
                sections.append(DeepResearcherAgent._title_case_phrase(title)[:100].rstrip(" .,:;"))
            return (DeepResearcherAgent._fallback_plan_topic(normalized), sections[:7])

        return None

    @staticmethod
    def _extract_structured_lesson_scope(text: str | None) -> dict[str, Any] | None:
        """Extract content-lesson prompts with a broad topic and narrower debate motion.

        These API prompts already contain enough structure to build a safer plan
        deterministically. Preloading that plan prevents the orchestrator from
        collapsing the whole run onto the motion-specific branch.
        """
        if not text:
            return None

        topic_match = EXACT_LESSON_TOPIC_RE.search(text)
        motion_match = FINAL_DEBATE_MOTION_RE.search(text)
        if not topic_match or not motion_match:
            return None

        topic = topic_match.group("topic").strip().rstrip(".")
        motion = motion_match.group("motion").strip()
        if not topic or not motion:
            return None

        report_type_match = REPORT_TYPE_RE.search(text)
        audience_match = AUDIENCE_RE.search(text)
        sections_match = MARKDOWN_SECTION_LIST_RE.search(text)
        sections: list[str] = []
        if sections_match:
            for line in sections_match.group("sections").splitlines():
                section_match = re.match(r"\s*\d+\.\s*(.+?)\s*$", line)
                if section_match:
                    sections.append(section_match.group(1).strip().rstrip("."))
        if not sections:
            numbered_sections = re.findall(r"^\s*\d{1,2}\.\s*(.+?)\s*$", text, flags=re.MULTILINE)
            if len(numbered_sections) >= 4:
                sections = [
                    re.sub(r"\s+", " ", section).strip().rstrip(".")
                    for section in numbered_sections
                    if len(section.strip()) >= 3
                ][:16]

        glossary: dict[str, str] = {}
        if re.search(r"\bOT\b", text):
            glossary["OT"] = "Official Teams"

        return {
            "topic": topic,
            "motion": motion,
            "report_type": report_type_match.group("report_type").strip() if report_type_match else None,
            "audience": audience_match.group("audience").strip() if audience_match else None,
            "sections": sections,
            "glossary": glossary,
        }

    @staticmethod
    def _clean_training_topic(topic: str) -> str:
        """Return a compact primary subject from a training/deck request."""
        topic = re.sub(r"\s+", " ", topic).strip(" .,:;\"'“”")
        topic = re.split(
            r"\b(?:Core Requirements|Research Topics|Output Format|Use every tool|Do not limit|Act autonomously)\b",
            topic,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;\"'“”")
        topic = re.sub(r"^(?:all\s+of\s+)?", "", topic, flags=re.IGNORECASE).strip()
        return topic[:180].rstrip(" .,:;") or "Research Topic"

    @staticmethod
    def _extract_training_content_scope(text: str | None) -> dict[str, Any] | None:
        """Classify broad training/deck prompts without making the training frame the topic."""
        if not text or not TRAINING_CONTENT_KEYWORDS_RE.search(text):
            return None

        topic = ""
        for pattern in TRAINING_TOPIC_PATTERNS:
            match = pattern.search(text)
            if match:
                topic = DeepResearcherAgent._clean_training_topic(match.group("topic"))
                break
        if not topic:
            return None

        audience_terms: list[str] = []
        lowered = text.lower()
        if "wsdc" in lowered:
            audience_terms.append("WSDC")
        if re.search(r"\bbp\b", text, flags=re.IGNORECASE):
            audience_terms.append("BP")
        if "competitive debate" in lowered:
            audience_terms.append("competitive debate")
        if "coach" in lowered:
            audience_terms.append("coaches")
        if "debater" in lowered:
            audience_terms.append("debaters")

        if re.search(r"\bslide[- ]deck|slides?\b", lowered):
            deliverable = "slide-deck-ready training dossier"
        elif "training session" in lowered:
            deliverable = "training-session research dossier"
        else:
            deliverable = "training-oriented research dossier"

        return {
            "topic": topic,
            "deliverable": deliverable,
            "audience": ", ".join(dict.fromkeys(audience_terms)) or None,
            "application_context": "WSDC/BP competitive debate training"
            if "wsdc" in lowered or re.search(r"\bbp\b", text, flags=re.IGNORECASE)
            else "training application",
        }

    @staticmethod
    def _source_strategy_for_claims() -> dict[str, Any]:
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
                "Use primary/authoritative sources for numbers; if unavailable, label secondary-source "
                "numbers as partially verified."
            ),
        }

    @staticmethod
    def _target_claim(
        claim_id: str,
        claim_type: str,
        claim: str,
        required_source_class: str,
    ) -> dict[str, str]:
        return {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "claim": claim,
            "required_source_class": required_source_class,
        }

    @staticmethod
    def _apply_query_budget_allocation(
        queries: list[dict[str, Any]],
        budget_profile: dict[str, Any],
        *,
        default_category: str = "evidence",
    ) -> list[dict[str, Any]]:
        """Attach generic task budget fields to researcher assignments."""
        if not queries:
            return queries

        weights = [max(1, int(query.get("relevance_weight") or 1)) for query in queries]
        weight_total = sum(weights) or 1
        active_budget = max(len(queries), int(budget_profile.get("active_search_calls") or len(queries)))
        per_task_cap = max(1, int(budget_profile.get("search_calls_per_task") or active_budget))

        allocated: list[dict[str, Any]] = []
        percents: list[float] = [round(weight / weight_total * 100.0, 1) for weight in weights]
        percents[-1] = round(percents[-1] + (100.0 - sum(percents)), 1)
        for index, query in enumerate(queries, start=1):
            task = dict(query)
            percent = float(task.get("budget_percent") or percents[index - 1])
            task["task_id"] = task.get("task_id") or f"Q{index}"
            task["task_category"] = task.get("task_category") or default_category
            task["relevance_weight"] = weights[index - 1]
            task["budget_percent"] = percent
            task["search_budget"] = int(
                task.get("search_budget") or max(1, min(per_task_cap, round(active_budget * percent / 100.0)))
            )
            allocated.append(task)
        return allocated

    @staticmethod
    def _budget_profile_for_state(
        state: DeepResearchAgentState,
        *,
        mode_override: str | None = None,
        section_count_override: int | None = None,
    ) -> dict[str, Any]:
        """Return a human-readable search budget plan for the current request."""
        depth_config = get_research_depth_config(state.research_depth)
        latest_query = DeepResearcherAgent._query_without_context(DeepResearcherAgent._latest_user_text(state))
        structured_scope = DeepResearcherAgent._extract_structured_lesson_scope(latest_query)
        training_scope = None if structured_scope else DeepResearcherAgent._extract_training_content_scope(latest_query)
        approved_plan = None
        if not structured_scope and not training_scope:
            approved_plan = DeepResearcherAgent._extract_approved_plan(state.clarifier_result or latest_query)

        mode = mode_override
        if mode is None:
            if structured_scope:
                mode = "lesson_first"
            elif training_scope:
                mode = "lesson_first"
            elif DeepResearcherAgent._is_financial_screen_query(latest_query):
                mode = "focused_screen"
            else:
                mode = "symmetric"

        section_count = section_count_override
        if section_count is None:
            if structured_scope:
                section_count = max(1, len(structured_scope.get("sections") or []))
            elif training_scope:
                section_count = 6
            elif approved_plan:
                section_count = max(1, len(approved_plan[1]))
            else:
                section_count = 4

        return depth_config.budget_profile(mode=mode, section_count=section_count)

    @staticmethod
    def _build_structured_lesson_plan_json(scope: dict[str, Any], query: str, depth_config) -> str:
        """Create a topic-first plan for structured lesson/dossier API prompts."""
        topic = str(scope["topic"])
        motion = str(scope["motion"])
        report_type = scope.get("report_type") or "content_research"
        audience = scope.get("audience") or "the stated audience"
        glossary = scope.get("glossary") or {}
        glossary_note = (
            "; ".join(f"{key} means {value}" for key, value in glossary.items())
            if glossary
            else "No abbreviation glossary supplied."
        )
        sections = DeepResearcherAgent._structured_lesson_section_titles(scope)
        budget_profile = depth_config.budget_profile(mode="lesson_first", section_count=len(sections))

        toc = [
            {
                "id": str(index + 1),
                "title": section,
                "subsections": [],
            }
            for index, section in enumerate(sections)
        ]

        broad_sections = sections[: max(3, min(6, len(sections)))]
        example_sections = [
            section
            for section in sections
            if any(keyword in section.lower() for keyword in ("finding", "example", "case", "tension", "misconception"))
        ] or sections
        motion_sections = [
            section
            for section in sections
            if any(keyword in section.lower() for keyword in ("motion", "bridge", "relevance", "scope", "anchor"))
        ] or [sections[-1]]

        queries = [
            {
                "query": (
                    f"{topic} broad lesson background definitions key terms core concepts stakeholders "
                    f"systems and age-appropriate explanations for {audience}; do not center the final debate motion"
                ),
                "tool": "advanced_web_search_tool",
                "target_sections": broad_sections,
                "rationale": (
                    "Cover the exact lesson topic first so the report teaches the broader content before any "
                    "motion-specific debate branch."
                ),
                "task_category": "foundations",
                "relevance_weight": 5,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C1",
                        "definition",
                        f"Core definitions, terms, stakeholders, and systems needed to explain {topic}",
                        "third_party_authoritative",
                    ),
                    DeepResearcherAgent._target_claim(
                        "C2",
                        "discovery",
                        f"Important background facts and misconceptions about {topic}",
                        "mixed",
                    ),
                ],
            },
            {
                "query": (
                    f"{topic} factual findings reputable sources concrete examples case studies misconceptions "
                    f"and visual teaching opportunities for {audience}"
                ),
                "tool": "advanced_web_search_tool",
                "target_sections": example_sections,
                "rationale": (
                    "Collect source-backed raw material and examples that teach the topic itself, not only one "
                    "narrow argument or policy controversy."
                ),
                "task_category": "examples",
                "relevance_weight": 3,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C3",
                        "discovery",
                        f"Concrete examples and case studies that accurately teach {topic}",
                        "mixed",
                    ),
                    DeepResearcherAgent._target_claim(
                        "C4",
                        "recommendation",
                        f"Teaching-relevant framing and visual opportunities for {topic}",
                        "practitioner",
                    ),
                ],
            },
            {
                "query": (
                    f"{topic} final debate motion {motion} bounded motion relevance ethical legal tradeoffs "
                    "case examples; keep as a bridge from the broader topic, not the whole report"
                ),
                "tool": "advanced_web_search_tool",
                "target_sections": motion_sections,
                "rationale": (
                    "Gather enough evidence to connect the broader lesson to the final motion without allowing "
                    "the motion to replace the lesson topic."
                ),
                "task_category": "bridge",
                "relevance_weight": 1,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C5",
                        "comparative",
                        f"How the final motion relates to {topic} without replacing the broader lesson scope",
                        "mixed",
                    ),
                ],
            },
        ]

        plan = {
            "task_analysis": {
                "user_intent": DeepResearcherAgent._query_without_context(query),
                "claim_profile": {
                    "definition_claims": [
                        DeepResearcherAgent._target_claim(
                            "C1",
                            "definition",
                            f"Core definitions, terms, stakeholders, and systems needed to explain {topic}",
                            "third_party_authoritative",
                        )
                    ],
                    "discovery_claims": [
                        DeepResearcherAgent._target_claim(
                            "C2", "discovery", f"Important background facts and misconceptions about {topic}", "mixed"
                        ),
                        DeepResearcherAgent._target_claim(
                            "C3",
                            "discovery",
                            f"Concrete examples and case studies that accurately teach {topic}",
                            "mixed",
                        ),
                    ],
                    "recommendation_claims": [
                        DeepResearcherAgent._target_claim(
                            "C4",
                            "recommendation",
                            f"Teaching-relevant framing and visual opportunities for {topic}",
                            "practitioner",
                        )
                    ],
                    "comparative_claims": [
                        DeepResearcherAgent._target_claim(
                            "C5",
                            "comparative",
                            f"How the final motion relates to {topic} without replacing the broader lesson scope",
                            "mixed",
                        )
                    ],
                    "claim_density_estimate": "medium",
                    "verifiability_estimate": "mixed",
                },
                "source_strategy": DeepResearcherAgent._source_strategy_for_claims(),
                "budget_profile": budget_profile,
                "structured_lesson_scope_used_directly": True,
                "exact_lesson_topic": topic,
                "final_debate_motion": motion,
                "report_type": report_type,
                "glossary": glossary,
                "out_of_scope": [
                    "Renaming the lesson around only the final motion",
                    "Letting an adjacent theme or single controversy replace the exact lesson topic",
                    "Writing a prop/opp debate case file",
                ],
            },
            "report_title": f"{topic} Content Research Dossier",
            "report_toc": toc,
            "budget_profile": budget_profile,
            "constraints": [
                {
                    "category": "content",
                    "constraint": (
                        f"The exact lesson topic '{topic}' is the primary scope. Research and report structure "
                        "must teach that broad topic before discussing the final debate motion."
                    ),
                    "rationale": (
                        "The final motion is an anchor for later debate, not permission to narrow the whole dossier."
                    ),
                    "verification": (
                        "The opening title, topic essentials, factual findings, and examples visibly cover "
                        "the exact topic."
                    ),
                },
                {
                    "category": "content",
                    "constraint": (
                        "Motion-specific material must be subordinate: use it in scope/anchor, bridge, relevance, "
                        "and selected examples, but do not make every researcher task or report section about it."
                    ),
                    "rationale": (
                        "Prevents broad content lessons from drifting into a single motion-specific controversy."
                    ),
                    "verification": (
                        "At least two researcher outputs and at least half the report body cover the broad topic."
                    ),
                },
                {
                    "category": "structure",
                    "constraint": (
                        "Preserve the requested Markdown sections and write a teacher-facing content dossier, "
                        "not a debate case file."
                    ),
                    "rationale": "The API prompt requested structured raw material for a lesson writer.",
                    "verification": (
                        "All requested sections appear and no Arguments For/Against/Rebuttals section is introduced."
                    ),
                },
                {
                    "category": "terminology",
                    "constraint": (
                        f"Interpret curriculum shorthand using this glossary: {glossary_note} "
                        "Do not expand abbreviations into unrelated meanings."
                    ),
                    "rationale": "Prevents lesson metadata abbreviations from becoming false research scope.",
                    "verification": "The report and queries do not reinterpret glossary abbreviations.",
                },
                {
                    "category": "source",
                    "constraint": (
                        "Use reputable official, academic, legal, medical, policy, or journalism sources, and cite "
                        "concrete claims close to where they are made."
                    ),
                    "rationale": "The lesson writer needs reliable raw material for student-facing slides.",
                    "verification": (
                        "Source table and bibliography contain source-backed claims across broad-topic and "
                        "motion sections."
                    ),
                },
            ],
            "output_style": {
                "mode": "standard_report",
                "topic_anchor": topic,
                "motion_anchor": motion,
                "target": "Teacher-facing Markdown content dossier that teaches the exact lesson topic first.",
                "avoid": [
                    "A report centered only on the final debate motion",
                    "A report centered only on one adjacent controversy",
                    "Prop/opp case-file structure",
                ],
            },
            "queries": DeepResearcherAgent._apply_query_budget_allocation(
                queries,
                budget_profile,
                default_category="lesson_evidence",
            ),
        }
        return json.dumps(plan, indent=2)

    @staticmethod
    def _build_training_content_plan_json(scope: dict[str, Any], query: str, depth_config) -> str:
        """Create a topic-first plan for broad training or slide-deck research prompts."""
        topic = str(scope["topic"])
        deliverable = str(scope.get("deliverable") or "training-oriented research dossier")
        audience = scope.get("audience") or "the stated training audience"
        application_context = str(scope.get("application_context") or "training application")
        sections = [
            "Core Concepts and Philosophical Foundations",
            "Theories, Mechanisms, and Causal Models",
            "Success Metrics, Evidence Standards, and Critiques",
            "Tactics, Organization, Digital Media, and Repression",
            "Intersectionality, Backlash, and Global Perspectives",
            "Comparative Case Studies and Debate Training Applications",
        ]
        budget_profile = depth_config.budget_profile(mode="lesson_first", section_count=len(sections))
        toc = [{"id": str(index + 1), "title": section, "subsections": []} for index, section in enumerate(sections)]
        queries = [
            {
                "query": (
                    f"Research {topic} foundations for a {deliverable}: definitions, philosophy, core concepts, "
                    f"stakeholders, and why the topic matters for {audience}."
                ),
                "tool": "advanced_web_search_tool",
                "seed_queries": [
                    f"{topic} definitions philosophical foundations academic",
                    f"{topic} social theory key concepts",
                    f"{topic} overview academic debate education",
                ],
                "target_sections": sections[:2],
                "rationale": "Lock the primary subject before applying the training or debate frame.",
                "task_category": "foundations",
                "relevance_weight": 5,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C1",
                        "definition",
                        f"Core definitions and philosophical foundations needed to teach {topic}",
                        "academic",
                    ),
                    DeepResearcherAgent._target_claim(
                        "C2",
                        "causal",
                        f"Major theories and mechanisms that explain {topic}",
                        "academic",
                    ),
                ],
            },
            {
                "query": (
                    f"Research empirical evidence, success metrics, quantitative claims, critiques, and replication "
                    f"debates relevant to {topic}."
                ),
                "tool": "advanced_web_search_tool",
                "seed_queries": [
                    f"{topic} success metrics empirical studies",
                    f"{topic} quantitative evidence critiques",
                    f"{topic} replication critique academic",
                ],
                "target_sections": [sections[2]],
                "rationale": "High-risk numbers and causal claims need primary or academic verification.",
                "task_category": "primary_data",
                "relevance_weight": 4,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C3",
                        "quantitative",
                        f"Authoritative metrics, statistics, and empirical findings about {topic}",
                        "primary_issuer",
                    ),
                    DeepResearcherAgent._target_claim(
                        "C4",
                        "discovery",
                        f"Important critiques, failed cases, and limitations in evidence about {topic}",
                        "academic",
                    ),
                ],
            },
            {
                "query": (
                    f"Research practical mechanisms in {topic}: tactics, leadership, coalitions, digital media, "
                    "surveillance, repression, and transnational diffusion."
                ),
                "tool": "advanced_web_search_tool",
                "seed_queries": [
                    f"{topic} tactics leadership digital media",
                    f"{topic} repression surveillance transnational diffusion",
                    f"{topic} organizational strategy case studies",
                ],
                "target_sections": [sections[3]],
                "rationale": "The training dossier needs usable mechanisms, not only abstract theory.",
                "task_category": "mechanisms",
                "relevance_weight": 4,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C5",
                        "causal",
                        f"Mechanisms, tactics, and organizational choices that shape outcomes in {topic}",
                        "academic",
                    )
                ],
            },
            {
                "query": (
                    f"Research perspectives and counterevidence for {topic}: intersectionality, backlash, "
                    "co-optation, Global South and Asian perspectives, and opposing viewpoints."
                ),
                "tool": "advanced_web_search_tool",
                "seed_queries": [
                    f"{topic} intersectionality backlash critique",
                    f"{topic} Global South Asian perspectives",
                    f"{topic} opposing viewpoints limitations",
                ],
                "target_sections": [sections[4]],
                "rationale": "Prevents a one-sided training deck and surfaces debate-useful tensions.",
                "task_category": "counterevidence",
                "relevance_weight": 4,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C6",
                        "discovery",
                        f"Counterarguments, backlash patterns, and underrepresented perspectives on {topic}",
                        "mixed",
                    )
                ],
            },
            {
                "query": (
                    f"Research comparative case studies and translate {topic} evidence into {application_context}: "
                    "slide titles, speaker-note material, argument banks, motions, and teaching examples."
                ),
                "tool": "advanced_web_search_tool",
                "seed_queries": [
                    f"{topic} comparative case studies",
                    f"{topic} debate motions argument bank",
                    f"{topic} teaching slides speaker notes",
                ],
                "target_sections": [sections[5]],
                "rationale": "Keeps the debate-training frame as application and output shape, not the main subject.",
                "task_category": "application",
                "relevance_weight": 3,
                "target_claims": [
                    DeepResearcherAgent._target_claim(
                        "C7",
                        "discovery",
                        f"Comparative cases and training applications that accurately teach {topic}",
                        "mixed",
                    ),
                    DeepResearcherAgent._target_claim(
                        "C8",
                        "recommendation",
                        f"Slide-deck and debate-training structure that follows from evidence about {topic}",
                        "practitioner",
                    ),
                ],
            },
        ]
        plan = {
            "task_analysis": {
                "user_intent": DeepResearcherAgent._query_without_context(query),
                "scope_profile": {
                    "primary_subject": topic,
                    "deliverable_type": deliverable,
                    "audience": audience,
                    "application_context": application_context,
                    "scope_mode": "topic_first",
                    "budget_mode": "lesson_first",
                    "section_count_target": len(sections),
                    "secondary_contexts": [application_context],
                    "forbidden_reframes": [
                        "Treating WSDC/BP debate training as the primary research subject",
                        "Turning the dossier into only debate strategy or motion prep",
                        "Letting one case study or ideology replace the broader topic",
                    ],
                    "classification_reason": (
                        "The request asks for research on the primary subject to create a training deliverable; "
                        "the training frame is output/application context."
                    ),
                },
                "claim_profile": {
                    "claim_density_estimate": "high",
                    "verifiability_estimate": "mixed",
                    "claims": [],
                },
                "source_strategy": DeepResearcherAgent._source_strategy_for_claims(),
                "budget_profile": budget_profile,
                "training_content_scope_used_directly": True,
                "out_of_scope": [
                    "Reframing the primary subject as WSDC/BP itself",
                    "Generic debate pedagogy disconnected from the requested topic",
                ],
            },
            "report_title": f"{topic} Training Research Dossier",
            "report_toc": toc,
            "budget_profile": budget_profile,
            "constraints": [
                {
                    "category": "scope",
                    "constraint": (
                        f"The primary research subject is '{topic}'. The {application_context} frame controls "
                        "deliverable format and applications, but must not replace the subject."
                    ),
                    "rationale": "This prevents the planner from misclassifying the request as debate pedagogy only.",
                    "verification": (
                        "The title, opening sections, and most researcher tasks are about the topic itself."
                    ),
                },
                {
                    "category": "structure",
                    "constraint": (
                        "Use no more than six top-level research sections before final slide/application material."
                    ),
                    "rationale": "A compact section map preserves coverage without scattering the search budget.",
                    "verification": "The final report follows the six section groups in /shared/plan.json.",
                },
                {
                    "category": "source",
                    "constraint": (
                        "Prioritize academic, primary, NGO/government, and credible journalism sources for factual "
                        "claims; use debate/training sources only for application guidance."
                    ),
                    "rationale": "The slide deck should be based on strong topical evidence, not generic debate notes.",
                    "verification": "Source citations for substantive claims come from topic-appropriate sources.",
                },
            ],
            "output_style": {
                "mode": "lesson_first",
                "topic_anchor": topic,
                "motion_anchor": None,
                "target": f"{deliverable} centered on {topic}, with debate applications at the end.",
                "avoid": [
                    "A report about WSDC/BP as the primary topic",
                    "A motion-only or debate-case-only structure",
                    "Generic debate training detached from the topic",
                ],
            },
            "queries": DeepResearcherAgent._apply_query_budget_allocation(
                queries,
                budget_profile,
                default_category="training_evidence",
            ),
        }
        return json.dumps(plan, indent=2)

    @staticmethod
    def _structured_lesson_section_titles(scope: dict[str, Any]) -> list[str]:
        """Return requested lesson/dossier section titles, with the current default shape."""
        sections = list(scope.get("sections") or [])
        if sections:
            return [str(section) for section in sections]
        return [
            "Research scope and final motion anchor",
            "Age and audience assumptions",
            "Topic essentials: definitions, key terms, and background concepts",
            "Factual findings with citations",
            "Key examples and case studies",
            "Important tensions and misconceptions",
            "Motion relevance notes",
            "Source table and final bibliography",
        ]

    @staticmethod
    def _build_plan_json_from_approved_context(
        title: str,
        sections: list[str],
        query: str,
        depth_config,
    ) -> str:
        """Create a compact /shared/plan.json from an already approved user plan."""
        clean_query = DeepResearcherAgent._query_without_context(query)
        is_financial_screen = DeepResearcherAgent._is_financial_screen_query(clean_query)
        budget_profile = depth_config.budget_profile(
            mode="focused_screen" if is_financial_screen else "symmetric",
            section_count=len(sections),
        )

        toc = [
            {
                "id": str(index + 1),
                "title": section,
                "subsections": [],
            }
            for index, section in enumerate(sections)
        ]

        if is_financial_screen:
            queries = [
                {
                    "query": (
                        "U.S. stocks 40% 50% below fair value current price fair value estimate "
                        "GuruFocus Morningstar analyst consensus; validate current quotes and methodology caveats"
                    ),
                    "tool": "advanced_web_search_tool",
                    "target_sections": sections,
                    "rationale": (
                        "Find and validate source-backed fair-value discount candidates in one bounded "
                        "researcher task without expanding the scope."
                    ),
                    "task_category": "candidate_screening",
                    "relevance_weight": 5,
                    "target_claims": [
                        DeepResearcherAgent._target_claim(
                            "C1",
                            "quantitative",
                            "Current ticker price, fair-value estimate, and discount percentage for each candidate",
                            "primary_issuer",
                        ),
                        DeepResearcherAgent._target_claim(
                            "C2",
                            "specification",
                            "Methodology caveats behind each cited fair-value estimate",
                            "third_party_authoritative",
                        ),
                    ],
                },
            ]
            constraints = [
                "Only include candidates with current price, fair-value estimate, discount percentage, and source.",
                "Keep the answer to the approved sections; do not add generic value-investing background.",
                "Use stock_quote_tool for current quote snapshots after identifying ticker candidates.",
                "Avoid buy/sell recommendations; present source-backed market research and caveats.",
                (
                    "Use exactly one researcher-agent task for this narrow stock screen; do not split into "
                    "parallel duplicate tasks."
                ),
                (
                    "Write a compact final report: table first, then concise evidence notes and caveats. "
                    "Avoid long methodology essays."
                ),
            ]
            output_style = {
                "mode": "focused_screen",
                "target": (
                    "Compact answer with a candidate table, source-backed current prices, fair-value estimates, "
                    "discount math, and caveats."
                ),
                "avoid": [
                    "Generic value-investing background",
                    "Long methodology sections",
                    "Recommendations",
                ],
            }
        else:
            queries = [
                {
                    "query": f"{clean_query} {section}",
                    "tool": "advanced_web_search_tool",
                    "target_sections": [section],
                    "rationale": "Answer the approved plan section directly.",
                    "task_category": "evidence",
                    "relevance_weight": 1,
                    "target_claims": [
                        DeepResearcherAgent._target_claim(
                            f"C{index + 1}",
                            "discovery",
                            f"Evidence needed to answer approved section: {section}",
                            "mixed",
                        )
                    ],
                }
                for index, section in enumerate(sections)
            ]
            constraints = [
                "Keep the report no broader than the approved plan.",
                "Cover each approved section directly.",
                "Use source-backed facts and cite returned URLs.",
            ]
            output_style = {
                "mode": "standard_report",
                "target": "Depth should match the approved sections and user intent.",
                "avoid": ["Expanding beyond the approved plan"],
            }

        plan = {
            "task_analysis": {
                "user_intent": clean_query,
                "claim_profile": {
                    "discovery_claims": [
                        DeepResearcherAgent._target_claim(
                            f"C{index + 1}",
                            "discovery",
                            f"Evidence needed to answer approved section: {section}",
                            "mixed",
                        )
                        for index, section in enumerate(sections)
                    ],
                    "quantitative_claims": [
                        DeepResearcherAgent._target_claim(
                            "C1",
                            "quantitative",
                            "Current ticker price, fair-value estimate, and discount percentage for each candidate",
                            "primary_issuer",
                        )
                    ]
                    if is_financial_screen
                    else [],
                    "claim_density_estimate": "medium",
                    "verifiability_estimate": "mixed",
                },
                "source_strategy": DeepResearcherAgent._source_strategy_for_claims(),
                "budget_profile": budget_profile,
                "approved_plan_used_directly": True,
                "out_of_scope": ["Expanding beyond the approved plan", "Generic background not requested by the user"],
            },
            "report_title": title,
            "report_toc": toc,
            "budget_profile": budget_profile,
            "constraints": constraints,
            "output_style": output_style,
            "queries": DeepResearcherAgent._apply_query_budget_allocation(queries, budget_profile),
        }
        return json.dumps(plan, indent=2)

    @staticmethod
    def _file_state_entry(content: str | list[str], *, created_at: str | None = None) -> dict[str, Any]:
        """Build deepagents-compatible virtual file data with required metadata."""
        now = datetime.now(UTC).isoformat()
        lines = content.splitlines() if isinstance(content, str) else [str(line) for line in content]
        return {
            "content": lines,
            "created_at": created_at or now,
            "modified_at": now,
        }

    @staticmethod
    def _normalize_files_state(files: dict[str, Any] | None) -> dict[str, Any]:
        """Ensure preloaded/checkpointed files match deepagents FileData shape."""
        if not files:
            return {}

        normalized: dict[str, Any] = {}
        now = datetime.now(UTC).isoformat()
        for path, value in files.items():
            if value is None:
                continue
            if isinstance(value, dict):
                content = value.get("content", [])
                if isinstance(content, str):
                    content_lines = content.splitlines()
                elif isinstance(content, list):
                    content_lines = [str(line) for line in content]
                else:
                    content_lines = [str(content)]
                created_at = str(value.get("created_at") or now)
                normalized[path] = {
                    **value,
                    "content": content_lines,
                    "created_at": created_at,
                    "modified_at": str(value.get("modified_at") or created_at),
                }
            elif isinstance(value, str):
                normalized[path] = DeepResearcherAgent._file_state_entry(value)
            elif isinstance(value, list):
                normalized[path] = DeepResearcherAgent._file_state_entry(value)
        return normalized

    def _seed_source_registry_from_files(self, files: dict[str, Any]) -> None:
        """Rehydrate verified sources from resume artifacts in virtual files."""
        if not files:
            return

        registry = self.source_registry_middleware._get_registry()
        seeded = 0
        for value in files.values():
            content = self._coerce_file_content(value)
            if self._is_unavailable_source_context(content):
                continue
            for url in re.findall(r"https?://[^\s<>'\")\]}]+", content):
                url = url.rstrip(".,;:")
                if url in {"https://exa.ai/", "https://exa.ai"}:
                    continue
                registry.add(SourceEntry(url=url, source_type="resume", tool_name="resume"))
                seeded += 1
        if seeded:
            logger.info("Deep Research: seeded %d source URL(s) from resume files", seeded)

    def _inject_approved_plan_if_available(self, state: DeepResearchAgentState) -> DeepResearchAgentState:
        """Carry approved-plan context and seed a fresh current-request plan floor.

        Stale/checkpointed plan files are always removed first. For structured
        lesson prompts or already-approved rich plans, seed a deterministic
        current-request plan so the orchestrator has an executable floor if a
        planner subagent write does not propagate across the shared filesystem.
        The orchestrator prompt still tells it to call planner-agent and refine
        the plan rather than treating the floor as a reason to skip planning.
        """
        files = self._normalize_files_state(state.files)
        stale_plan_paths = {"/plan.json", "plan.json", "/shared/plan.json", "shared/plan.json"}
        files = {path: value for path, value in files.items() if path not in stale_plan_paths}

        latest_query = self._latest_user_text(state)
        clean_query = self._query_without_context(latest_query)
        depth_config = get_research_depth_config(state.research_depth)
        structured_scope = self._extract_structured_lesson_scope(clean_query)
        if structured_scope:
            logger.info("Deep Research: seeding current-request structured lesson plan floor")
            plan_context = self._format_structured_lesson_context(structured_scope)
            plan_json = self._build_structured_lesson_plan_json(structured_scope, latest_query, depth_config)
            files = self._with_plan_floor(files, plan_json)
            return state.model_copy(update={"files": files, "clarifier_result": plan_context})

        training_scope = self._extract_training_content_scope(clean_query)
        if training_scope:
            logger.info("Deep Research: seeding current-request training-content scope plan floor")
            plan_context = self._format_approved_plan_context(
                f"{training_scope['topic']} Training Research Dossier",
                [
                    "Core Concepts and Philosophical Foundations",
                    "Theories, Mechanisms, and Causal Models",
                    "Success Metrics, Evidence Standards, and Critiques",
                    "Tactics, Organization, Digital Media, and Repression",
                    "Intersectionality, Backlash, and Global Perspectives",
                    "Comparative Case Studies and Debate Training Applications",
                ],
            )
            plan_context += (
                "\n\nPlanner Context Notes:\n"
                f"- Primary subject: {training_scope['topic']}\n"
                f"- Deliverable/application context: {training_scope.get('application_context')}\n"
                "- Treat the training/debate frame as output shape and application, not as the main research topic.\n"
                "- Keep the executable plan topic-first with no more than six top-level sections."
            )
            plan_json = self._build_training_content_plan_json(training_scope, latest_query, depth_config)
            files = self._with_plan_floor(files, plan_json)
            return state.model_copy(update={"files": files, "clarifier_result": plan_context})

        plan_context = state.clarifier_result or latest_query
        approved_plan = self._extract_approved_plan(plan_context)
        if not approved_plan and state.clarifier_result:
            approved_plan = self._extract_approved_plan(latest_query)
        if not approved_plan:
            explicit_plan = self._explicit_plan_from_rich_query(latest_query)
            if explicit_plan:
                title, sections = explicit_plan
                logger.info("Deep Research: seeding current-request rich-query plan floor")
                plan_context = self._format_approved_plan_context(title, sections)
                plan_json = self._build_plan_json_from_approved_context(title, sections, latest_query, depth_config)
                files = self._with_plan_floor(files, plan_json)
                return state.model_copy(update={"files": files, "clarifier_result": plan_context})
            if self._is_financial_screen_query(clean_query):
                title = "Stocks Trading 40-50% Below Estimated Fair Value"
                sections = [
                    "Candidate stocks matching the requested discount range",
                    "Current price and fair-value evidence",
                    "Methodology caveats and source limitations",
                ]
                logger.info("Deep Research: seeding current-request focused stock-screen plan floor")
                plan_context = self._format_approved_plan_context(title, sections)
                plan_json = self._build_plan_json_from_approved_context(title, sections, latest_query, depth_config)
                files = self._with_plan_floor(files, plan_json)
                return state.model_copy(update={"files": files, "clarifier_result": plan_context})
            return state.model_copy(update={"files": files})

        title, sections = approved_plan
        if self._is_generic_approved_plan(title, sections):
            explicit_plan = self._explicit_plan_from_rich_query(latest_query)
            if explicit_plan:
                title, sections = explicit_plan
                logger.info("Deep Research: replacing generic approved plan with rich-query plan floor")
                plan_context = self._format_approved_plan_context(title, sections)
                plan_json = self._build_plan_json_from_approved_context(title, sections, latest_query, depth_config)
                files = self._with_plan_floor(files, plan_json)
                return state.model_copy(update={"files": files, "clarifier_result": plan_context})
            logger.info("Deep Research: ignoring generic approved plan so planner-agent can create a specific TOC")
            return state.model_copy(
                update={
                    "files": files,
                    "clarifier_result": self._strip_approved_plan_context(state.clarifier_result),
                }
            )

        logger.info("Deep Research: seeding current-request approved-plan floor")
        plan_json = self._build_plan_json_from_approved_context(title, sections, latest_query, depth_config)
        files = self._with_plan_floor(files, plan_json)
        return state.model_copy(update={"files": files})

    @staticmethod
    def _with_plan_floor(files: dict[str, Any], plan_json: str) -> dict[str, Any]:
        """Add a fresh executable plan to both canonical DeepAgents plan paths."""
        updated = dict(files)
        entry = DeepResearcherAgent._file_state_entry(plan_json)
        updated["/shared/plan.json"] = entry
        updated["/plan.json"] = entry
        return updated

    @staticmethod
    def _tool_limits_for_state(state: DeepResearchAgentState) -> dict[str, int]:
        """Return per-run expensive-tool limits based on task scope."""
        latest_query = DeepResearcherAgent._query_without_context(DeepResearcherAgent._latest_user_text(state))
        if DeepResearcherAgent._is_financial_screen_query(latest_query):
            return {
                "planner:search": 2,
                "planner:exa_web_search_tool": 2,
                "planner:advanced_web_search_tool": 2,
                "planner:web_search_tool": 2,
                "search": 4,
                "exa_web_search_tool": 4,
                "advanced_web_search_tool": 4,
                "web_search_tool": 4,
                "stock_quote_tool": 2,
            }
        depth_config = get_research_depth_config(state.research_depth)
        budget_profile = DeepResearcherAgent._budget_profile_for_state(state)
        search_limit = int(budget_profile.get("total_search_calls") or depth_config.advanced_web_search_limit)
        return {
            "planner:search": depth_config.planner_search_limit,
            "planner:exa_web_search_tool": depth_config.planner_search_limit,
            "planner:advanced_web_search_tool": depth_config.planner_search_limit,
            "planner:web_search_tool": min(depth_config.web_search_limit, depth_config.planner_search_limit),
            "search": search_limit,
            "exa_web_search_tool": depth_config.advanced_web_search_limit,
            "advanced_web_search_tool": depth_config.advanced_web_search_limit,
            "web_search_tool": depth_config.web_search_limit,
            "stock_quote_tool": depth_config.stock_quote_limit,
        }

    @staticmethod
    def _parallel_tool_limits_for_state(state: DeepResearchAgentState) -> dict[str, int]:
        """Return per-response parallel tool limits for long-running agent tools."""
        depth_config = get_research_depth_config(state.research_depth)
        return {"task": depth_config.max_parallel_researcher_tasks}

    @staticmethod
    def _result_messages(result: Any) -> list[Any]:
        """Return a mutable list of messages from a deepagents result payload."""
        if isinstance(result, dict):
            messages = result.get("messages")
            return list(messages) if isinstance(messages, list) else []
        messages = getattr(result, "messages", None)
        return list(messages) if isinstance(messages, list) else []

    @staticmethod
    def _looks_like_provider_payload(content: str) -> bool:
        """Reject raw provider protocol blocks as report prose."""
        stripped = content.lstrip()
        if (
            stripped.startswith("[{'thinking'")
            or stripped.startswith('[{"thinking"')
            or stripped.startswith("{'thinking'")
            or stripped.startswith('{"thinking"')
            or stripped.startswith("[{'signature'")
            or stripped.startswith('[{"signature"')
        ):
            return True
        lowered = content.lower()
        return any(
            marker in lowered
            for marker in (
                "'type': 'thinking'",
                '"type": "thinking"',
                "'type': 'tool_use'",
                '"type": "tool_use"',
                "input_json_delta",
            )
        )

    @staticmethod
    def _coerce_file_content(value: Any) -> str:
        """Convert deepagents file-state content into plain text."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(str(line) for line in value)
        if isinstance(value, dict):
            return DeepResearcherAgent._coerce_file_content(value.get("content", ""))
        return ""

    @staticmethod
    def _extract_report_file_content(result: dict | Any) -> str:
        """Extract /report.md from deepagents state when the model wrote a file."""
        if isinstance(result, dict):
            files = result.get("files") or {}
        else:
            files = getattr(result, "files", {}) or {}

        if not isinstance(files, dict):
            return ""

        candidates = (
            files.get("/report.md"),
            files.get("report.md"),
        )
        best = ""
        for candidate in candidates:
            content = DeepResearcherAgent._coerce_file_content(candidate)
            if DeepResearcherAgent._looks_like_provider_payload(content):
                continue
            if len(content) > len(best):
                best = content
        return best

    @staticmethod
    def _extract_research_notes_content(result: dict | Any) -> str:
        """Extract researcher note files as a last-resort report fallback."""
        if isinstance(result, dict):
            files = result.get("files") or {}
        else:
            files = getattr(result, "files", {}) or {}

        if not isinstance(files, dict):
            return ""

        ignored_paths = {
            "/plan.json",
            "plan.json",
            "/shared/plan.json",
            "shared/plan.json",
            "/shared/consolidated_findings.md",
            "shared/consolidated_findings.md",
            "/report.md",
            "report.md",
        }
        note_blocks = []
        for path, value in sorted(files.items()):
            normalized_path = str(path).lstrip("/")
            if str(path) in ignored_paths or normalized_path in ignored_paths:
                continue
            if not (str(path).startswith("/shared/") or normalized_path.startswith("shared/")):
                continue
            if not (normalized_path.endswith(".txt") or normalized_path.endswith(".md")):
                continue

            content = DeepResearcherAgent._coerce_file_content(value).strip()
            if len(content) < 200:
                continue
            note_blocks.append(f"### {Path(normalized_path).name}\n\n{content}")

        if not note_blocks:
            return ""

        return "\n\n".join(note_blocks)

    def _build_report_from_research_notes(self, result: dict | Any) -> str:
        """Render a usable report from researcher notes when final synthesis fails."""
        notes = self._extract_research_notes_content(result)
        if not notes:
            return ""

        source_list = self.source_registry_middleware.get_source_list_text() or ""
        sources_section = f"\n\n## Sources\n\n{source_list}" if source_list and "## Sources" not in notes else ""
        return (
            "# Research Findings\n\n"
            "Based on the evidence gathered for this job, the following findings were available.\n\n"
            "## Evidence Notes\n\n"
            f"{notes}"
            f"{sources_section}"
        )

    @staticmethod
    def _extract_files(result: dict | Any) -> dict[str, Any]:
        if isinstance(result, dict):
            files = result.get("files") or {}
        else:
            files = getattr(result, "files", {}) or {}
        return files if isinstance(files, dict) else {}

    @staticmethod
    def _set_files(result: dict | Any, files: dict[str, Any]) -> None:
        if isinstance(result, dict):
            result["files"] = files
        elif hasattr(result, "files"):
            result.files = files

    @staticmethod
    def _backend_glob_paths(backend: StateBackend, pattern: str) -> list[str]:
        """Return paths from a DeepAgents StateBackend glob result."""
        try:
            result = backend.glob(pattern)
        except Exception:
            logger.debug("Unable to glob virtual filesystem pattern %s", pattern, exc_info=True)
            return []
        if isinstance(result, list):
            return [str(path) for path in result]
        matches = getattr(result, "matches", None)
        if isinstance(matches, list):
            return [str(path) for path in matches]
        paths = getattr(result, "paths", None)
        if isinstance(paths, list):
            return [str(path) for path in paths]
        return []

    @classmethod
    def _read_backend_file(cls, backend: StateBackend, path: str) -> str:
        """Read one virtual file path as text."""
        try:
            result = backend.read(path)
        except Exception:
            logger.debug("Unable to read virtual filesystem path %s", path, exc_info=True)
            return ""
        if getattr(result, "error", None):
            return ""
        file_data = getattr(result, "file_data", None)
        return cls._coerce_file_content(getattr(file_data, "content", file_data)).strip()

    @classmethod
    def _collect_live_virtual_files(cls) -> dict[str, Any]:
        """Collect compact live `/shared` files from the active DeepAgents state."""
        backend = StateBackend()
        paths: set[str] = {
            "/shared/plan.json",
            "/plan.json",
            "/shared/claim_table.json",
            "/shared/evidence_packet.json",
            "/shared/fact_ledger.json",
            "/shared/consolidated_findings.md",
            "/shared/research.md",
            "/shared/sources.json",
            "/shared/gaps.md",
            "/shared/contradictions.md",
            "/report.md",
        }
        for pattern in (
            "/shared/claims/*.json",
            "/shared/extracts/*.json",
            "/shared/section_briefs/*.md",
            "/shared/claude_code/*.md",
            "/shared/*.md",
            "/shared/*.txt",
            "/shared/*.json",
        ):
            paths.update(cls._backend_glob_paths(backend, pattern))

        files: dict[str, Any] = {}
        for path in sorted(paths):
            content = cls._read_backend_file(backend, path)
            if content:
                files[path] = content
        return files

    @staticmethod
    def _upsert_live_virtual_file(path: str, content: str) -> None:
        """Create or replace one DeepAgents virtual file in the active state."""
        backend = StateBackend()
        existing = backend.read(path)
        if getattr(existing, "error", None) or getattr(existing, "file_data", None) is None:
            result = backend.write(path, content)
            if getattr(result, "error", None):
                logger.warning("Unable to write virtual file %s: %s", path, result.error)
            return
        backend._send_files_update({path: backend._prepare_for_storage(create_file_data(content))})  # noqa: SLF001

    @classmethod
    def _record_claude_code_usage(
        cls,
        stage: str,
        *,
        used: bool,
        reason: str,
        artifact_path: str = "",
        artifact_chars: int = 0,
    ) -> None:
        """Write a small virtual telemetry artifact for Claude Code handoffs."""

        safe_stage = re.sub(r"[^a-zA-Z0-9_-]+", "_", stage or "specialist")
        payload = {
            "stage": safe_stage,
            "used": used,
            "reason": reason,
            "artifact_path": artifact_path,
            "artifact_chars": artifact_chars,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        cls._upsert_live_virtual_file(
            f"/shared/claude_code/usage_{safe_stage}.json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _append_claude_code_memos(research_md: str, files: dict[str, Any]) -> str:
        """Append specialist memos to the main research compile for synthesis."""

        memo_blocks: list[str] = []
        for path, value in sorted(files.items()):
            normalized = str(path).lstrip("/")
            if not normalized.startswith("shared/claude_code/") or not normalized.endswith(".md"):
                continue
            content = DeepResearcherAgent._coerce_file_content(value).strip()
            if not content:
                continue
            memo_blocks.append(f"### /{normalized}\n\n{DeepResearcherAgent._truncate_artifact(content, limit=12000)}")
        if not memo_blocks:
            return research_md
        return research_md.rstrip() + "\n\n## Claude Code Specialist Memos\n\n" + "\n\n".join(memo_blocks) + "\n"

    def _claude_code_context(self, *, files: dict[str, Any], stage: str) -> str:
        """Build a bounded context packet for Claude Code specialist calls."""
        preferred = [
            "/shared/plan.json",
            "/shared/research.md",
            "/shared/sources.json",
            "/shared/gaps.md",
            "/shared/contradictions.md",
            "/shared/claim_table.json",
            "/shared/evidence_packet.json",
            "/shared/consolidated_findings.md",
        ]
        blocks = [
            f"Original request:\n{self._active_request_text}",
            f"Research depth: {self._active_research_depth}",
            f"Specialist stage: {stage}",
            "Available AI-Q tools conceptually: planner-agent, researcher-agent, web/search tools, "
            "claim table, evidence packet, source scoring, gap fill, final synthesis, citation verification.",
        ]
        seen: set[str] = set()
        normalized_files = {str(path): value for path, value in files.items()}
        for path in preferred:
            content = self._coerce_file_content(
                normalized_files.get(path) or normalized_files.get(path.lstrip("/"))
            ).strip()
            if not content:
                continue
            seen.add(path)
            blocks.append(f"--- {path} ---\n{self._truncate_artifact(content, limit=28000)}")
        for path, value in sorted(normalized_files.items()):
            display_path = path if path.startswith("/") else f"/{path}"
            if display_path in seen:
                continue
            if not display_path.startswith("/shared/section_briefs/") and not display_path.startswith(
                "/shared/claude_code/"
            ):
                continue
            content = self._coerce_file_content(value).strip()
            if content:
                blocks.append(f"--- {display_path} ---\n{self._truncate_artifact(content, limit=12000)}")
        return "\n\n".join(blocks)

    @staticmethod
    def _is_claim_fragment_path(path: str) -> bool:
        normalized = str(path).lstrip("/").lower()
        name = Path(normalized).name
        return (normalized.startswith("shared/claims/") and name.startswith("claims_") and name.endswith(".json")) or (
            normalized.startswith("shared/") and name.startswith("claims_") and name.endswith(".json")
        )

    @staticmethod
    def _merge_claim_fragments_into_result(result: dict | Any, *, job_id: str | None = None) -> bool:
        """Merge per-researcher claim fragments into canonical /shared/claim_table.json."""
        files = dict(DeepResearcherAgent._extract_files(result))
        if not files:
            return False

        fragments = [
            DeepResearcherAgent._coerce_file_content(value).strip()
            for path, value in files.items()
            if DeepResearcherAgent._is_claim_fragment_path(str(path))
        ]
        fragments = [content for content in fragments if content]
        if not fragments:
            return False

        table, errors = merge_claim_tables_json(fragments)
        if table is None:
            logger.warning("Unable to merge claim fragments: %s", "; ".join(errors))
            return False
        if job_id:
            table.job_id = job_id
        if errors:
            table.model_extra["merge_errors"] = errors

        content = table.model_dump_json(indent=2)
        entry = DeepResearcherAgent._file_state_entry(content)
        files["/shared/claim_table.json"] = entry
        files["shared/claim_table.json"] = entry
        DeepResearcherAgent._set_files(result, files)
        logger.info(
            "Merged %d claim fragment(s) into /shared/claim_table.json (%d claims)",
            len(fragments),
            len(table.all_claims()),
        )
        return True

    @staticmethod
    def _is_fact_ledger_fragment_path(path: str) -> bool:
        normalized = str(path).lstrip("/").lower()
        name = Path(normalized).name
        return normalized.startswith("shared/") and name.startswith("fact_ledger_") and name.endswith(".json")

    @staticmethod
    def _merge_fact_ledger_fragments_into_result(result: dict | Any) -> bool:
        """Merge per-researcher fact-ledger fragments into /shared/fact_ledger.json."""
        files = dict(DeepResearcherAgent._extract_files(result))
        if not files:
            return False

        fragments = [
            DeepResearcherAgent._coerce_file_content(value).strip()
            for path, value in files.items()
            if DeepResearcherAgent._is_fact_ledger_fragment_path(str(path))
        ]
        fragments = [content for content in fragments if content]
        if not fragments:
            return False

        ledger, errors = merge_fact_ledgers_json(fragments)
        if ledger is None:
            logger.warning("Unable to merge fact-ledger fragments: %s", "; ".join(errors))
            return False
        if errors:
            ledger.model_extra["merge_errors"] = errors
        entry = DeepResearcherAgent._file_state_entry(ledger.model_dump_json(indent=2))
        files["/shared/fact_ledger.json"] = entry
        files["shared/fact_ledger.json"] = entry
        DeepResearcherAgent._set_files(result, files)
        logger.info(
            "Merged %d fact-ledger fragment(s) into /shared/fact_ledger.json (%d entries)",
            len(fragments),
            len(ledger.entries),
        )
        return True

    def _merge_structured_research_artifacts_into_result(self, result: dict | Any) -> None:
        """Build deterministic shared artifacts from researcher fragments."""
        self._merge_claim_fragments_into_result(result, job_id=self.job_id)
        self._merge_fact_ledger_fragments_into_result(result)
        self._ensure_section_briefs_into_result(result)
        self._build_evidence_packet_into_result(result)
        self._build_research_compile_into_result(result)

    @staticmethod
    def _is_section_brief_path(path: str) -> bool:
        normalized = str(path).lstrip("/").lower()
        name = Path(normalized).name
        return normalized.startswith("shared/section_briefs/") and name.endswith(".md")

    def _ensure_section_briefs_into_result(self, result: dict | Any) -> bool:
        """Create a minimal synthesis-ready section brief when researchers only wrote notes.

        The researcher prompt asks every task to write section briefs, but M3 can
        occasionally stop after narrative notes. The final writer should still
        get a clean intermediate layer instead of raw, scattered note files, so
        this deterministic backfill creates one compact bridge file.
        """
        files = dict(self._extract_files(result))
        if not files:
            return False

        existing_briefs = [
            self._coerce_file_content(value).strip()
            for path, value in files.items()
            if self._is_section_brief_path(str(path))
        ]
        if any(len(content) >= 200 for content in existing_briefs):
            return False

        ignored_fragments = (
            "plan.json",
            "report.md",
            "claim_table.json",
            "evidence_packet.json",
            "fact_ledger.json",
            "resume_sources",
            "resume_instructions",
            "skill.md",
        )
        note_blocks: list[tuple[str, str]] = []
        for path, value in files.items():
            normalized = str(path).lstrip("/")
            name_lower = normalized.lower()
            if any(part in name_lower for part in ignored_fragments):
                continue
            if name_lower.startswith("shared/claims/") or name_lower.startswith("shared/extracts/"):
                continue
            if not (name_lower.endswith(".txt") or name_lower.endswith(".md")):
                continue
            content = self._coerce_file_content(value).strip()
            if len(content) < 200 or self._looks_like_provider_payload(content):
                continue
            if self._is_unavailable_source_context(content):
                continue
            note_blocks.append((self._display_shared_path(normalized), content))

        if not note_blocks:
            return False

        sections = [
            "# Section Briefs Compiled From Research Notes",
            "",
            "This deterministic brief was created because no researcher-specific section brief was present. "
            "Use it as a synthesis bridge, then ground factual claims in `/shared/evidence_packet.json`, "
            "`/shared/claim_table.json`, and verified sources.",
        ]
        for path, content in note_blocks[:8]:
            title = Path(path).name.replace("_", " ").replace("-", " ")
            sections.extend(
                [
                    "",
                    f"## {title}",
                    "",
                    self._truncate_artifact(content, limit=8000),
                ]
            )

        content = "\n".join(sections).strip() + "\n"
        entry = self._file_state_entry(content)
        files["/shared/section_briefs/compiled_from_notes.md"] = entry
        files["shared/section_briefs/compiled_from_notes.md"] = entry
        self._set_files(result, files)
        logger.info("Backfilled /shared/section_briefs/compiled_from_notes.md from %d note file(s)", len(note_blocks))
        return True

    @staticmethod
    def _is_extract_fragment_path(path: str) -> bool:
        normalized = str(path).lstrip("/").lower()
        name = Path(normalized).name
        return (normalized.startswith("shared/extracts/") and name.endswith(".json")) or (
            normalized.startswith("shared/") and name.startswith("extracts_") and name.endswith(".json")
        )

    def _build_evidence_packet_into_result(self, result: dict | Any) -> bool:
        """Assemble `/shared/evidence_packet.json` for the M3 synthesis pass."""
        files = dict(self._extract_files(result))
        if not files:
            return False

        claim_table_content = ""
        extract_contents: list[str] = []
        for path, value in files.items():
            normalized = str(path).lstrip("/").lower()
            if normalized in {"shared/claim_table.json", "claim_table.json"}:
                claim_table_content = self._coerce_file_content(value).strip()
                continue
            if self._is_extract_fragment_path(str(path)):
                content = self._coerce_file_content(value).strip()
                if content:
                    extract_contents.append(content)

        if not (claim_table_content or extract_contents):
            return False
        registry_sources = self.source_registry_middleware._get_registry().all_sources()

        packet = build_evidence_packet(
            job_id=self.job_id,
            request_text=self._active_request_text,
            claim_table_content=claim_table_content or None,
            extract_contents=extract_contents,
            registry_sources=registry_sources,
        )
        content = packet.model_dump_json(indent=2)
        entry = self._file_state_entry(content)
        files["/shared/evidence_packet.json"] = entry
        files["shared/evidence_packet.json"] = entry
        self._set_files(result, files)
        logger.info(
            "Built /shared/evidence_packet.json with %d ranked source(s), %d claim(s)",
            packet.source_count,
            packet.claim_count,
        )
        return True

    def _build_research_compile_into_result(self, result: dict | Any, *, final_report: str | None = None) -> bool:
        """Assemble `/shared/research.md`, sources, gaps, and contradictions."""
        files = dict(self._extract_files(result))
        if not files:
            return False
        normalized_paths = {str(path).lstrip("/").lower() for path in files}
        has_structured_input = any(
            path in {"shared/plan.json", "shared/claim_table.json", "shared/evidence_packet.json"}
            or path.startswith("shared/claims/")
            or path.startswith("shared/extracts/")
            for path in normalized_paths
        )
        if not has_structured_input and not final_report:
            return False
        artifacts = build_virtual_research_artifacts(
            files=files,
            registry_sources=self.source_registry_middleware._get_registry().all_sources(),
            job_id=self.job_id,
            request_text=self._active_request_text,
            final_report=final_report,
        )
        if "/shared/research.md" in artifacts:
            artifacts["/shared/research.md"] = self._append_claude_code_memos(
                artifacts["/shared/research.md"],
                files,
            )
        if not artifacts:
            return False
        for path, content in artifacts.items():
            entry = self._file_state_entry(content)
            files[path] = entry
            files[path.lstrip("/")] = entry
        self._set_files(result, files)
        logger.info("Built compiled research artifacts: %s", ", ".join(sorted(artifacts)))
        return True

    def _mirror_run_artifacts(self, result: dict | Any, *, final_report: str) -> None:
        """Mirror compact run artifacts to disk for audit/debug."""
        files = dict(self._extract_files(result))
        try:
            manifest = mirror_run_artifacts(
                files=files,
                registry_sources=self.source_registry_middleware._get_registry().all_sources(),
                job_id=self.job_id,
                request_text=self._active_request_text,
                final_report=final_report,
            )
        except Exception:
            logger.warning("Unable to mirror durable research run artifacts", exc_info=True)
            return
        files["/shared/run_manifest.json"] = self._file_state_entry(manifest.model_dump_json(indent=2))
        files["shared/run_manifest.json"] = files["/shared/run_manifest.json"]
        self._set_files(result, files)
        logger.info("Mirrored durable research run artifacts to %s", manifest.run_dir)

    @staticmethod
    def _claim_table_quality_reason(result: dict | Any) -> str | None:
        """Return a failure reason when present claim-table artifacts are invalid."""
        files = DeepResearcherAgent._extract_files(result)
        claim_contents: list[tuple[str, str]] = []
        for path, value in files.items():
            normalized = str(path).lstrip("/").lower()
            name = Path(normalized).name
            if normalized in {"shared/claim_table.json", "claim_table.json"} or (
                normalized.startswith("shared/") and name.startswith("claims_") and name.endswith(".json")
            ):
                content = DeepResearcherAgent._coerce_file_content(value).strip()
                if content:
                    claim_contents.append((str(path), content))

        if not claim_contents:
            return None

        total = 0
        supported = 0
        invalid_paths = []
        for path, content in claim_contents:
            table, errors = validate_claim_table_json(content)
            if table is None:
                invalid_paths.append(f"{path}: {'; '.join(errors)[:240]}")
                continue
            claims = table.all_claims()
            total += len(claims)
            supported += sum(1 for entry in claims if entry.status in {"verified", "partially_verified"})

        if invalid_paths:
            return f"invalid_claim_table ({invalid_paths[0]})"
        if total == 0:
            return "empty_claim_table"
        if supported == 0:
            return "claim_table_has_no_supported_claims"
        if total >= 4 and supported / total < 0.25:
            return f"claim_table_mostly_unverified ({supported}/{total} supported)"
        return None

    @staticmethod
    def _evidence_packet_content(result: dict | Any) -> str:
        """Return `/shared/evidence_packet.json` content from DeepAgents file state."""
        files = DeepResearcherAgent._extract_files(result)
        for path, value in files.items():
            normalized = str(path).lstrip("/").lower()
            if normalized in {"shared/evidence_packet.json", "evidence_packet.json"}:
                return DeepResearcherAgent._coerce_file_content(value).strip()
        return ""

    @staticmethod
    def _report_fact_audit_reason(result: dict | Any, report_text: str) -> str | None:
        """Return a retry reason for severe fact/citation integrity failures."""
        audit = evaluate_report_fact_audit(report_text, DeepResearcherAgent._evidence_packet_content(result))
        if not audit.hard_failed:
            return None
        first = audit.hard_issues[0]
        return f"fact_audit_failed ({first.code}: {first.message})"

    @staticmethod
    def _fact_ledger_quality_reason(result: dict | Any) -> str | None:
        """Return a retry reason when current/live facts conflict in the ledger."""
        files = DeepResearcherAgent._extract_files(result)
        for path, value in files.items():
            normalized = str(path).lstrip("/").lower()
            if normalized not in {"shared/fact_ledger.json", "fact_ledger.json"}:
                continue
            content = DeepResearcherAgent._coerce_file_content(value).strip()
            if not content:
                continue
            ledger, errors = validate_fact_ledger_json(content)
            if ledger is None:
                return f"invalid_fact_ledger ({'; '.join(errors)[:240]})"
            conflicts = live_fact_conflicts(ledger)
            if conflicts:
                first = conflicts[0]
                return f"live_fact_conflict ({first.entity} {first.fact_type}: {first.message})"
        return None

    @staticmethod
    def _merge_files_for_report_compiler(*states_or_results: Any) -> dict[str, Any]:
        """Merge any available DeepAgents file state for final report compilation."""
        merged: dict[str, Any] = {}
        for item in states_or_results:
            merged.update(DeepResearcherAgent._extract_files(item))
        return merged

    @staticmethod
    def _display_shared_path(path: str) -> str:
        """Return the user-visible path for files stored in the /shared route backend."""
        normalized = str(path).strip() or "research_notes.md"
        if normalized.startswith("/shared/") or normalized == "/shared":
            return normalized
        if normalized.startswith("shared/"):
            return f"/{normalized}"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        if normalized in {"/report.md", "/request.md"}:
            return normalized
        return f"/shared{normalized}"

    @staticmethod
    def _truncate_artifact(content: str, limit: int = _REPORT_COMPILER_MAX_ARTIFACT_CHARS) -> str:
        content = content.strip()
        if len(content) <= limit:
            return content
        head = content[: int(limit * 0.7)].rstrip()
        tail = content[-int(limit * 0.3) :].lstrip()
        return f"{head}\n\n[... middle truncated from {len(content)} characters ...]\n\n{tail}"

    @staticmethod
    def _is_unavailable_source_context(content: str) -> bool:
        lowered = content.lower()
        return (
            lowered.strip().startswith("error:")
            or "is unavailable because" in lowered
            or "api_key is not set" in lowered
            or "api key is not set" in lowered
            or "all web search queries returned zero results" in lowered
        )

    @staticmethod
    def _title_from_request(request_text: str) -> str:
        first_line = next((line.strip() for line in request_text.splitlines() if line.strip()), "Research Report")
        first_line = re.sub(r"^#+\s*", "", first_line)
        first_line = re.sub(r"^you are\s+.*?deep research.*?\.\s*", "", first_line, flags=re.IGNORECASE)
        if len(first_line) > 90:
            first_line = first_line[:87].rstrip() + "..."
        return first_line or "Research Report"

    def _seed_source_registry_from_text(self, content: str, *, tool_name: str = "artifact") -> int:
        if not content or self._is_unavailable_source_context(content):
            return 0
        registry = self.source_registry_middleware._get_registry()
        seeded = 0
        for url in re.findall(r"https?://[^\s<>'\")\]}]+", content):
            url = url.rstrip(".,;:!?)'\"}>")
            if not url or url in {"https://exa.ai/", "https://exa.ai"}:
                continue
            registry.add(SourceEntry(url=url, source_type="artifact", tool_name=tool_name))
            seeded += 1
        return seeded

    def _collect_report_compiler_artifacts(self, *states_or_results: Any) -> list[tuple[str, str]]:
        """Return bounded, relevant artifact text for deterministic report compilation."""
        files = self._merge_files_for_report_compiler(*states_or_results)
        artifacts: list[tuple[int, str, str]] = []
        ignored_names = ("plan.json", "resume_sources", "resume_instructions", "skill.md")
        for path, value in files.items():
            normalized = str(path).lstrip("/")
            name_lower = normalized.lower()
            if any(part in name_lower for part in ignored_names):
                continue
            if not (name_lower.endswith(".txt") or name_lower.endswith(".md")):
                continue
            content = self._coerce_file_content(value).strip()
            if len(content) < 200 or self._looks_like_provider_payload(content):
                continue
            if self._is_unavailable_source_context(content):
                priority = 0
            elif "consolidated_findings" in name_lower:
                priority = 3
            elif name_lower.endswith("report.md"):
                priority = 2
            else:
                priority = 1
            artifacts.append((priority, self._display_shared_path(normalized), content))

        artifacts.sort(key=lambda item: (-item[0], item[1]))
        return [(path, content) for _priority, path, content in artifacts]

    def _source_inventory_text(self, *, limit: int = 80) -> str:
        sources = self.source_registry_middleware._get_registry().all_sources()
        seen: set[str] = set()
        lines: list[str] = []
        for source in sources:
            url = source.url
            if not url:
                continue
            normalized = url.rstrip("/")
            if normalized in seen or normalized in {"https://exa.ai", "https://exa.ai/"}:
                continue
            seen.add(normalized)
            title = source.title or url
            lines.append(f"[{len(lines) + 1}] {title}: {url}")
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    @staticmethod
    def _coerce_llm_report_text(message: Any) -> str:
        raw = getattr(message, "content", message)
        if isinstance(raw, list):
            parts = [
                part.get("text", "")
                for part in raw
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"} and part.get("text")
            ]
            return "\n".join(parts).strip()
        return raw.strip() if isinstance(raw, str) else str(raw).strip()

    def _deterministic_compiled_report(
        self,
        request_text: str,
        artifacts: list[tuple[str, str]],
        source_inventory: str,
    ) -> str:
        title = self._title_from_request(request_text)
        sections = []
        for path, content in artifacts[:8]:
            clean_name = Path(path).name.replace("_", " ").replace("-", " ")
            sections.append(f"## {clean_name}\n\n{self._truncate_artifact(content, 18000)}")

        source_section = source_inventory or (
            "No validated source URLs were captured. Treat this report as an evidence-limited synthesis."
        )
        limitations = (
            "This report was compiled from persisted research artifacts after the final report writer "
            "did not emit a usable report. Claims should be treated with extra caution where the source "
            "inventory is sparse."
        )
        return (
            f"# {title}\n\n"
            "## Synthesis Status\n\n"
            f"{limitations}\n\n" + "\n\n".join(sections) + "\n\n## Sources\n\n" + source_section + "\n"
        )

    async def _compile_report_from_artifacts(self, state: DeepResearchAgentState, result: dict | Any) -> str:
        """Compile a final report from persisted artifacts when the agent final turn is reasoning-only."""
        artifacts = self._collect_report_compiler_artifacts(state, result)
        if not artifacts:
            return self._build_report_from_research_notes(result)

        for _path, content in artifacts:
            self._seed_source_registry_from_text(content)
        source_inventory = self._source_inventory_text()
        request_text = self._query_without_context(self._latest_user_text(state))

        evidence_blocks: list[str] = []
        remaining = _REPORT_COMPILER_MAX_INPUT_CHARS
        for path, content in artifacts:
            block = f"--- ARTIFACT: {path} ---\n{self._truncate_artifact(content)}"
            if len(block) > remaining:
                block = block[:remaining].rstrip()
            evidence_blocks.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break

        prompt = (
            "Write a publication-ready markdown research report from the persisted evidence below.\n"
            "Return markdown prose only. Do not mention internal tools, agents, file paths, retries, or artifacts.\n"
            "Use only claims supported by the evidence. If evidence is thin, state the limitation plainly.\n"
            "Include a Sources section using only the supplied source inventory. Do not invent URLs.\n\n"
            f"Original request:\n{request_text}\n\n"
            f"Source inventory:\n{source_inventory or 'No validated source URLs captured.'}\n\n"
            f"Evidence:\n{chr(10).join(evidence_blocks)}"
        )

        try:
            llm = self._orchestrator_llm_for_state(state)
            response = await llm.ainvoke(
                [
                    SystemMessage(content="You are a careful report compiler. Output only final markdown."),
                    HumanMessage(content=prompt),
                ]
            )
            compiled = self._coerce_llm_report_text(response)
            if (
                len(compiled) >= _MIN_REPORT_LENGTH
                and not self._looks_like_provider_payload(compiled)
                and "## " in compiled
            ):
                logger.warning(
                    "Compiled final report from persisted artifacts after reasoning-only finalizer output (%d chars)",
                    len(compiled),
                )
                return compiled
        except Exception as exc:
            logger.warning("LLM report compiler failed; using deterministic artifact compiler: %s", exc)

        compiled = self._deterministic_compiled_report(request_text, artifacts, source_inventory)
        logger.warning("Using deterministic artifact compiler for final report (%d chars)", len(compiled))
        return compiled

    @staticmethod
    def _extract_report_content(messages: list) -> str:
        """Extract report content from the last message, falling back to write_file tool calls if text is too short."""
        if not messages:
            return ""
        last_msg = messages[-1]
        raw = last_msg.content or ""
        if isinstance(raw, list):
            content = " ".join(
                p.get("text", "")
                for p in raw
                if isinstance(p, dict) and p.get("type") in {"text", "output_text"} and p.get("text")
            )
        else:
            content = raw if isinstance(raw, str) else str(raw)
        if DeepResearcherAgent._looks_like_provider_payload(content):
            content = ""
        if len(content) >= _MIN_REPORT_LENGTH:
            return content
        # If the last message is an AIMessage with a write_file tool call,
        # the LLM may have written the report via tool instead of text output.
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            for tc in last_msg.tool_calls:
                if tc.get("name") == "write_file":
                    file_content = tc.get("args", {}).get("content", "")
                    if isinstance(file_content, str) and len(file_content) > len(content):
                        content = file_content
        return content

    @staticmethod
    def _extract_report_content_from_result(result: dict | Any) -> str:
        """Extract the most substantive report from messages or /report.md state."""
        if isinstance(result, dict):
            messages = result.get("messages", [])
        else:
            messages = getattr(result, "messages", [])

        message_content = DeepResearcherAgent._extract_report_content(messages)
        file_content = DeepResearcherAgent._extract_report_file_content(result)
        if is_model_failure_report(file_content) or DeepResearcherAgent._looks_like_provider_payload(file_content):
            file_content = ""
        if DeepResearcherAgent._looks_like_provider_payload(message_content):
            message_content = ""
        if len(file_content) > len(message_content):
            return file_content
        return message_content

    @staticmethod
    def _minimum_final_source_count(state: DeepResearchAgentState | dict[str, Any] | None) -> int:
        """Return the minimum distinct verified sources expected in a final report."""
        tier = None
        if state is not None:
            tier = state.get("research_depth") if isinstance(state, dict) else getattr(state, "research_depth", None)
        depth_config = get_research_depth_config(tier)
        return {
            "shallow": 4,
            "medium": 10,
            "deeper": 12,
            "deep": 20,
        }[depth_config.tier]

    @staticmethod
    def _count_valid_report_sources(content: str, registry: Any) -> int:
        """Count distinct reference-section sources that resolve to captured tool outputs."""
        from aiq_agent.common.citation_verification import _CITATION_LINE_RE
        from aiq_agent.common.citation_verification import _REFERENCE_SECTION_RE
        from aiq_agent.common.citation_verification import _URL_IN_LINE_RE
        from aiq_agent.common.citation_verification import _URL_TRIM_CHARS
        from aiq_agent.common.citation_verification import _is_knowledge_citation

        ref_match = _REFERENCE_SECTION_RE.search(content)
        if not ref_match:
            return 0
        ref_section = content[ref_match.start() :]
        valid_sources: set[str] = set()
        for line_match in _CITATION_LINE_RE.finditer(ref_section):
            ref_text = line_match.group(2).strip()
            url_match = _URL_IN_LINE_RE.search(ref_text)
            if url_match:
                url = url_match.group(0).rstrip(_URL_TRIM_CHARS)
                canonical = registry.resolve_url(url)
                if canonical:
                    valid_sources.add(canonical)
                continue
            is_kl, citation_key = _is_knowledge_citation(ref_text, registry)
            if is_kl and citation_key and registry.has_citation_key(citation_key):
                valid_sources.add(citation_key)
        return len(valid_sources)

    def _is_report_complete(
        self,
        result: dict | Any,
        state: DeepResearchAgentState | dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if the agent produced a complete report using tool calls or heuristics.
        """
        if isinstance(result, dict):
            messages = result.get("messages", [])
        else:
            messages = getattr(result, "messages", [])
        if not messages:
            return False, "no_messages"

        content = self._extract_report_content_from_result(result)

        if is_model_failure_report(content):
            return False, "model_call_failed"

        if len(content) < _MIN_REPORT_LENGTH:
            return False, f"too_short ({len(content)} chars)"

        if content.count("## ") < 2:
            return False, "missing_section_headers"

        has_sources = (
            bool(
                re.search(
                    r"^#{2,3}\s+(?:(?:\d+|[A-Z])[\).:-]?\s+)?(?:Sources|References)\b",
                    content,
                    re.MULTILINE | re.IGNORECASE,
                )
            )
            or "Reference List" in content
        )
        if not has_sources:
            return False, "missing_sources_section"

        # Quick citation quality check — only reject if ALL citations are invalid
        # (full verification with repair/renumbering happens in run() post-processing)
        registry = self.source_registry_middleware._get_registry()
        if registry.all_sources():
            from aiq_agent.common.citation_verification import _CITATION_LINE_RE
            from aiq_agent.common.citation_verification import _REFERENCE_SECTION_RE
            from aiq_agent.common.citation_verification import _URL_IN_LINE_RE
            from aiq_agent.common.citation_verification import _is_knowledge_citation

            ref_match = _REFERENCE_SECTION_RE.search(content)
            if ref_match:
                ref_section = content[ref_match.start() :]
                has_any_valid = False
                for line_match in _CITATION_LINE_RE.finditer(ref_section):
                    ref_text = line_match.group(2).strip()
                    # Check URL citations
                    url_match = _URL_IN_LINE_RE.search(ref_text)
                    if url_match:
                        url = url_match.group(0).rstrip(".,;)")
                        if registry.resolve_url(url):
                            has_any_valid = True
                            break
                        continue
                    # Check knowledge-layer citation keys (lenient — passes registry for fuzzy match)
                    is_kl, citation_key = _is_knowledge_citation(ref_text, registry)
                    if is_kl and citation_key:
                        has_any_valid = True
                        break
                if not has_any_valid:
                    return False, "no_valid_citations"

        giving_up_patterns = [
            "please confirm",
            "do you want me to",
            "should i proceed",
            "choose one",
            "option (1)",
            "option (2)",
            "allow me to",
            "i need your permission",
            "i can't produce",
            "i cannot produce",
            "what i need from you",
        ]
        content_lower = content.lower()
        for pattern in giving_up_patterns:
            if pattern in content_lower:
                return False, f"agent_gave_up (detected: '{pattern}')"

        registry = self.source_registry_middleware._get_registry()
        if registry.all_sources():
            available_source_count = len(registry.all_sources())
            min_source_count = self._minimum_final_source_count(state)
            valid_source_count = self._count_valid_report_sources(content, registry)
            if available_source_count >= min_source_count and valid_source_count < min_source_count:
                return (
                    False,
                    f"too_few_sources_used ({valid_source_count}/{min_source_count}; "
                    f"available {available_source_count})",
                )
            request_text = ""
            if state is not None:
                if isinstance(state, dict):
                    for message in reversed(state.get("messages", []) or []):
                        if isinstance(message, HumanMessage):
                            content = message.content
                            request_text = content if isinstance(content, str) else str(content)
                            break
                else:
                    request_text = self._latest_user_text(state)
                request_text = self._query_without_context(request_text)
            source_quality = evaluate_report_source_quality(content, registry, request_text=request_text)
            if not source_quality.passed:
                logger.warning("Deep research report has source-quality warning: %s", source_quality.reason)

        claim_quality_reason = self._claim_table_quality_reason(result)
        if claim_quality_reason:
            logger.warning("Deep research report has claim-table warning: %s", claim_quality_reason)
        fact_ledger_reason = self._fact_ledger_quality_reason(result)
        if fact_ledger_reason:
            return False, fact_ledger_reason
        fact_audit_reason = self._report_fact_audit_reason(result, content)
        if fact_audit_reason:
            return False, fact_audit_reason

        return True, "complete_via_heuristic"

    async def run(self, state: DeepResearchAgentState) -> DeepResearchAgentState:
        """
        Execute deep research with multi-phase workflow.
        """
        state = state.model_copy(update={"files": self._normalize_files_state(state.files)})
        self._seed_source_registry_from_files(state.files)
        state = self._inject_approved_plan_if_available(state)
        state = self.deepagents_runtime.prepare_state(state)
        agent = self._build_orchestrator_agent(state)
        tool_counts_token = set_session_tool_counts({})
        tool_limits_token = set_session_tool_limits(self._tool_limits_for_state(state))
        exhausted_tools_token = set_session_exhausted_tools(set())
        parallel_tool_limits_token = set_session_parallel_tool_limits(self._parallel_tool_limits_for_state(state))
        plan_validation_failures_token = set_session_plan_validation_failures(0)
        planner_model_turns_token = set_session_planner_model_turns(0)
        recent_artifact_writes_token = set_session_recent_artifact_writes({})
        report_edit_failures_token = set_session_report_edit_failures(0)
        task_search_counts_token = set_session_task_search_counts({})

        messages = state.messages
        scope_request = self._query_without_context(self._latest_user_text(state))
        self._active_request_text = scope_request
        self._active_research_depth = str(state.research_depth or "deeper")
        if messages:
            query_content = messages[-1].content
            query = query_content if isinstance(query_content, str) else str(query_content)
            logger.info("=" * 80)
            logger.info("Deep Research Subagent: Starting workflow")
            logger.info("Query: %s...", query[:100])
            logger.info("=" * 80)

        result = None
        last_error = None
        try:
            max_retries = 2
            workflow_started_at = perf_counter()
            for attempt in range(max_retries):
                try:
                    attempt_started_at = perf_counter()
                    invoke_config: dict[str, Any] = {
                        "configurable": {
                            "thread_id": self.job_id or f"deep-research-{uuid.uuid4()}",
                            "checkpoint_ns": "deep_research",
                        }
                    }
                    if self.callbacks:
                        invoke_config["callbacks"] = self.callbacks
                    result = await agent.ainvoke(
                        state,
                        config=invoke_config,
                    )
                    logger.info(
                        "Deep Research timing: primary agent attempt %d completed in %.1fs",
                        attempt + 1,
                        perf_counter() - attempt_started_at,
                    )
                    self._merge_structured_research_artifacts_into_result(result)
                    last_error = None
                except Exception as ex:
                    logger.error("Deep Research attempt %d failed: %s", attempt + 1, ex, exc_info=True)
                    last_error = ex
                    # Auth errors must propagate immediately — retrying won't fix them.
                    if _AuthError and isinstance(ex, _AuthError):
                        raise ex
                    # If we hit the recursion limit or asyncio error, we might want to stop
                    if "recursion" in str(ex).lower() or "reuse already awaited" in str(ex):
                        raise ex
                    continue

                is_complete, reason = self._is_report_complete(result, state)
                if is_complete:
                    logger.info(f"Report completed successfully. Reason: {reason}")
                    break

                logger.warning("Report incomplete (attempt %d/%d): %s", attempt + 1, max_retries, reason)

                has_captured_sources = bool(self.source_registry_middleware._get_registry().all_sources())
                feedback_msg = f"Your report is not yet complete. Reason: {reason}. "
                if "missing_sources_section" in reason:
                    if has_captured_sources:
                        feedback_msg += "You must include a '## Sources' section listing all URLs."
                    else:
                        feedback_msg += (
                            "You answered without capturing any search/tool sources. "
                            "This is invalid for deep research. Run the actual workflow now: "
                            "use write_todos, call planner-agent if /shared/plan.json does not exist, "
                            "delegate to researcher-agent, and have the researcher use the configured search tools. "
                            "Only cite URLs returned by tools."
                        )
                elif "too_short" in reason:
                    feedback_msg += (
                        "The report is too short or contained only provider reasoning blocks. "
                        "Expand your analysis and write the actual final report now, using write_file "
                        "with file_path='/report.md', then return the report prose."
                    )
                elif "missing_section_headers" in reason:
                    feedback_msg += "Use markdown headers (##) to structure the report."
                elif "no_valid_citations" in reason:
                    feedback_msg += (
                        "None of your cited sources match actual tool results. "
                        "Re-check your findings and cite only URLs returned by your search tools."
                    )
                    # Include the consolidated source list so the orchestrator
                    # has an authoritative reference for the retry
                    source_list = self.source_registry_middleware.get_source_list_text()
                    if source_list:
                        feedback_msg += "\n\n" + source_list
                elif "too_few_sources_used" in reason:
                    feedback_msg += (
                        "The report used too few distinct verified sources relative to the sources already collected. "
                        "Repair the report without restarting research: call get_verified_sources, use the existing "
                        "source inventory, and spread citations across more distinct URLs/domains in the body and "
                        "Sources section. Use only sources that came from tools, and keep claims aligned to what "
                        "those sources actually support."
                    )
                    source_list = self.source_registry_middleware.get_source_list_text()
                    if source_list:
                        feedback_msg += "\n\n" + source_list
                elif "source_quality_failed" in reason:
                    feedback_msg += (
                        "The report uses sources too narrowly or relies on weak derivative sources. "
                        "Repair the report without restarting research: call get_verified_sources, "
                        "spread body citations across at least four distinct domains when available, "
                        "avoid letting one domain support most claims, and verify numeric claims "
                        "with primary or authoritative sources. If a statistic is only found in a blog, "
                        "mark it as reported by that blog rather than established fact."
                    )
                    source_list = self.source_registry_middleware.get_source_list_text()
                    if source_list:
                        feedback_msg += "\n\n" + source_list
                elif "claim_table" in reason:
                    feedback_msg += (
                        "The claim-resolution artifact is missing usable support or contains invalid JSON. "
                        "Do not restart the whole workflow. Read /shared/plan.json and all researcher notes, "
                        "then repair or create /shared/claim_table.json with entries that resolve the report's "
                        "important claims as verified, partially_verified, or unverified. The final report must "
                        "state verified claims confidently, hedge partially verified claims, and omit or clearly "
                        "label unverified claims."
                    )
                elif "live_fact_conflict" in reason or "invalid_fact_ledger" in reason:
                    feedback_msg += (
                        "The fact ledger has stale or conflicting current/live facts. "
                        "Repair the existing report without restarting research. Re-read /shared/fact_ledger.json, "
                        "/shared/evidence_packet.json, and /shared/sources.json. For funding, valuation, pricing, "
                        "release status, model limits, and benchmarks, use the newest dated verified ledger entry for "
                        "current claims. Older values may appear only as historical context with their dates. If the "
                        "ledger itself is invalid, repair it from existing researcher notes and extracts before final "
                        "writing."
                    )
                elif "fact_audit_failed" in reason:
                    feedback_msg += (
                        "The report has a factual-integrity problem: duplicate reference numbers, an arXiv ID "
                        "mismatch, or a precise numeric claim that does not match the cited evidence extracts. "
                        "Repair the existing report without restarting research. Re-read /shared/evidence_packet.json, "
                        "/shared/claim_table.json, and /shared/sources.json. For each flagged claim, either revise the "
                        "number/ID to match the cited extract, cite the correct source, hedge it explicitly if only "
                        "partial support exists, or remove the precise value. Do not invent new citations."
                    )

                if has_captured_sources:
                    feedback_msg += (
                        " IMPORTANT: Do NOT restart the research from scratch."
                        " First check if /report.md already exists using read_file."
                        " If it does, use that content as your report — just fix the specific issue above"
                        " and return the corrected report in your final message."
                    )
                else:
                    feedback_msg += (
                        " IMPORTANT: You must restart from the planning/research step because no sources were captured."
                        " Do not return a final report until researcher-agent has produced source-backed notes."
                    )

                if isinstance(result, dict):
                    next_state = {**result}
                    messages = result.get("messages", [])
                else:
                    next_state = result.model_dump() if hasattr(result, "model_dump") else dict(result)
                    messages = getattr(result, "messages", next_state.get("messages", []))
                next_state["messages"] = list(messages) + [HumanMessage(content=feedback_msg)]

                try:
                    retry_started_at = perf_counter()
                    result = await agent.ainvoke(
                        next_state,
                        config={"callbacks": self.callbacks} if self.callbacks else None,
                    )
                    logger.info(
                        "Deep Research timing: feedback retry %d completed in %.1fs",
                        attempt + 1,
                        perf_counter() - retry_started_at,
                    )
                    self._merge_structured_research_artifacts_into_result(result)
                    last_error = None
                except Exception as ex:
                    logger.error("Deep Research feedback retry %d failed: %s", attempt + 1, ex, exc_info=True)
                    last_error = ex
                    if "recursion" in str(ex).lower() or "reuse already awaited" in str(ex):
                        raise ex
                    # Non-fatal: ainvoke raised before producing a result, so
                    # `result` still holds the previous iteration's value.
                    # The next loop iteration will rebuild next_state from it.
                    continue

                # Evaluate the feedback-retry result before the next iteration
                is_complete, reason = self._is_report_complete(result, state)
                if is_complete:
                    logger.info(f"Report completed after feedback retry. Reason: {reason}")
                    break

                # Update state so next iteration builds on progress, not the original state
                state = result

            if result is None and last_error is not None:
                raise last_error

            final_message = "Research failed to produce a report."
            if result:
                self._merge_structured_research_artifacts_into_result(result)
                final_message = self._extract_report_content_from_result(result)
                if len(final_message) < _MIN_REPORT_LENGTH:
                    compiled_report = await self._compile_report_from_artifacts(state, result)
                    if len(compiled_report) > len(final_message):
                        logger.warning(
                            "Final report was too short (%d chars); using compiled artifact report (%d chars)",
                            len(final_message),
                            len(compiled_report),
                        )
                        final_message = compiled_report

            if result and len(final_message) >= _MIN_REPORT_LENGTH:
                files = dict(self._extract_files(result))
                files["/report.md"] = self._file_state_entry(final_message)
                files["report.md"] = self._file_state_entry(final_message)
                if isinstance(result, dict):
                    result["files"] = files
                elif hasattr(result, "files"):
                    result.files = files

            if is_model_failure_report(final_message) or len(final_message) < _MIN_REPORT_LENGTH:
                failure_detail = final_message.strip() or "empty final report"
                if last_error is not None:
                    raise RuntimeError(
                        f"Deep research failed before producing a report: {failure_detail}"
                    ) from last_error
                raise RuntimeError(f"Deep research failed before producing a report: {failure_detail}")

            scope_ok, scope_reason = report_matches_request_scope(final_message, scope_request)
            if not scope_ok:
                raise RuntimeError(f"Deep research report drifted from the requested topic: {scope_reason}")

            # Post-process: verify citations against source registry
            postprocess_started_at = perf_counter()
            if self.source_registry_middleware._get_registry().all_sources():
                registry = self.source_registry_middleware._get_registry()
                verification = verify_citations(final_message, registry)
                if verification.removed_citations:
                    removed_details = []
                    for c in verification.removed_citations:
                        url_match = re.search(r"https?://\S+", c.get("line", ""))
                        url_str = url_match.group(0).rstrip(".,;)") if url_match else "(no url)"
                        removed_details.append(f"[{c['number']}] {c['reason']}: {url_str}")
                    logger.info(
                        "Citation verification removed %d invalid citation(s):\n  %s",
                        len(verification.removed_citations),
                        "\n  ".join(removed_details),
                    )
                final_message = verification.verified_report
                reference_rebuild = rebuild_references(final_message, registry)
                if reference_rebuild.rebuilt:
                    final_message = reference_rebuild.report
                    logger.info(
                        "Rebuilt References section deterministically: %d reference(s), removed inline citations=%s, "
                        "ambiguous=%s",
                        reference_rebuild.reference_count,
                        reference_rebuild.removed_inline_citations,
                        reference_rebuild.ambiguous_reference_numbers,
                    )
                if not verification.valid_citations:
                    logger.warning(
                        "Deep researcher produced no valid citations after verification; "
                        "returning sanitized report without fabricating references."
                    )
            else:
                from aiq_agent.common.tool_validation import validate_tool_availability

                _, available_count, unavailable = validate_tool_availability(
                    self.tools,
                    research_type="deep research",
                    enable_logging=False,
                )
                raise EmptySourceRegistryError(
                    "deep research",
                    unavailable_tools=unavailable,
                    available_count=available_count,
                )

            # Post-process: sanitize report (strip body URLs, shortened URLs, unsafe URLs)
            sanitization = sanitize_report(final_message)
            final_message = sanitization.sanitized_report
            final_message = sanitize_report_structure(final_message)
            if result:
                residual_audit = evaluate_report_fact_audit(
                    final_message,
                    self._evidence_packet_content(result),
                )
                note = fact_audit_note(residual_audit)
                if note and "## Source Accuracy Notes" not in final_message:
                    final_message = final_message.rstrip() + "\n\n" + note + "\n"
            if result:
                self._build_research_compile_into_result(result, final_report=final_message)
                self._mirror_run_artifacts(result, final_report=final_message)
            logger.info(
                "Deep Research timing: post-processing completed in %.1fs; total workflow %.1fs",
                perf_counter() - postprocess_started_at,
                perf_counter() - workflow_started_at,
            )

            # Re-emit the verified/sanitized report so the frontend overwrites
            # the raw version that on_llm_end auto-emitted during ainvoke().
            for cb in self.callbacks:
                if hasattr(cb, "emit_final_report"):
                    cb.emit_final_report(final_message)
                    break

            result_messages = self._result_messages(result)
            if result_messages:
                last_msg = result_messages[-1]
                if hasattr(last_msg, "model_copy"):
                    result_messages[-1] = last_msg.model_copy(update={"content": final_message})
                else:
                    result_messages[-1] = type(last_msg)(content=final_message)
                if isinstance(result, dict):
                    result["messages"] = result_messages
                elif hasattr(result, "messages"):
                    result.messages = result_messages

            logger.info("=" * 80)
            logger.info("Deep Research Subagent: Workflow complete")
            logger.info("Final report length: %d characters", len(final_message))
            logger.info("=" * 80)
            normalized_result = result
            if not isinstance(result, dict):
                if hasattr(result, "model_dump"):
                    normalized_result = result.model_dump()
                elif hasattr(result, "__dict__"):
                    normalized_result = dict(result.__dict__)
            return DeepResearchAgentState.model_validate(normalized_result)

        except Exception as ex:
            logger.error("Deep Research Subagent failed: %s", ex, exc_info=True)
            raise
        finally:
            reset_session_tool_counts(tool_counts_token)
            reset_session_tool_limits(tool_limits_token)
            reset_session_exhausted_tools(exhausted_tools_token)
            reset_session_parallel_tool_limits(parallel_tool_limits_token)
            reset_session_plan_validation_failures(plan_validation_failures_token)
            reset_session_planner_model_turns(planner_model_turns_token)
            reset_session_recent_artifact_writes(recent_artifact_writes_token)
            reset_session_report_edit_failures(report_edit_failures_token)
            reset_session_task_search_counts(task_search_counts_token)

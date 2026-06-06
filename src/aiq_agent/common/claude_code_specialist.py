# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional Claude Code handoff for high-leverage research orchestration tasks."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaudeCodeDecision:
    """Routing decision for a Claude Code specialist handoff."""

    should_use: bool
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ClaudeCodeResult:
    """Result returned by the Claude Code subprocess wrapper."""

    ok: bool
    output: str
    error: str = ""
    timed_out: bool = False
    command: tuple[str, ...] = ()
    artifact_path: str = ""


def should_use_claude_code(
    task_description: str,
    *,
    current_context_size: int = 0,
    num_sources: int = 0,
    stage: str = "",
    tier: str = "",
) -> ClaudeCodeDecision:
    """Return whether a subtask should be routed to Claude Code + M3."""

    text = f"{task_description} {stage}".lower()
    score = 0
    reasons: list[str] = []
    if any(term in text for term in ("synthesize", "compare", "comparison", "timeline", "matrix")):
        score += 1
        reasons.append("complex synthesis/comparison")
    if any(term in text for term in ("analyze", "extract", "contradiction", "orchestrate", "plan direction")):
        score += 1
        reasons.append("multi-step analysis")
    if current_context_size > 25_000 or num_sources > 8:
        score += 1
        reasons.append("large context/source set")
    if any(term in text for term in ("script", "code", "table", "chart", "dataset", "json", "markdown")):
        score += 2
        reasons.append("structured/code-generated artifact")
    if any(term in text for term in ("critique", "review", "missing", "bias", "weakness")):
        score += 1
        reasons.append("reflection/critique")
    if tier in {"medium", "deeper", "deep"} and stage == "scope_classifier":
        score += 2
        reasons.append(f"{tier} mandatory scope-classification boundary")
    elif tier in {"medium", "deeper", "deep"} and stage == "plan_director":
        score += 2
        reasons.append(f"{tier} mandatory plan-direction boundary")
    elif tier in {"medium", "deeper", "deep"} and stage == "synthesis_review":
        score += 1
        reasons.append(f"{tier} {stage} boundary")
    return ClaudeCodeDecision(should_use=score >= 2, score=score, reasons=tuple(reasons))


def claude_code_enabled(*, tier: str = "", stage: str = "") -> bool:
    """Return whether Claude Code handoff is enabled by environment."""

    global_setting = os.environ.get("AIQ_CLAUDE_CODE_ENABLED", "").strip().lower()
    if global_setting in {"0", "false", "no", "off"}:
        return False
    if global_setting in {"1", "true", "yes", "on"}:
        base_enabled = True
    else:
        base_enabled = tier in {"medium", "deeper", "deep"} and stage in {
            "scope_classifier",
            "plan_director",
            "synthesis_review",
        }
    if not base_enabled:
        return False
    tier_key = f"AIQ_CLAUDE_CODE_{tier.upper()}_ENABLED" if tier else ""
    if tier_key and os.environ.get(tier_key, "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    stage_key = f"AIQ_CLAUDE_CODE_{stage.upper()}_ENABLED" if stage else ""
    if stage_key and os.environ.get(stage_key, "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


async def run_claude_code_specialist(
    *,
    task: str,
    context: str,
    stage: str,
    cwd: Path | str,
    tier: str = "",
    timeout_seconds: int | None = None,
    output_path: Path | str | None = None,
) -> ClaudeCodeResult:
    """Run Claude Code in print mode and return the bounded final artifact.

    Permission bypass is intentionally opt-in through
    ``AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS=true``. The subprocess receives the
    exact task/context and returns text only; it does not replace the normal
    research search tools.
    """

    if not claude_code_enabled(tier=tier, stage=stage):
        return ClaudeCodeResult(ok=False, output="", error="Claude Code handoff disabled")

    executable = os.environ.get("AIQ_CLAUDE_CODE_PATH") or shutil.which("claude")
    if not executable:
        return ClaudeCodeResult(ok=False, output="", error="Claude Code CLI not found")

    timeout = timeout_seconds or int(os.environ.get("AIQ_CLAUDE_CODE_TIMEOUT", "600"))
    max_chars = int(os.environ.get("AIQ_CLAUDE_CODE_CONTEXT_CHARS", "180000"))
    trimmed_context = _trim_context(context, max_chars)
    artifact_path = _resolve_output_path(stage=stage, output_path=output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = _build_prompt(task=task, context=trimmed_context, stage=stage, output_path=artifact_path)

    command = [
        executable,
        "--bare",
        "--print",
        "--output-format",
        "text",
        "--permission-mode",
        "bypassPermissions"
        if _bypass_permissions_enabled()
        else os.environ.get("AIQ_CLAUDE_CODE_PERMISSION_MODE", "auto"),
        "--add-dir",
        str(Path(cwd).resolve()),
        "--add-dir",
        str(artifact_path.parent.resolve()),
    ]
    if _bypass_permissions_enabled():
        command.append("--dangerously-skip-permissions")
    command.append(prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=_claude_code_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        return ClaudeCodeResult(
            ok=False,
            output="",
            error=f"Claude Code timed out after {timeout}s",
            timed_out=True,
            command=tuple(command[:-1]),
            artifact_path=str(artifact_path),
        )
    except Exception as exc:
        return ClaudeCodeResult(
            ok=False,
            output="",
            error=str(exc),
            command=tuple(command[:-1]),
            artifact_path=str(artifact_path),
        )

    error = stderr.decode("utf-8", errors="replace").strip()
    artifact_output = ""
    if artifact_path.exists():
        artifact_output = artifact_path.read_text(encoding="utf-8", errors="replace").strip()
    if not artifact_output:
        error = "\n".join(
            part for part in (error, f"Claude Code did not write required artifact: {artifact_path}") if part
        )
    return ClaudeCodeResult(
        ok=proc.returncode == 0 and bool(artifact_output),
        output=artifact_output,
        error=error,
        command=tuple(command[:-1]),
        artifact_path=str(artifact_path),
    )


def _bypass_permissions_enabled() -> bool:
    return os.environ.get("AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS", "").strip().lower() in {"1", "true", "yes", "on"}


def _claude_code_env() -> dict[str, str]:
    env = dict(os.environ)
    provider = os.environ.get("AIQ_CLAUDE_CODE_PROVIDER", "minimax").strip().lower()
    if provider == "minimax":
        api_key = os.environ.get("AIQ_CLAUDE_CODE_API_KEY") or os.environ.get("MINIMAX_API_KEY")
        if api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
            # Claude Code --bare (2.1.x) authenticates strictly through
            # ANTHROPIC_API_KEY; keep AUTH_TOKEN too for non-bare compatibility.
            env["ANTHROPIC_API_KEY"] = api_key
        env["ANTHROPIC_BASE_URL"] = os.environ.get(
            "AIQ_CLAUDE_CODE_BASE_URL",
            "https://api.minimax.io/anthropic",
        )
        model = os.environ.get("AIQ_CLAUDE_CODE_MODEL", "MiniMax-M3")
        env.setdefault("ANTHROPIC_MODEL", model)
        env.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", model)
        env.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", model)
        env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", model)
        env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        return env

    strip_api_env = os.environ.get("AIQ_CLAUDE_CODE_STRIP_API_ENV", "false").strip().lower()
    if strip_api_env in {"1", "true", "yes", "on"}:
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"):
            env.pop(key, None)
    return env


def _trim_context(context: str, max_chars: int) -> str:
    context = context.strip()
    if len(context) <= max_chars:
        return context
    head = context[: int(max_chars * 0.65)].rstrip()
    tail = context[-int(max_chars * 0.35) :].lstrip()
    return f"{head}\n\n[... middle truncated from {len(context)} characters ...]\n\n{tail}"


def _resolve_output_path(*, stage: str, output_path: Path | str | None = None) -> Path:
    if output_path is not None:
        return Path(output_path)
    safe_stage = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (stage or "specialist"))
    return Path(tempfile.gettempdir()) / "aiq_claude_code" / f"{safe_stage}.md"


def _stage_contract(stage: str) -> str:
    if stage == "scope_classifier":
        return (
            "Write a Markdown memo with exactly these top-level headings:\n"
            "# Claude Scope Classification Memo\n"
            "## Primary Subject\n"
            "## Deliverable And Audience\n"
            "## Application Context\n"
            "## Forbidden Reframes\n"
            "## Research Modules\n"
            "## Budget And Execution Notes\n"
            "## Handoff Notes\n\n"
            "Requirements:\n"
            "- Do not perform web research, do not cite URLs, and do not invent facts.\n"
            "- Classify the user's request before planning. Separate the topic being researched from the "
            "format, audience, and application context.\n"
            "- Research Modules should propose 6 or fewer top-level modules unless the user explicitly requires more.\n"
            "- Budget And Execution Notes should explain which modules need the largest share of search "
            "budget and why.\n"
            "- Handoff Notes must say exactly how AI-Q should translate this memo into /shared/plan.json.\n"
        )
    if stage == "plan_director":
        return (
            "Write a Markdown memo with exactly these top-level headings:\n"
            "# Claude Plan Direction Memo\n"
            "## Research Task Split\n"
            "## Risky Claims To Verify\n"
            "## Source Strategy\n"
            "## Final Report Shape\n"
            "## Seed Queries\n"
            "## Handoff Notes\n\n"
            "Requirements:\n"
            "- Keep it concise and operational. This is planning advice, not the final report.\n"
            "- Do not perform web research, do not cite new URLs, and do not invent sources.\n"
            "- Preserve the user's topic, output format, audience, and approved plan scope.\n"
            "- Seed queries must be short, search-engine-friendly strings grouped by researcher task.\n"
            "- Handoff Notes must say exactly how the AI-Q orchestrator should use this memo next.\n"
        )
    if stage == "synthesis_review":
        return (
            "Write a Markdown review with exactly these top-level headings:\n"
            "# Claude Synthesis Review\n"
            "## Supported Core Narrative\n"
            "## Weak Or Missing Evidence\n"
            "## Contradictions And Caveats\n"
            "## Recommended Final Structure\n"
            "## Citation And Source Warnings\n"
            "## Handoff Notes\n\n"
            "Requirements:\n"
            "- Review only the supplied research artifacts. Do not perform web research.\n"
            "- Identify unsupported claims and source-quality risks before style concerns.\n"
            "- Recommended Final Structure must be directly usable by the final writer.\n"
            "- Handoff Notes must say exactly which files the AI-Q orchestrator should read next.\n"
        )
    return (
        "Write a Markdown specialist memo with clear headings, concrete recommendations, "
        "and a final ## Handoff Notes section. Do not perform web research unless the task explicitly permits it.\n"
    )


def _build_prompt(*, task: str, context: str, stage: str, output_path: Path | str) -> str:
    return (
        "You are Claude Code running as a bounded research specialist inside an AI-Q deep research pipeline.\n"
        "Your job is to produce exactly one artifact for the AI-Q orchestrator to read.\n"
        "Do not perform broad web search. Work only from the supplied context unless the task explicitly asks "
        "you to write/run small local analysis code. Do not mention internal permission mode or implementation "
        "details.\n\n"
        "OUTPUT CONTRACT:\n"
        f"- Required artifact path: {Path(output_path).resolve()}\n"
        "- You MUST write the artifact to that exact path.\n"
        "- Do not create alternate output files.\n"
        "- After writing the file, print exactly one line: ARTIFACT_WRITTEN\n"
        "- The AI-Q orchestrator will ignore ordinary stdout and will read only the required artifact file.\n\n"
        f"ARTIFACT FORMAT CONTRACT:\n{_stage_contract(stage)}\n"
        f"Stage: {stage}\n\n"
        f"Task:\n{task.strip()}\n\n"
        f"Context:\n{context.strip()}\n"
    )

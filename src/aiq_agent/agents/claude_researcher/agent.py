# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async agent that delegates a research run to Claude Code.

This is intentionally a bridge rather than a second framework. The AI-Q job
runner still owns auth, job status, cancellation, and report delivery; Claude
Code owns the concurrent research run folder and uses repo-local search/scrape
tools with a deterministic artifact contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.chat_researcher.utils import coerce_content_text
from aiq_agent.common import DEFAULT_RESEARCH_DEPTH
from aiq_agent.common import ResearchDepthTier

from .models import ClaudeResearchAgentState

RUN_ARTIFACTS = (
    "plan.md",
    "queries.json",
    "sources.json",
    "logs/progress.md",
    "contradictions.md",
    "gaps.md",
    "research.md",
    "final.md",
)

DEPTH_PROFILES = {
    "shallow": {
        "modules": "2-4",
        "searches": "4-8",
        "sources": "6-12 candidate sources",
        "reads": "4-8 source summaries",
        "notes": "concise answer with explicit gaps",
        "min_modules": 2,
        "min_sources": 4,
        "min_summaries": 2,
        "min_research_chars": 1000,
        "min_final_chars": 1200,
    },
    "medium": {
        "modules": "4-6",
        "searches": "8-14",
        "sources": "16-28 candidate sources",
        "reads": "10-16 source summaries",
        "notes": "balanced research compile with source table and gap check",
        "min_modules": 4,
        "min_sources": 8,
        "min_summaries": 4,
        "min_research_chars": 2000,
        "min_final_chars": 2500,
    },
    "deeper": {
        "modules": "5-7",
        "searches": "18-32",
        "sources": "35-70 candidate sources",
        "reads": "20-35 source summaries",
        "notes": "module-by-module evidence, contradictions, gap-fill pass, and calibrated final report",
        "min_modules": 5,
        "min_sources": 18,
        "min_summaries": 8,
        "min_research_chars": 4000,
        "min_final_chars": 5000,
    },
    "deep": {
        "modules": "7-10",
        "searches": "35-60",
        "sources": "70-140 candidate sources",
        "reads": "40-70 source summaries",
        "notes": "exhaustive evidence dossier, counterevidence, contradiction table, and validation pass",
        "min_modules": 7,
        "min_sources": 35,
        "min_summaries": 15,
        "min_research_chars": 7000,
        "min_final_chars": 9000,
    },
}

PLACEHOLDER_CONTENT = {
    "README.md": "# Claude Research Run",
    "plan.md": "# Research Plan",
    "queries.json": '"modules": []',
    "sources.json": '"sources": []',
    "contradictions.md": "# Contradictions",
    "gaps.md": "# Gaps",
    "research.md": "# Research Compile",
    "final.md": "# Final Report",
}


class ClaudeResearcherAgent:
    """Run Claude Code as a bounded research worker for async jobs."""

    def __init__(
        self,
        *,
        callbacks: list[Any] | None = None,
        config: Any | None = None,
        job_id: str | None = None,
        **_: Any,
    ) -> None:
        self.callbacks = callbacks or []
        self.config = config
        self.job_id = job_id
        self.repo_root = Path(__file__).resolve().parents[4]
        self.run_root = Path(
            os.environ.get(
                "AIQ_CLAUDE_RESEARCH_RUN_ROOT",
                str(self.repo_root / "runs" / "claude_research"),
            )
        ).expanduser()
        self.default_depth: ResearchDepthTier = getattr(config, "default_depth", DEFAULT_RESEARCH_DEPTH)
        self.timeout_seconds = int(getattr(config, "timeout_seconds", 3600))
        self.artifact_watch_interval = float(
            getattr(config, "artifact_watch_interval", os.environ.get("AIQ_CLAUDE_RESEARCH_WATCH_INTERVAL", "5"))
        )

    async def run(self, state: ClaudeResearchAgentState) -> ClaudeResearchAgentState:
        """Execute Claude Code and return a state whose last message is the report."""
        query = self._latest_user_text(state)
        if not query:
            return state.model_copy(
                update={"messages": state.messages + [AIMessage(content="No research query provided.")]}
            )

        depth = state.research_depth or self.default_depth
        run_id = self.job_id or hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        run_dir = self.run_root / self._safe_run_id(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._emit_status("claude_research.start", f"Starting Claude research run in {run_dir}")
        report = await self._run_claude(query=query, run_dir=run_dir, depth=depth)
        self._emit_run_artifacts(run_dir)
        self._emit_final_report(report)
        self._emit_status("claude_research.complete", f"Claude research run complete: {run_dir}")

        return state.model_copy(update={"messages": state.messages + [AIMessage(content=report)]})

    async def _run_claude(self, *, query: str, run_dir: Path, depth: str) -> str:
        tool_script = self.repo_root / "scripts" / "claude_research_tool.py"
        guide = self.repo_root / "docs" / "claude-research-operating-guide.md"
        claude_bin = os.environ.get("AIQ_CLAUDE_CODE_PATH", "claude")
        if not tool_script.exists():
            raise RuntimeError(f"Claude research tool is missing: {tool_script}")
        if not guide.exists():
            raise RuntimeError(f"Claude research guide is missing: {guide}")

        query_file = run_dir / "input.txt"
        query_file.write_text(query, encoding="utf-8")
        log_file = run_dir / "logs" / "claude-code.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        shell_path = self._find_posix_shell()
        if shell_path is None:
            message = (
                "Claude Code research requires a POSIX shell so it can run the repo-local "
                "search/scrape tool, but neither /bin/bash nor /bin/sh is available in "
                "the backend runtime."
            )
            self._emit_status("claude_research.environment_error", message)
            raise RuntimeError(message)
        env.setdefault("SHELL", shell_path)
        env.setdefault("AIQ_CLAUDE_CODE_PROVIDER", "minimax")
        env.setdefault("AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS", "true")
        env.setdefault("AIQ_CLAUDE_RESEARCH_DEPTH", depth)
        if env.get("AIQ_CLAUDE_CODE_PROVIDER", "minimax") == "minimax" and env.get("MINIMAX_API_KEY"):
            env["ANTHROPIC_BASE_URL"] = env.get("AIQ_CLAUDE_CODE_BASE_URL", "https://api.minimax.io/anthropic")
            api_key = env.get("AIQ_CLAUDE_CODE_API_KEY") or env["MINIMAX_API_KEY"]
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
            # Claude Code --bare (2.1.x) ignores ANTHROPIC_AUTH_TOKEN and
            # requires ANTHROPIC_API_KEY even for Anthropic-compatible gateways.
            env["ANTHROPIC_API_KEY"] = api_key
            model = env.get("AIQ_CLAUDE_CODE_MODEL", "MiniMax-M3")
            env.setdefault("ANTHROPIC_MODEL", model)
            env.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", model)
            env.setdefault("ANTHROPIC_DEFAULT_OPUS_MODEL", model)
            env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", model)
            env.setdefault("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
            env.setdefault("API_TIMEOUT_MS", str(self.timeout_seconds * 1000))
        else:
            env.setdefault(
                "ANTHROPIC_BASE_URL", env.get("AIQ_CLAUDE_CODE_BASE_URL", "https://api.minimax.io/anthropic")
            )

        init_process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(tool_script),
            "init-run",
            "--run-dir",
            str(run_dir),
            cwd=str(self.repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        init_stdout, _ = await init_process.communicate()
        if init_process.returncode != 0:
            raise RuntimeError(
                "Claude research run-folder initialization failed: "
                + init_stdout.decode("utf-8", errors="replace")[-2000:]
            )
        self._write_initial_progress(run_dir=run_dir, depth=str(depth))

        permission_args = ["--permission-mode", os.environ.get("AIQ_CLAUDE_CODE_PERMISSION_MODE", "auto")]
        if re.fullmatch(r"(?i)(1|true|yes|on)", env.get("AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS", "")):
            permission_args = ["--permission-mode", "bypassPermissions", "--dangerously-skip-permissions"]

        prompt = self._build_claude_prompt(
            query=query,
            run_dir=run_dir,
            depth=depth,
            tool_script=tool_script,
            guide=guide,
        )

        self._emit_status("claude_research.launch", "Launching Claude Code subprocess")
        process = await asyncio.create_subprocess_exec(
            claude_bin,
            "--bare",
            "--print",
            "--output-format",
            "text",
            "--input-format",
            "text",
            *permission_args,
            "--add-dir",
            str(self.repo_root),
            "--add-dir",
            str(run_dir),
            cwd=str(self.repo_root),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if process.stdin is not None:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

        output_chunks: list[str] = []
        stop_watcher = asyncio.Event()
        watcher_task = asyncio.create_task(self._watch_run_artifacts(run_dir, stop_watcher))
        try:
            assert process.stdout is not None
            async with asyncio.timeout(self.timeout_seconds):
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace")
                    output_chunks.append(text)
                    with log_file.open("a", encoding="utf-8") as fh:
                        fh.write(text)
                    stripped = text.strip()
                    if stripped:
                        self._emit_status("claude_research.stdout", self._truncate_status(stripped))
                return_code = await process.wait()
        except BaseException:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
            raise
        finally:
            stop_watcher.set()
            try:
                await asyncio.wait_for(watcher_task, timeout=3)
            except TimeoutError:
                watcher_task.cancel()

        if return_code != 0:
            tail = "".join(output_chunks)[-4000:]
            raise RuntimeError(f"Claude research failed with exit code {return_code}: {tail}")

        self._validate_depth_artifacts(run_dir=run_dir, depth=str(depth), output_tail="".join(output_chunks)[-4000:])

        final_path = run_dir / "final.md"
        return final_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _find_posix_shell() -> str | None:
        for candidate in ("/bin/bash", "/usr/bin/bash", "/bin/sh", "/usr/bin/sh"):
            if Path(candidate).exists():
                return candidate
        for candidate in ("bash", "sh"):
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def _build_claude_prompt(
        self,
        *,
        query: str,
        run_dir: Path,
        depth: str,
        tool_script: Path,
        guide: Path,
    ) -> str:
        depth_contract = self._depth_contract(str(depth))
        return f"""You are Claude Code running the concurrent Claude Research engine for this repository.

Read and obey this operating guide:
{guide}

Run folder:
{run_dir}

Research depth:
{depth}

Depth accountability contract:
{depth_contract}

User query:
{query}

You MUST use the repo-local tool for search and scraping:
{sys.executable} {tool_script}

Required final artifacts:
- {run_dir}/plan.md
- {run_dir}/queries.json
- {run_dir}/sources.json
- {run_dir}/logs/progress.md
- {run_dir}/contradictions.md
- {run_dir}/gaps.md
- {run_dir}/research.md
- {run_dir}/final.md

While working, update {run_dir}/logs/progress.md after each phase with short,
user-visible progress notes. The app streams this file into the UI. Do not put
hidden chain-of-thought there; write observable actions, sources considered,
modules completed, gaps found, and next actions.

Do not treat stdout as the deliverable. The app will inspect the files above.
Your first file write after reading this prompt must be {run_dir}/logs/progress.md
with one sentence saying that planning has started. Keep updating it during the
run.
When all required artifacts are complete, print exactly:
CLAUDE_RESEARCH_COMPLETE {run_dir}
"""

    def _emit_run_artifacts(self, run_dir: Path) -> None:
        for filename in RUN_ARTIFACTS:
            path = run_dir / filename
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            virtual_path = f"/claude_research/{run_dir.name}/{filename}"
            self._emit_file(virtual_path, content)

    async def _watch_run_artifacts(self, run_dir: Path, stop_event: asyncio.Event) -> None:
        """Stream Claude's run-folder work into the normal artifact channel."""
        seen: dict[Path, tuple[int, int]] = {}
        for path in self._iter_watch_files(run_dir):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            seen[path] = (stat.st_size, stat.st_mtime_ns)

        loop = asyncio.get_running_loop()
        last_status_at = loop.time()
        latest_artifact = "run folder initialized"
        while not stop_event.is_set():
            await asyncio.sleep(self.artifact_watch_interval)
            changed = False
            for path in self._iter_watch_files(run_dir):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                fingerprint = (stat.st_size, stat.st_mtime_ns)
                if seen.get(path) == fingerprint:
                    continue
                seen[path] = fingerprint
                if stat.st_size == 0:
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
                if not content.strip() or self._looks_like_placeholder(path, content):
                    continue
                relative = path.relative_to(run_dir)
                latest_artifact = relative.as_posix()
                changed = True
                virtual_path = f"/claude_research/{run_dir.name}/{relative.as_posix()}"
                self._emit_file(
                    virtual_path,
                    content,
                    workflow_source="claude_research",
                    agent_id="claude-code",
                    in_progress=True,
                )
                self._emit_status(
                    "claude_research.artifact",
                    f"Claude updated {relative.as_posix()} ({len(content):,} chars)",
                )
            now = loop.time()
            if changed:
                last_status_at = now
            elif now - last_status_at >= 30:
                self._emit_status(
                    "claude_research.progress",
                    f"Claude Code is still running; latest observed artifact: {latest_artifact}",
                )
                last_status_at = now

    def _write_initial_progress(self, *, run_dir: Path, depth: str) -> None:
        progress_path = run_dir / "logs" / "progress.md"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "# Claude Research Progress\n\n"
            f"- Backend initialized the run folder for `{depth}` research.\n"
            "- Claude Code subprocess is being launched; waiting for planning artifacts.\n"
        )
        progress_path.write_text(content, encoding="utf-8")
        self._emit_file(
            f"/claude_research/{run_dir.name}/logs/progress.md",
            content,
            workflow_source="claude_research",
            agent_id="claude-code",
            in_progress=True,
        )
        self._emit_status("claude_research.progress", "Initialized Claude research progress tracking")

    def _validate_depth_artifacts(self, *, run_dir: Path, depth: str, output_tail: str) -> None:
        """Reject placeholder or materially under-depth Claude runs."""
        profile = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["deeper"])
        failures: list[str] = []

        required_files = ("plan.md", "queries.json", "sources.json", "research.md", "final.md")
        for filename in required_files:
            path = run_dir / filename
            if not path.exists():
                failures.append(f"{filename} is missing")
                continue
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if not content or self._looks_like_placeholder(path, content):
                failures.append(f"{filename} still contains only placeholder content")

        gaps_path = run_dir / "gaps.md"
        gaps_content = gaps_path.read_text(encoding="utf-8", errors="replace").strip() if gaps_path.exists() else ""
        has_depth_exception = len(gaps_content) >= 200 and not self._looks_like_placeholder(
            run_dir / "gaps.md",
            gaps_content,
        )

        module_count = self._count_query_modules(run_dir / "queries.json")
        source_count = self._count_sources(run_dir / "sources.json")
        summaries_dir = run_dir / "source_summaries"
        summary_count = len(list(summaries_dir.glob("*.md"))) if summaries_dir.exists() else 0
        research_chars = self._non_placeholder_length(run_dir / "research.md")
        final_chars = self._non_placeholder_length(run_dir / "final.md")

        if module_count < int(profile["min_modules"]) and not has_depth_exception:
            failures.append(
                f"queries.json has {module_count} modules; expected at least {profile['min_modules']} for {depth}"
            )
        if source_count < int(profile["min_sources"]) and not has_depth_exception:
            failures.append(
                f"sources.json has {source_count} sources; expected at least {profile['min_sources']} for {depth}"
            )
        if summary_count < int(profile["min_summaries"]) and not has_depth_exception:
            failures.append(
                f"source_summaries has {summary_count} files; expected at least {profile['min_summaries']} for {depth}"
            )
        if research_chars < int(profile["min_research_chars"]) and not has_depth_exception:
            failures.append(
                f"research.md has {research_chars} substantive chars; expected at least {profile['min_research_chars']}"
            )
        if final_chars < int(profile["min_final_chars"]):
            failures.append(
                f"final.md has {final_chars} substantive chars; expected at least {profile['min_final_chars']}"
            )
        if self._declares_tool_execution_failure(run_dir):
            failures.append("run artifacts declare that live search/scrape tooling was unavailable")

        if failures:
            summary = "\n".join(f"- {failure}" for failure in failures)
            self._emit_status("claude_research.depth_failure", summary)
            raise RuntimeError(
                "Claude research did not satisfy the depth accountability contract:\n"
                f"{summary}\n\nClaude output tail:\n{output_tail}"
            )

    @staticmethod
    def _count_query_modules(path: Path) -> int:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        modules = data.get("modules") if isinstance(data, dict) else None
        if isinstance(modules, list):
            return len(modules)
        queries = data.get("queries") if isinstance(data, dict) else None
        return len(queries) if isinstance(queries, list) else 0

    @staticmethod
    def _count_sources(path: Path) -> int:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        sources = data.get("sources") if isinstance(data, dict) else None
        if not isinstance(sources, list):
            return 0
        return sum(1 for source in sources if ClaudeResearcherAgent._is_usable_source(source))

    @staticmethod
    def _is_usable_source(source: Any) -> bool:
        if not isinstance(source, dict):
            return False
        status = str(source.get("status") or "").strip().lower().replace("_", "-")
        if status in {"targeted-not-verified", "not-verified", "unverified"}:
            return False
        if str(source.get("url") or "").startswith(("http://", "https://")):
            return True
        return False

    def _non_placeholder_length(self, path: Path) -> int:
        if not path.exists():
            return 0
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if self._looks_like_placeholder(path, content):
            return 0
        return len(content)

    def _declares_tool_execution_failure(self, run_dir: Path) -> bool:
        failure_markers = (
            "no suitable shell found",
            "bash tool returned",
            "shell/bash tool",
            "could not run `scripts/claude_research_tool.py`",
            "could not run scripts/claude_research_tool.py",
            "without live web search or scraping",
            "live search and scraping were not possible",
            "all `sources.json` entries are therefore marked `targeted-not-verified`",
        )
        for relative in (
            "logs/progress.md",
            "logs/claude-code.log",
            "gaps.md",
            "research.md",
            "final.md",
        ):
            path = run_dir / relative
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace").lower()
            if any(marker in content for marker in failure_markers):
                return True
        return False

    def _iter_watch_files(self, run_dir: Path) -> list[Path]:
        paths = [run_dir / filename for filename in RUN_ARTIFACTS]
        paths.extend(
            [
                run_dir / "logs" / "progress.md",
                run_dir / "logs" / "search_failures.md",
                run_dir / "logs" / "scrape_failures.md",
                run_dir / "logs" / "claude-code.log",
            ]
        )
        for folder in ("notes", "source_summaries"):
            directory = run_dir / folder
            if directory.exists():
                paths.extend(sorted(path for path in directory.glob("*.md") if path.is_file()))
        return paths

    @staticmethod
    def _looks_like_placeholder(path: Path, content: str) -> bool:
        marker = PLACEHOLDER_CONTENT.get(path.name)
        stripped = content.strip()
        return bool(marker and stripped == marker) or stripped in {"{}", "[]"}

    @staticmethod
    def _depth_contract(depth: str) -> str:
        profile = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["deeper"])
        return (
            f"- Plan {profile['modules']} self-contained modules.\n"
            f"- Run roughly {profile['searches']} targeted searches unless the user explicitly requested a tiny task.\n"
            f"- Consider {profile['sources']} and scrape/read {profile['reads']}.\n"
            f"- Expected rigor: {profile['notes']}.\n"
            "- Every major module in queries.json must be represented in notes/ or research.md.\n"
            "- High-risk numbers, rankings, claims about recency, or named entities need authoritative support.\n"
            "- If you intentionally undershoot this depth because the query is narrow, sources are unavailable, "
            "or the user imposed a smaller budget, explain that exception in gaps.md."
        )

    def _emit_final_report(self, report: str) -> None:
        emitted = False
        for callback in self.callbacks:
            emit = getattr(callback, "emit_final_report", None)
            if callable(emit):
                emit(report)
                emitted = True
        if not emitted:
            self._emit_file("/report.md", report)

    def _emit_file(self, path: str, content: str, **extra_data: Any) -> None:
        try:
            from aiq_api.jobs.callbacks import ArtifactType
        except Exception:
            return
        for callback in self.callbacks:
            emit = getattr(callback, "_emit_artifact", None)
            if callable(emit):
                emit(
                    ArtifactType.FILE,
                    content,
                    name=path,
                    file_path=path,
                    path=path,
                    filename=Path(path).name,
                    **extra_data,
                )

    def _emit_status(self, name: str, message: str) -> None:
        try:
            from aiq_api.jobs.callbacks import EventCategory
            from aiq_api.jobs.callbacks import EventData
            from aiq_api.jobs.callbacks import EventState
            from aiq_api.jobs.callbacks import IntermediateStepEvent
        except Exception:
            return
        for callback in self.callbacks:
            emit = getattr(callback, "_emit", None)
            if callable(emit):
                emit(
                    IntermediateStepEvent(
                        category=EventCategory.WORKFLOW,
                        state=EventState.UPDATE,
                        name=name,
                        data=EventData(output=message),
                    )
                )

    @staticmethod
    def _truncate_status(value: str, limit: int = 1200) -> str:
        value = value.strip()
        if len(value) <= limit:
            return value
        return value[:limit].rstrip() + "\n[...truncated...]"

    @staticmethod
    def _latest_user_text(state: ClaudeResearchAgentState) -> str:
        for message in reversed(state.messages):
            if isinstance(message, HumanMessage):
                return coerce_content_text(message.content).strip()
        return ""

    @staticmethod
    def _safe_run_id(value: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
        return safe[:120] or f"claude-research-{uuid.uuid4().hex[:12]}"

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
import os
import re
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
    "contradictions.md",
    "gaps.md",
    "research.md",
    "final.md",
)


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
        env.setdefault("AIQ_CLAUDE_CODE_PROVIDER", "minimax")
        env.setdefault("AIQ_CLAUDE_CODE_BYPASS_PERMISSIONS", "true")
        env.setdefault("AIQ_CLAUDE_RESEARCH_DEPTH", depth)
        env.setdefault("ANTHROPIC_BASE_URL", env.get("AIQ_CLAUDE_CODE_BASE_URL", "https://api.minimax.io/anthropic"))
        if env.get("AIQ_CLAUDE_CODE_PROVIDER", "minimax") == "minimax" and env.get("MINIMAX_API_KEY"):
            env.setdefault("ANTHROPIC_API_KEY", env["MINIMAX_API_KEY"])

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

        process = await asyncio.create_subprocess_exec(
            claude_bin,
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
                return_code = await process.wait()
        except BaseException:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
            raise

        if return_code != 0:
            tail = "".join(output_chunks)[-4000:]
            raise RuntimeError(f"Claude research failed with exit code {return_code}: {tail}")

        final_path = run_dir / "final.md"
        if not final_path.exists() or not final_path.read_text(encoding="utf-8").strip():
            tail = "".join(output_chunks)[-4000:]
            raise RuntimeError(f"Claude research completed without final.md. Output tail: {tail}")

        return final_path.read_text(encoding="utf-8").strip()

    def _build_claude_prompt(
        self,
        *,
        query: str,
        run_dir: Path,
        depth: str,
        tool_script: Path,
        guide: Path,
    ) -> str:
        return f"""You are Claude Code running the concurrent Claude Research engine for this repository.

Read and obey this operating guide:
{guide}

Run folder:
{run_dir}

Research depth:
{depth}

User query:
{query}

You MUST use the repo-local tool for search and scraping:
{sys.executable} {tool_script}

Required final artifacts:
- {run_dir}/plan.md
- {run_dir}/queries.json
- {run_dir}/sources.json
- {run_dir}/contradictions.md
- {run_dir}/gaps.md
- {run_dir}/research.md
- {run_dir}/final.md

Do not treat stdout as the deliverable. The app will inspect the files above.
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

    def _emit_final_report(self, report: str) -> None:
        emitted = False
        for callback in self.callbacks:
            emit = getattr(callback, "emit_final_report", None)
            if callable(emit):
                emit(report)
                emitted = True
        if not emitted:
            self._emit_file("/report.md", report)

    def _emit_file(self, path: str, content: str) -> None:
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
    def _latest_user_text(state: ClaudeResearchAgentState) -> str:
        for message in reversed(state.messages):
            if isinstance(message, HumanMessage):
                return coerce_content_text(message.content).strip()
        return ""

    @staticmethod
    def _safe_run_id(value: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
        return safe[:120] or f"claude-research-{uuid.uuid4().hex[:12]}"

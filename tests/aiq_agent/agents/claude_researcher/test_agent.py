import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiq_agent.agents.claude_researcher.agent import ClaudeResearcherAgent


def test_claude_prompt_includes_depth_accountability(tmp_path):
    agent = ClaudeResearcherAgent(config=SimpleNamespace(artifact_watch_interval=0.01))

    prompt = agent._build_claude_prompt(
        query="Research a complex enterprise AI decision.",
        run_dir=tmp_path,
        depth="deeper",
        tool_script=tmp_path / "tool.py",
        guide=tmp_path / "guide.md",
    )

    assert "Depth accountability contract:" in prompt
    assert "Plan 5-7 self-contained modules." in prompt
    assert "Run roughly 18-32 targeted searches" in prompt
    assert "Consider 35-70 candidate sources" in prompt
    assert "scrape/read 20-35 source summaries" in prompt
    assert f"{tmp_path}/logs/progress.md" in prompt
    assert "The app streams this file into the UI" in prompt
    assert "Your first file write after reading this prompt" in prompt


def test_initial_progress_is_written_and_emitted(tmp_path):
    callback = MagicMock()
    agent = ClaudeResearcherAgent(callbacks=[callback], config=SimpleNamespace(artifact_watch_interval=0.01))
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    agent._write_initial_progress(run_dir=run_dir, depth="medium")

    progress = run_dir / "logs" / "progress.md"
    assert progress.exists()
    assert "Backend initialized the run folder" in progress.read_text(encoding="utf-8")
    assert callback._emit_artifact.call_args.kwargs["name"] == f"/claude_research/{run_dir.name}/logs/progress.md"
    assert callback._emit.call_args.args[0].name == "claude_research.progress"


def test_artifact_watcher_emits_changed_run_files(tmp_path):
    callback = MagicMock()
    agent = ClaudeResearcherAgent(callbacks=[callback], config=SimpleNamespace(artifact_watch_interval=0.01))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "plan.md").write_text("# Research Plan\n", encoding="utf-8")
    (run_dir / "logs").mkdir()

    async def _exercise():
        stop = asyncio.Event()
        task = asyncio.create_task(agent._watch_run_artifacts(run_dir, stop))
        await asyncio.sleep(0.03)
        (run_dir / "plan.md").write_text("# Research Plan\n\n## Main Question\nA real plan.\n", encoding="utf-8")
        (run_dir / "logs" / "progress.md").write_text("# Progress\n\n- Planned modules.\n", encoding="utf-8")
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(_exercise())

    artifact_names = [call.kwargs["name"] for call in callback._emit_artifact.call_args_list]
    workflow_names = [
        call.args[0].name
        for call in callback._emit.call_args_list
        if getattr(call.args[0], "event_type", "") == "workflow.update"
    ]

    assert f"/claude_research/{run_dir.name}/plan.md" in artifact_names
    assert f"/claude_research/{run_dir.name}/logs/progress.md" in artifact_names
    assert "claude_research.artifact" in workflow_names


def test_depth_validation_rejects_placeholder_report(tmp_path):
    callback = MagicMock()
    agent = ClaudeResearcherAgent(callbacks=[callback], config=SimpleNamespace(artifact_watch_interval=0.01))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "plan.md").write_text("# Research Plan\n\n## Real enough plan\n", encoding="utf-8")
    (run_dir / "queries.json").write_text('{"modules": [{"title": "A"}, {"title": "B"}]}', encoding="utf-8")
    (run_dir / "sources.json").write_text('{"sources": [{"url": "https://example.com"}]}', encoding="utf-8")
    (run_dir / "research.md").write_text("# Research Compile\n\nSome research notes.\n", encoding="utf-8")
    (run_dir / "final.md").write_text("# Final Report\n\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="depth accountability contract"):
        agent._validate_depth_artifacts(run_dir=run_dir, depth="shallow", output_tail="")

    workflow_names = [call.args[0].name for call in callback._emit.call_args_list]
    assert "claude_research.depth_failure" in workflow_names

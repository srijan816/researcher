from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "scripts" / "claude_research_tool.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("claude_research_tool", TOOL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_init_run_creates_required_artifacts(tmp_path: Path) -> None:
    tool = _load_tool()
    run_dir = tmp_path / "run"

    tool.init_run(run_dir)

    for filename in tool.RUN_FILES:
        assert (run_dir / filename).exists()
    assert (run_dir / "notes").is_dir()
    assert (run_dir / "source_summaries").is_dir()
    assert (run_dir / "logs").is_dir()


def test_fetcher_placeholder_is_rejected() -> None:
    tool = _load_tool()

    assert tool._looks_like_fetcher_placeholder("<200 https://example.com/>")
    assert not tool._looks_like_fetcher_placeholder("Example Domain\nThis domain is for use in examples.")


def test_append_sources_dedupes_by_url(tmp_path: Path) -> None:
    tool = _load_tool()
    run_dir = tmp_path / "run"

    tool._append_sources(
        run_dir,
        [
            {"url": "https://example.com", "title": "Old"},
            {"url": "https://example.com", "title": "New", "source_class": "unknown"},
        ],
    )

    data = tool._read_json(run_dir / "sources.json", {"sources": []})
    assert data["sources"] == [
        {"url": "https://example.com", "title": "New", "source_class": "unknown"},
    ]

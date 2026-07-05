# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in report image helpers (pure functions + fail-open stage)."""

from __future__ import annotations

import json

import pytest

from aiq_agent.common.report_images import DEFAULT_REPORT_IMAGES
from aiq_agent.common.report_images import MAX_REPORT_IMAGES
from aiq_agent.common.report_images import ReportImageSpec
from aiq_agent.common.report_images import build_image_planning_outline
from aiq_agent.common.report_images import build_planning_prompt
from aiq_agent.common.report_images import clamp_image_count
from aiq_agent.common.report_images import generate_and_save_report_images
from aiq_agent.common.report_images import insert_report_images
from aiq_agent.common.report_images import parse_image_specs
from aiq_agent.common.report_images import report_images_dir
from aiq_agent.common.report_images import report_images_enabled

_REPORT = """# Solar Sails

Intro paragraph about photon propulsion.

## How They Work

Radiation pressure pushes a large reflective membrane.

## Mission History

IKAROS launched in 2010.
"""


def _spec(heading: str = "How They Work") -> ReportImageSpec:
    return ReportImageSpec(after_heading=heading, prompt="a solar sail in orbit", caption="A solar sail")


# ---------------------------------------------------------------- parse_image_specs


def test_parse_image_specs_accepts_raw_json_array():
    text = json.dumps(
        [{"after_heading": "How They Work", "prompt": "a sail", "caption": "Sail"}]
    )
    specs = parse_image_specs(text)
    assert specs == [ReportImageSpec(after_heading="How They Work", prompt="a sail", caption="Sail")]


def test_parse_image_specs_accepts_code_fenced_json():
    text = '```json\n[{"after_heading": "H", "prompt": "p", "caption": "c"}]\n```'
    assert len(parse_image_specs(text)) == 1


def test_parse_image_specs_accepts_json_embedded_in_prose():
    text = 'Here you go:\n[{"after_heading": "H", "prompt": "p", "caption": "c"}]\nEnjoy!'
    assert len(parse_image_specs(text)) == 1


def test_parse_image_specs_skips_invalid_entries_and_caps_at_max():
    entries = [{"after_heading": f"H{i}", "prompt": "p", "caption": "c"} for i in range(5)]
    entries.insert(0, {"prompt": "missing heading"})
    entries.insert(1, "not-a-dict")
    entries.insert(2, {"after_heading": "no prompt"})
    specs = parse_image_specs(json.dumps(entries))
    assert len(specs) == MAX_REPORT_IMAGES
    assert specs[0].after_heading == "H0"


def test_parse_image_specs_respects_requested_max_images():
    entries = [{"after_heading": f"H{i}", "prompt": "p", "caption": "c"} for i in range(5)]
    assert len(parse_image_specs(json.dumps(entries), max_images=2)) == 2
    assert len(parse_image_specs(json.dumps(entries), max_images=4)) == 4


def test_parse_image_specs_returns_empty_for_garbage_or_empty_input():
    assert parse_image_specs("") == []
    assert parse_image_specs("no json here") == []
    assert parse_image_specs('{"an": "object, not array"}') == []


def test_parse_image_specs_defaults_caption_to_heading():
    specs = parse_image_specs('[{"after_heading": "H", "prompt": "p"}]')
    assert specs[0].caption == "H"


# ------------------------------------------------------------- insert_report_images


def test_insert_report_images_inserts_markdown_after_heading():
    updated, inserted = insert_report_images(_REPORT, [(_spec(), "/api/jobs/async/job/j1/images/image-1.png")])
    assert inserted == 1
    lines = updated.split("\n")
    idx = lines.index("## How They Work")
    assert lines[idx + 1] == ""
    assert lines[idx + 2] == "![A solar sail](/api/jobs/async/job/j1/images/image-1.png)"


def test_insert_report_images_matches_headings_case_insensitively():
    _, inserted = insert_report_images(_REPORT, [(_spec("## how they WORK"), "/img/x.png")])
    assert inserted == 1


def test_insert_report_images_skips_unknown_heading_and_empty_url():
    updated, inserted = insert_report_images(
        _REPORT,
        [(_spec("Nonexistent Section"), "/img/a.png"), (_spec(), "")],
    )
    assert inserted == 0
    assert updated == _REPORT


def test_insert_report_images_sanitizes_brackets_in_caption():
    spec = ReportImageSpec(after_heading="Mission History", prompt="p", caption="IKAROS [2010]")
    updated, inserted = insert_report_images(_REPORT, [(spec, "/img/a.png")])
    assert inserted == 1
    assert "![IKAROS (2010)](/img/a.png)" in updated


def test_insert_report_images_noop_for_empty_list():
    assert insert_report_images(_REPORT, []) == (_REPORT, 0)


# ---------------------------------------------------------- planning prompt / counts


def test_clamp_image_count_defaults_to_three_and_caps_at_four():
    assert clamp_image_count(None) == DEFAULT_REPORT_IMAGES == 3
    assert clamp_image_count(1) == 1
    assert clamp_image_count(4) == 4
    assert clamp_image_count(0) == 1
    assert clamp_image_count(99) == MAX_REPORT_IMAGES == 4


def test_build_planning_prompt_demands_photorealism():
    prompt = build_planning_prompt("## Outline")
    assert "hyper-realistic photograph" in prompt
    assert "NOT an illustration" in prompt
    assert "NOT a sketch" in prompt
    assert "NOT a cartoon" in prompt
    assert "NOT a diagram" in prompt
    assert "Do NOT propose charts" in prompt
    assert "## Outline" in prompt


def test_build_planning_prompt_is_count_aware():
    assert "select the 3 best" in build_planning_prompt("o")
    assert "select the 1 best" in build_planning_prompt("o", image_count=1)
    assert "select the 4 best" in build_planning_prompt("o", image_count=4)


def test_grok_style_suffix_leans_on_photographic_detail():
    from aiq_agent.common.report_images import _GROK_STYLE_SUFFIX

    assert "hyper-realistic" in _GROK_STYLE_SUFFIX
    assert "natural expressive lighting" in _GROK_STYLE_SUFFIX
    assert "photographic detail" in _GROK_STYLE_SUFFIX
    assert "no illustration or sketch style" in _GROK_STYLE_SUFFIX


# ------------------------------------------------------- outline / env / dir helpers


def test_build_image_planning_outline_contains_headings_and_snippets():
    outline = build_image_planning_outline(_REPORT)
    assert "# Solar Sails" in outline
    assert "## How They Work" in outline
    assert "Radiation pressure" in outline


def test_build_image_planning_outline_truncates_snippets():
    report = "## Long\n" + "x" * 1000
    outline = build_image_planning_outline(report, snippet_chars=50)
    assert len(outline) < 200


def test_build_image_planning_outline_without_headings_returns_prefix():
    assert build_image_planning_outline("plain text only") == "plain text only"


def test_report_images_enabled_env_switch(monkeypatch):
    monkeypatch.delenv("AIQ_REPORT_IMAGES_ENABLED", raising=False)
    assert report_images_enabled() is True
    monkeypatch.setenv("AIQ_REPORT_IMAGES_ENABLED", "false")
    assert report_images_enabled() is False
    monkeypatch.setenv("AIQ_REPORT_IMAGES_ENABLED", "1")
    assert report_images_enabled() is True


def test_report_images_dir_sanitizes_job_id(monkeypatch, tmp_path):
    monkeypatch.setenv("AIQ_REPORT_IMAGES_DIR", str(tmp_path))
    path = report_images_dir("../../etc/passwd")
    assert path.parent == tmp_path
    assert ".." not in path.name and "/" not in path.name


# ------------------------------------------------------------ fail-open stage entry


class _ExplodingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("planner down")


class _EmptyLLM:
    async def ainvoke(self, messages):
        class _Msg:
            content = "[]"

        return _Msg()


@pytest.mark.asyncio
async def test_generate_and_save_is_fail_open_on_llm_error(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    report, count = await generate_and_save_report_images(
        report=_REPORT, job_id="j1", llm=_ExplodingLLM(), image_url_prefix="/api/x"
    )
    assert (report, count) == (_REPORT, 0)


@pytest.mark.asyncio
async def test_generate_and_save_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    report, count = await generate_and_save_report_images(
        report=_REPORT, job_id="j1", llm=_EmptyLLM(), image_url_prefix="/api/x", api_key=None
    )
    assert (report, count) == (_REPORT, 0)


@pytest.mark.asyncio
async def test_generate_and_save_no_specs_leaves_report_unchanged(monkeypatch):
    report, count = await generate_and_save_report_images(
        report=_REPORT, job_id="j1", llm=_EmptyLLM(), image_url_prefix="/api/x", api_key="k"
    )
    assert (report, count) == (_REPORT, 0)

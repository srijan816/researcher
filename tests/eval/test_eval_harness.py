# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the eval harness (scripts/eval).

Covers the questions.yaml schema, the judge's pure prompt-building/parsing
functions, and scorecard aggregation. No network access.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "scripts" / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.yaml"

VALID_DEPTHS = {"shallow", "medium", "deeper", "deep"}
VALID_SOURCE_CLASSES = {
    "first_party",
    "primary_issuer",
    "academic",
    "authoritative_third_party",
    "trade_press",
    "vendor_marketing",
    "content_marketing",
    "forum",
    "unknown",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge = _load_module("eval_judge", EVAL_DIR / "judge.py")
run_eval = _load_module("eval_run_eval", EVAL_DIR / "run_eval.py")


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    with QUESTIONS_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data["questions"]


# ------------------------------------------------------------- questions.yaml


def test_questions_file_has_18_entries(questions):
    assert len(questions) == 18


def test_question_ids_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_question_required_fields(questions):
    for question in questions:
        for field in ("id", "domain", "prompt", "depth", "rubric", "notes"):
            assert field in question, f"{question.get('id')} missing {field}"
        assert isinstance(question["prompt"], str) and len(question["prompt"].strip()) > 20


def test_question_depths_valid(questions):
    for question in questions:
        assert question["depth"] in VALID_DEPTHS, question["id"]
    depths = {q["depth"] for q in questions}
    assert {"medium", "deeper"} <= depths, "expected a mix of medium and deeper questions"


def test_rubric_schema(questions):
    for question in questions:
        rubric = question["rubric"]
        facts = rubric["required_facts"]
        assert isinstance(facts, list) and 3 <= len(facts) <= 6, question["id"]
        assert all(isinstance(fact, str) and fact.strip() for fact in facts), question["id"]
        assert isinstance(rubric["recency_required"], bool), question["id"]
        classes = rubric["expected_source_classes"]
        assert isinstance(classes, list) and classes, question["id"]
        assert set(classes) <= VALID_SOURCE_CLASSES, f"{question['id']}: unknown source classes {classes}"
        min_domains = rubric["min_distinct_domains"]
        assert isinstance(min_domains, int) and min_domains >= 1, question["id"]


def test_domain_coverage(questions):
    domains = [q["domain"] for q in questions]
    assert domains.count("finance") == 4
    assert domains.count("decision_support") == 3
    assert domains.count("debate_prep") == 3
    assert domains.count("meeting_prep") == 2
    assert domains.count("tech_landscape") == 3
    assert domains.count("current_events") == 3


def test_current_events_require_recency(questions):
    for question in questions:
        if question["domain"] == "current_events":
            assert question["rubric"]["recency_required"] is True, question["id"]


# ---------------------------------------------------------- runner selection


def test_select_questions_default_returns_all(questions):
    assert run_eval.select_questions(questions, None) == questions


def test_select_questions_subset(questions):
    subset = run_eval.select_questions(questions, f"{questions[2]['id']}, {questions[0]['id']}")
    assert [q["id"] for q in subset] == [questions[2]["id"], questions[0]["id"]]


def test_select_questions_unknown_id_raises(questions):
    with pytest.raises(ValueError, match="no-such-question"):
        run_eval.select_questions(questions, "no-such-question")


def test_extract_metrics_collects_available_fields():
    job = {"metrics": {"duration_seconds": 120}}
    state = {
        "artifacts": {
            "quality_verification.json": {"content": {"metrics": {"warning_count": 2}}},
            "tool_calls": [{"name": "web_search_tool"}, {"name": "advanced_web_search_tool"}, {"name": "write_file"}],
            "sources": {"found": 12, "cited": 8},
        }
    }
    metrics = run_eval.extract_metrics(job, state)
    assert metrics["duration_seconds"] == 120
    assert metrics["warning_count"] == 2
    assert metrics["tool_calls"] == 3
    assert metrics["search_calls"] == 2
    assert metrics["sources_found"] == 12
    assert metrics["sources_cited"] == 8


def test_extract_metrics_empty_payloads():
    assert run_eval.extract_metrics({}, None) == {}


# ------------------------------------------------------------- judge prompts


def _question(recency: bool = True) -> dict:
    return {
        "id": "q-test",
        "prompt": "What is the current state of widget manufacturing?",
        "depth": "medium",
        "rubric": {
            "required_facts": ["States the widget output figure", "Names two widget makers", "Dates every claim"],
            "recency_required": recency,
            "expected_source_classes": ["trade_press", "first_party"],
            "min_distinct_domains": 4,
        },
    }


def test_build_judge_prompt_includes_prompt_and_facts():
    prompt = judge.build_judge_prompt(_question(), "# Report\nWidgets are up [1].", today="2026-06-10")
    assert "widget manufacturing" in prompt
    assert "States the widget output figure" in prompt
    assert "Names two widget makers" in prompt
    assert "trade_press" in prompt
    assert "2026-06-10" in prompt
    assert "Widgets are up [1]." in prompt


def test_build_judge_prompt_recency_toggle():
    with_recency = judge.build_judge_prompt(_question(recency=True), "report")
    without_recency = judge.build_judge_prompt(_question(recency=False), "report")
    assert "Recency IS required" in with_recency
    assert '"recency": null' in without_recency
    assert "Recency is NOT required" in without_recency


def test_build_judge_prompt_truncates_long_reports():
    report = "x" * (judge.MAX_REPORT_CHARS + 500)
    prompt = judge.build_judge_prompt(_question(), report)
    assert "(truncated)" in prompt
    assert len(prompt) < judge.MAX_REPORT_CHARS + 5_000


def test_parse_judge_response_plain_json():
    parsed = judge.parse_judge_response('{"citation_support": {"score": 7, "rationale": "ok"}}')
    assert parsed["citation_support"]["score"] == 7


def test_parse_judge_response_fenced_and_prefixed():
    fenced = '```json\n{"factual_accuracy": {"score": 9, "rationale": "good"}}\n```'
    assert judge.parse_judge_response(fenced)["factual_accuracy"]["score"] == 9
    prefixed = 'Here are the scores:\n{"comprehensiveness": 6} trailing words'
    assert judge.parse_judge_response(prefixed)["comprehensiveness"] == 6


def test_parse_judge_response_rejects_garbage():
    with pytest.raises(ValueError):
        judge.parse_judge_response("no json here at all")


def test_extract_scores_normalizes_and_clamps():
    parsed = {
        "citation_support": {"score": 7, "rationale": "fine"},
        "factual_accuracy": 14,
        "comprehensiveness": {"score": -3},
        "recency": {"score": 8},
        "report_cleanliness": "not-a-number",
    }
    scores, rationales = judge.extract_scores(parsed, recency_required=True)
    assert scores["citation_support"] == 7
    assert scores["factual_accuracy"] == 10
    assert scores["comprehensiveness"] == 0
    assert scores["recency"] == 8
    assert scores["report_cleanliness"] is None
    assert rationales["citation_support"] == "fine"


def test_extract_scores_recency_null_when_not_required():
    parsed = {dim: {"score": 5} for dim in judge.DIMENSIONS}
    scores, _ = judge.extract_scores(parsed, recency_required=False)
    assert scores["recency"] is None
    assert scores["citation_support"] == 5


# --------------------------------------------------------------- aggregation


def _result(qid: str, base: int, recency: int | None = None, **extra) -> dict:
    scores = {
        "citation_support": base,
        "factual_accuracy": base + 1,
        "comprehensiveness": base + 2,
        "recency": recency,
        "report_cleanliness": base,
    }
    return {"id": qid, "depth": "medium", "scores": scores, "rationales": {}, "error": None, **extra}


def test_aggregate_scorecard_means_and_overall():
    results = [
        _result("q1", 6, recency=8, metrics={"search_calls": 9, "runner_wall_seconds": 300.0}),
        _result("q2", 8, recency=None),
    ]
    card = judge.aggregate_scorecard(results)
    assert card["judged"] == 2
    assert card["failed"] == 0
    assert card["dimension_means"]["citation_support"] == 7.0
    assert card["dimension_means"]["recency"] == 8.0
    q1 = next(q for q in card["questions"] if q["id"] == "q1")
    q2 = next(q for q in card["questions"] if q["id"] == "q2")
    # q1: (6+7+8+8+6)/5 ; q2 has no recency: (8+9+10+8)/4
    assert q1["overall"] == 7.0
    assert q2["overall"] == 8.75
    assert card["overall_mean"] == round((7.0 + 8.75) / 2, 2)
    assert q1["metrics"]["search_calls"] == 9


def test_aggregate_scorecard_counts_failures():
    failed = {"id": "q3", "scores": dict.fromkeys(judge.DIMENSIONS), "error": "no report.md"}
    card = judge.aggregate_scorecard([_result("q1", 5, recency=5), failed])
    assert card["judged"] == 1
    assert card["failed"] == 1
    # q1: (5+6+7+5+5)/5 = 5.6 and the failed question is excluded from the mean
    assert card["overall_mean"] == 5.6
    q3 = next(q for q in card["questions"] if q["id"] == "q3")
    assert q3["overall"] is None
    assert q3["error"] == "no report.md"


def test_render_scorecard_md_contains_rows_and_means():
    card = judge.aggregate_scorecard(
        [_result("q1", 6, recency=7, metrics={"search_calls": 4, "runner_wall_seconds": 120})]
    )
    markdown = judge.render_scorecard_md(card, run_name="smoke")
    assert "# Eval Scorecard: smoke" in markdown
    assert "| q1 |" in markdown
    assert "## Dimension means" in markdown
    assert "| 4 | 120 |" in markdown


def test_render_compare_md_deltas():
    card_a = judge.aggregate_scorecard([_result("q1", 5, recency=5)])
    card_b = judge.aggregate_scorecard([_result("q1", 7, recency=7)])
    markdown = judge.render_compare_md(card_a, card_b, "baseline", "candidate")
    assert "baseline" in markdown and "candidate" in markdown
    # q1 overall: base 5 -> (5+6+7+5+5)/5 = 5.6; base 7 -> 7.6
    assert "| q1 | 5.6 | 7.6 | +2.0 |" in markdown


def test_scorecard_json_roundtrip(tmp_path: Path):
    card = judge.aggregate_scorecard([_result("q1", 6, recency=6)])
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == card

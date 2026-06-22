# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM judge for deep-research eval runs.

Scores each ``report.md`` in a run directory (produced by ``run_eval.py``)
against the question rubric using a single structured call to an
Anthropic-compatible ``/v1/messages`` endpoint, then aggregates results into
``scorecard.md`` and ``scorecard.json`` in the run directory.

Dimensions (0-10 each):
    citation_support    sampled cited claims are plausibly supported; orphan
                        [n] markers and uncited references are penalized
    factual_accuracy    rubric required_facts are present and correct
    comprehensiveness   coverage of the prompt
    recency             only scored when the rubric requires recency
    report_cleanliness  no thought traces, broken refs, duplicated sections

Environment variables:
    AIQ_EVAL_JUDGE_BASE_URL   Anthropic-compatible base URL
                              (default: $ANTHROPIC_BASE_URL or https://api.minimax.io/anthropic)
    AIQ_EVAL_JUDGE_MODEL      judge model name (default: MiniMax-M3)
    MINIMAX_API_KEY / ANTHROPIC_API_KEY   auth token (first non-empty wins)

Usage:
    uv run python scripts/eval/judge.py <run-name-or-dir>
    uv run python scripts/eval/judge.py --compare runs/a/scorecard.json runs/b/scorecard.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger("eval.judge")

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = EVAL_DIR / "questions.yaml"
DEFAULT_RUNS_DIR = EVAL_DIR / "runs"

DEFAULT_JUDGE_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_JUDGE_MODEL = "MiniMax-M3"
ANTHROPIC_VERSION = "2023-06-01"

DIMENSIONS = ("citation_support", "factual_accuracy", "comprehensiveness", "recency", "report_cleanliness")
MAX_REPORT_CHARS = 60_000
JUDGE_MAX_TOKENS = 2000
JUDGE_RETRIES = 3
RETRY_BACKOFF_SECONDS = 4.0
HTTP_TIMEOUT_SECONDS = 180.0


# --------------------------------------------------------------------------- pure functions


def resolve_judge_base_url(env: dict[str, str] | None = None) -> str:
    env = env if env is not None else dict(os.environ)
    return (
        env.get("AIQ_EVAL_JUDGE_BASE_URL") or env.get("ANTHROPIC_BASE_URL") or DEFAULT_JUDGE_BASE_URL
    ).rstrip("/")


def resolve_judge_api_key(env: dict[str, str] | None = None) -> str | None:
    env = env if env is not None else dict(os.environ)
    return env.get("MINIMAX_API_KEY") or env.get("ANTHROPIC_API_KEY") or None


def resolve_judge_model(env: dict[str, str] | None = None) -> str:
    env = env if env is not None else dict(os.environ)
    return env.get("AIQ_EVAL_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def build_judge_prompt(question: dict[str, Any], report: str, today: str | None = None) -> str:
    """Build the single structured judging prompt for one question/report pair."""
    rubric = question.get("rubric", {})
    recency_required = bool(rubric.get("recency_required", False))
    required_facts = rubric.get("required_facts", [])
    source_classes = rubric.get("expected_source_classes", [])
    min_domains = rubric.get("min_distinct_domains", 0)
    truncated = len(report) > MAX_REPORT_CHARS
    report_text = report[:MAX_REPORT_CHARS]

    facts_block = "\n".join(f"  {i + 1}. {fact}" for i, fact in enumerate(required_facts)) or "  (none specified)"
    recency_instruction = (
        "Recency IS required: penalize stale figures, undated claims, and superseded facts heavily. "
        'Score the "recency" dimension 0-10.'
        if recency_required
        else 'Recency is NOT required for this question: set "recency" to null.'
    )
    today_line = f"Today's date is {today}.\n" if today else ""
    recency_json = '{"score": <int 0-10>, "rationale": "..."}' if recency_required else "null"

    return f"""You are a strict research-report judge. {today_line}Evaluate the report below against the
original research prompt and rubric. Score each dimension as an integer from 0 (unusable) to 10 (excellent).

ORIGINAL RESEARCH PROMPT:
{question.get("prompt", "").strip()}

RUBRIC REQUIRED FACTS (the report should contain each, correctly and with support):
{facts_block}

RUBRIC EXPECTATIONS:
- Expected source classes: {", ".join(map(str, source_classes)) or "unspecified"}
- Minimum distinct cited web domains: {min_domains}
- {recency_instruction}

SCORING DIMENSIONS:
1. citation_support: Sample up to 10 cited claims (claims carrying [n] markers or inline source attributions).
   For each, judge whether the reference list and surrounding text plausibly support the claim. Penalize orphan
   [n] markers that match no reference, references never cited in the body, and claims whose cited source
   clearly cannot support them.
2. factual_accuracy: Check the report against each rubric required fact. Missing or wrong facts lower the
   score; fabricated precision (numbers/dates with no source) lowers it further.
3. comprehensiveness: How completely does the report answer the original prompt, including all sub-questions?
4. recency: {recency_instruction}
5. report_cleanliness: Penalize agent thought traces (e.g. "I will now", "Let me search"), broken or
   duplicated references, duplicated sections, leftover placeholders, and malformed markdown.

REPORT{" (truncated)" if truncated else ""}:
<report>
{report_text}
</report>

Respond with ONLY a JSON object, no prose, no markdown fences, exactly this shape:
{{"citation_support": {{"score": <int 0-10>, "rationale": "<one or two sentences>"}},
 "factual_accuracy": {{"score": <int 0-10>, "rationale": "..."}},
 "comprehensiveness": {{"score": <int 0-10>, "rationale": "..."}},
 "recency": {recency_json},
 "report_cleanliness": {{"score": <int 0-10>, "rationale": "..."}}}}"""


def parse_judge_response(text: str) -> dict[str, Any]:
    """Extract a JSON object from a judge response, tolerating fences and prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in judge response: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def _clamp_score(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(10, round(number)))


def extract_scores(parsed: dict[str, Any], recency_required: bool) -> tuple[dict[str, int | None], dict[str, str]]:
    """Normalize a parsed judge payload into per-dimension scores and rationales."""
    scores: dict[str, int | None] = {}
    rationales: dict[str, str] = {}
    for dim in DIMENSIONS:
        raw = parsed.get(dim)
        if dim == "recency" and not recency_required:
            scores[dim] = None
            continue
        if isinstance(raw, dict):
            scores[dim] = _clamp_score(raw.get("score"))
            rationale = raw.get("rationale")
            if isinstance(rationale, str):
                rationales[dim] = rationale.strip()
        else:
            scores[dim] = _clamp_score(raw)
    return scores, rationales


def question_overall(scores: dict[str, int | None]) -> float | None:
    """Mean of available dimension scores for one question."""
    values = [v for v in scores.values() if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def aggregate_scorecard(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question judge results into a scorecard payload.

    Each result dict needs: ``id``, ``scores`` (dim -> int|None) and may carry
    ``rationales``, ``metrics``, ``depth``, ``error``.
    """
    questions = []
    dimension_values: dict[str, list[int]] = {dim: [] for dim in DIMENSIONS}
    overall_values: list[float] = []
    failed = 0
    for result in results:
        scores = result.get("scores") or dict.fromkeys(DIMENSIONS)
        overall = question_overall(scores)
        if result.get("error") or overall is None:
            failed += 1
        else:
            overall_values.append(overall)
            for dim, value in scores.items():
                if value is not None and dim in dimension_values:
                    dimension_values[dim].append(value)
        questions.append(
            {
                "id": result.get("id"),
                "depth": result.get("depth"),
                "scores": scores,
                "overall": overall,
                "rationales": result.get("rationales", {}),
                "metrics": result.get("metrics"),
                "error": result.get("error"),
            }
        )
    dimension_means = {
        dim: (round(sum(values) / len(values), 2) if values else None) for dim, values in dimension_values.items()
    }
    return {
        "questions": questions,
        "dimension_means": dimension_means,
        "overall_mean": round(sum(overall_values) / len(overall_values), 2) if overall_values else None,
        "judged": len(results) - failed,
        "failed": failed,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _metric(metrics: dict[str, Any] | None, *keys: str) -> Any:
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if metrics.get(key) is not None:
            return metrics[key]
    return None


def render_scorecard_md(scorecard: dict[str, Any], run_name: str = "") -> str:
    """Render the aggregated scorecard as a markdown document."""
    lines = [
        f"# Eval Scorecard{f': {run_name}' if run_name else ''}",
        "",
        "Scores are 0-10 per dimension; `recency` is only scored where the rubric requires it.",
        "",
        "| Question | Depth | Citations | Factual | Comprehensive | Recency "
        "| Cleanliness | Overall | Search calls | Duration (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in scorecard.get("questions", []):
        scores = entry.get("scores", {})
        metrics = entry.get("metrics")
        row = [
            str(entry.get("id")),
            _fmt(entry.get("depth")),
            _fmt(scores.get("citation_support")),
            _fmt(scores.get("factual_accuracy")),
            _fmt(scores.get("comprehensiveness")),
            _fmt(scores.get("recency")),
            _fmt(scores.get("report_cleanliness")),
            _fmt(entry.get("overall")),
            _fmt(_metric(metrics, "search_calls", "tool_calls")),
            _fmt(_metric(metrics, "duration_seconds", "runner_wall_seconds")),
        ]
        if entry.get("error"):
            row[2:8] = ["err"] * 6
        lines.append("| " + " | ".join(row) + " |")

    means = scorecard.get("dimension_means", {})
    lines += [
        "",
        "## Dimension means",
        "",
        "| Citations | Factual | Comprehensive | Recency | Cleanliness | Overall |",
        "|---|---|---|---|---|---|",
        "| "
        + " | ".join(
            _fmt(value)
            for value in (
                means.get("citation_support"),
                means.get("factual_accuracy"),
                means.get("comprehensiveness"),
                means.get("recency"),
                means.get("report_cleanliness"),
                scorecard.get("overall_mean"),
            )
        )
        + " |",
        "",
        f"Judged: {scorecard.get('judged', 0)}  Failed: {scorecard.get('failed', 0)}",
        "",
    ]
    errored = [entry for entry in scorecard.get("questions", []) if entry.get("error")]
    if errored:
        lines.append("## Errors")
        lines.append("")
        for entry in errored:
            lines.append(f"- `{entry.get('id')}`: {entry.get('error')}")
        lines.append("")
    return "\n".join(lines)


def render_compare_md(card_a: dict[str, Any], card_b: dict[str, Any], name_a: str, name_b: str) -> str:
    """Render a per-question diff of two scorecard.json payloads."""
    by_id_a = {q.get("id"): q for q in card_a.get("questions", [])}
    by_id_b = {q.get("id"): q for q in card_b.get("questions", [])}
    all_ids = list(dict.fromkeys(list(by_id_a) + list(by_id_b)))

    def delta(a: Any, b: Any) -> str:
        if a is None or b is None:
            return "-"
        diff = round(float(b) - float(a), 2)
        return f"+{diff}" if diff > 0 else str(diff)

    lines = [
        f"# Scorecard comparison: {name_a} vs {name_b}",
        "",
        f"| Question | {name_a} overall | {name_b} overall | Delta |",
        "|---|---|---|---|",
    ]
    for qid in all_ids:
        a_overall = (by_id_a.get(qid) or {}).get("overall")
        b_overall = (by_id_b.get(qid) or {}).get("overall")
        lines.append(f"| {qid} | {_fmt(a_overall)} | {_fmt(b_overall)} | {delta(a_overall, b_overall)} |")

    lines += ["", "## Dimension means", "", f"| Dimension | {name_a} | {name_b} | Delta |", "|---|---|---|---|"]
    means_a = card_a.get("dimension_means", {})
    means_b = card_b.get("dimension_means", {})
    for dim in DIMENSIONS:
        lines.append(f"| {dim} | {_fmt(means_a.get(dim))} | {_fmt(means_b.get(dim))} | "
                     f"{delta(means_a.get(dim), means_b.get(dim))} |")
    lines.append(
        f"| overall | {_fmt(card_a.get('overall_mean'))} | {_fmt(card_b.get('overall_mean'))} | "
        f"{delta(card_a.get('overall_mean'), card_b.get('overall_mean'))} |"
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- judge API call


def call_judge(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    retries: int = JUDGE_RETRIES,
) -> dict[str, Any]:
    """Call the Anthropic-compatible /v1/messages endpoint; return parsed JSON scores."""
    headers = {
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.post(f"{base_url}/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            text = "".join(
                block.get("text", "")
                for block in body.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            return parse_judge_response(text)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("Judge call attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Judge call failed after {retries} attempts: {last_error}")


# --------------------------------------------------------------------------- run judging


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {q["id"]: q for q in (data or {}).get("questions", [])}


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def judge_run(
    run_dir: Path,
    questions: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    only_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Judge every question folder in the run dir; write per-question judge.json files."""
    results: list[dict[str, Any]] = []
    today = time.strftime("%Y-%m-%d")
    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
        for qdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            qid = qdir.name
            if only_ids and qid not in only_ids:
                continue
            question = questions.get(qid)
            if question is None:
                logger.warning("Skipping %s: not in questions file", qid)
                continue
            metrics = _load_json_if_exists(qdir / "metrics.json")
            depth = question.get("depth")
            job = _load_json_if_exists(qdir / "job.json")
            report_path = qdir / "report.md"
            base = {"id": qid, "depth": depth, "metrics": metrics}
            if not report_path.exists():
                error = "no report.md"
                if job:
                    error = f"no report.md (job status: {job.get('status')})"
                results.append({**base, "scores": dict.fromkeys(DIMENSIONS), "error": error})
                continue
            report = report_path.read_text(encoding="utf-8")
            recency_required = bool(question.get("rubric", {}).get("recency_required", False))
            try:
                prompt = build_judge_prompt(question, report, today=today)
                parsed = call_judge(client, base_url, api_key, model, prompt)
                scores, rationales = extract_scores(parsed, recency_required)
                result = {**base, "scores": scores, "rationales": rationales, "error": None}
            except Exception as exc:  # noqa: BLE001 - keep judging remaining questions
                logger.error("Judging %s failed: %s", qid, exc)
                result = {**base, "scores": dict.fromkeys(DIMENSIONS), "error": str(exc)}
            (qdir / "judge.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            results.append(result)
            logger.info("[%s] judged: %s", qid, result.get("scores"))
    return aggregate_scorecard(results)


# --------------------------------------------------------------------------- main


def resolve_run_dir(name_or_path: str, runs_dir: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate
    return runs_dir / name_or_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM judge for deep-research eval runs.")
    parser.add_argument("run", nargs="?", help="Run name (under scripts/eval/runs) or run directory path")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--only", help="Comma-separated question ids to judge")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SCORECARD_A", "SCORECARD_B"),
        help="Diff two scorecard.json files and print a markdown comparison (no judging)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    if args.compare:
        path_a, path_b = (Path(p) for p in args.compare)
        card_a = json.loads(path_a.read_text(encoding="utf-8"))
        card_b = json.loads(path_b.read_text(encoding="utf-8"))
        print(render_compare_md(card_a, card_b, path_a.parent.name or "A", path_b.parent.name or "B"))
        return 0

    if not args.run:
        logger.error("A run name/dir is required (or use --compare)")
        return 2

    run_dir = resolve_run_dir(args.run, args.runs_dir)
    if not run_dir.is_dir():
        logger.error("Run directory not found: %s", run_dir)
        return 2

    api_key = resolve_judge_api_key()
    if not api_key:
        logger.error("No judge API key found; set MINIMAX_API_KEY or ANTHROPIC_API_KEY")
        return 2
    base_url = resolve_judge_base_url()
    model = resolve_judge_model()
    logger.info("Judging %s with model %s via %s", run_dir, model, base_url)

    questions = load_questions(args.questions_file)
    only_ids = {item.strip() for item in args.only.split(",") if item.strip()} if args.only else None
    scorecard = judge_run(run_dir, questions, base_url, api_key, model, only_ids=only_ids)

    (run_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, default=str), encoding="utf-8")
    markdown = render_scorecard_md(scorecard, run_name=run_dir.name)
    (run_dir / "scorecard.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

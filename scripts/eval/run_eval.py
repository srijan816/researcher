# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark eval runner for the deep-research async job API.

For each selected benchmark question this script:

1. POSTs ``{"agent_type": "deep_researcher", "input": ..., "research_depth": ...}``
   to ``{AIQ_EVAL_BASE_URL}/v1/jobs/async/submit``.
2. Polls ``GET /v1/jobs/async/job/{job_id}`` with backoff until a terminal status.
3. Fetches ``/report?format=md`` and ``/state``.
4. Saves artifacts under ``scripts/eval/runs/<run-name>/<qid>/``:
   ``report.md``, ``state.json``, ``job.json``, ``metrics.json``, and
   ``error.json`` on failure.

Runs are resumable: question ids that already have a ``report.md`` in the run
directory are skipped.

Environment variables:
    AIQ_EVAL_BASE_URL              backend base URL (default http://127.0.0.1:9000)
    AIQ_EVAL_API_KEY               optional bearer token for authenticated backends
    AIQ_EVAL_JOB_TIMEOUT_SECONDS   per-job timeout (default 2400)

Usage:
    uv run python scripts/eval/run_eval.py --questions fin-hbm-memory-market,dec-vector-db-selection
    uv run python scripts/eval/run_eval.py --run-name baseline --concurrency 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger("eval.run")

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = EVAL_DIR / "questions.yaml"
DEFAULT_RUNS_DIR = EVAL_DIR / "runs"

DEFAULT_BASE_URL = "http://127.0.0.1:9000"
DEFAULT_JOB_TIMEOUT_SECONDS = 2400
TERMINAL_STATUSES = frozenset({"success", "failure", "interrupted", "not_found"})
VALID_DEPTHS = frozenset({"shallow", "medium", "deeper", "deep"})
POLL_INITIAL_SECONDS = 5.0
POLL_MAX_SECONDS = 30.0
POLL_BACKOFF_FACTOR = 1.5
HTTP_TIMEOUT_SECONDS = 60.0


# --------------------------------------------------------------------------- helpers


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the questions file."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    questions = (data or {}).get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def select_questions(questions: list[dict[str, Any]], ids_csv: str | None) -> list[dict[str, Any]]:
    """Filter questions by a comma-separated id list. Unknown ids are an error."""
    if not ids_csv:
        return list(questions)
    wanted = [item.strip() for item in ids_csv.split(",") if item.strip()]
    by_id = {q["id"]: q for q in questions}
    unknown = [qid for qid in wanted if qid not in by_id]
    if unknown:
        raise ValueError(f"Unknown question ids: {', '.join(unknown)}")
    return [by_id[qid] for qid in wanted]


def auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_metrics(job: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    """Best-effort metrics extraction from job/state payloads.

    The backend does not guarantee a metrics field; collect whatever is
    available (job.metrics, output.metrics, quality_verification artifact
    metrics, tool/search call counts).
    """
    metrics: dict[str, Any] = {}
    if isinstance(job.get("metrics"), dict):
        metrics.update(job["metrics"])
    output = job.get("output")
    if isinstance(output, dict) and isinstance(output.get("metrics"), dict):
        metrics.update(output["metrics"])

    artifacts = (state or {}).get("artifacts")
    if isinstance(artifacts, dict):
        for name in ("quality_verification.json", "quality_verification"):
            artifact = artifacts.get(name)
            if isinstance(artifact, dict):
                content = artifact.get("content", artifact)
                if isinstance(content, dict) and isinstance(content.get("metrics"), dict):
                    metrics.update(content["metrics"])
        tool_calls = artifacts.get("tool_calls")
        if isinstance(tool_calls, list):
            metrics.setdefault("tool_calls", len(tool_calls))
            search_calls = sum(
                1
                for call in tool_calls
                if isinstance(call, dict) and "search" in str(call.get("name", call.get("tool", ""))).lower()
            )
            if search_calls:
                metrics.setdefault("search_calls", search_calls)
        sources = artifacts.get("sources")
        if isinstance(sources, dict):
            for key in ("found", "cited"):
                value = sources.get(key)
                if isinstance(value, int):
                    metrics.setdefault(f"sources_{key}", value)
    return metrics


# --------------------------------------------------------------------------- API calls


def submit_job(client: httpx.Client, base_url: str, prompt: str, depth: str) -> str:
    payload = {"agent_type": "deep_researcher", "input": prompt, "research_depth": depth}
    response = client.post(f"{base_url}/v1/jobs/async/submit", json=payload)
    response.raise_for_status()
    body = response.json()
    job_id = body.get("job_id")
    if not job_id:
        raise RuntimeError(f"Submit response missing job_id: {body}")
    return str(job_id)


def poll_until_terminal(
    client: httpx.Client,
    base_url: str,
    job_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Poll job status with backoff until terminal or timeout."""
    deadline = time.monotonic() + timeout_seconds
    interval = POLL_INITIAL_SECONDS
    last_job: dict[str, Any] = {}
    while True:
        try:
            response = client.get(f"{base_url}/v1/jobs/async/job/{job_id}")
            response.raise_for_status()
            last_job = response.json()
        except httpx.HTTPError as exc:
            logger.warning("Polling %s failed (%s); retrying", job_id, exc)
        else:
            status = str(last_job.get("status", "")).lower()
            if last_job.get("terminal") is True or status in TERMINAL_STATUSES:
                return last_job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Job {job_id} not terminal after {timeout_seconds:.0f}s (last: {last_job})")
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        interval = min(interval * POLL_BACKOFF_FACTOR, POLL_MAX_SECONDS)


def fetch_report_md(client: httpx.Client, base_url: str, job_id: str) -> str | None:
    response = client.get(f"{base_url}/v1/jobs/async/job/{job_id}/report", params={"format": "md"})
    content_type = response.headers.get("content-type", "")
    if response.status_code == 200 and "markdown" in content_type:
        return response.text
    logger.warning("No markdown report for job %s (status=%s)", job_id, response.status_code)
    return None


def fetch_state(client: httpx.Client, base_url: str, job_id: str) -> dict[str, Any] | None:
    try:
        response = client.get(f"{base_url}/v1/jobs/async/job/{job_id}/state")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        logger.warning("Could not fetch state for job %s: %s", job_id, exc)
        return None


# --------------------------------------------------------------------------- per question


def run_question(
    question: dict[str, Any],
    run_dir: Path,
    base_url: str,
    api_key: str | None,
    job_timeout_seconds: float,
    depth_override: str | None,
) -> dict[str, Any]:
    """Run one benchmark question end-to-end. Never raises; returns a summary dict."""
    qid = question["id"]
    qdir = run_dir / qid
    qdir.mkdir(parents=True, exist_ok=True)
    depth = depth_override or question.get("depth", "deeper")
    summary: dict[str, Any] = {"id": qid, "depth": depth, "status": "error", "job_id": None}

    started = time.monotonic()
    try:
        with httpx.Client(headers=auth_headers(api_key), timeout=HTTP_TIMEOUT_SECONDS) as client:
            job_id = submit_job(client, base_url, question["prompt"], depth)
            summary["job_id"] = job_id
            logger.info("[%s] submitted job %s (depth=%s)", qid, job_id, depth)

            job = poll_until_terminal(client, base_url, job_id, job_timeout_seconds)
            summary["status"] = str(job.get("status", "unknown"))
            (qdir / "job.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")

            state = fetch_state(client, base_url, job_id)
            if state is not None:
                (qdir / "state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

            metrics = extract_metrics(job, state)
            metrics["runner_wall_seconds"] = round(time.monotonic() - started, 1)
            (qdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

            report = fetch_report_md(client, base_url, job_id)
            if report:
                (qdir / "report.md").write_text(report, encoding="utf-8")
                summary["report"] = True
            else:
                summary["report"] = False
                if summary["status"] == "success":
                    summary["status"] = "success_no_report"
    except Exception as exc:  # noqa: BLE001 - record failure, keep the run going
        logger.error("[%s] failed: %s", qid, exc)
        summary["error"] = str(exc)
        (qdir / "error.json").write_text(
            json.dumps({"id": qid, "error": str(exc), "job_id": summary.get("job_id")}, indent=2),
            encoding="utf-8",
        )
    summary["wall_seconds"] = round(time.monotonic() - started, 1)
    return summary


# --------------------------------------------------------------------------- main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deep-research benchmark questions against a local backend.")
    parser.add_argument("--questions", help="Comma-separated question ids (default: all)")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--depth", choices=sorted(VALID_DEPTHS), help="Override research depth for all questions")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel jobs (default 1)")
    parser.add_argument("--run-name", default=None, help="Run folder name (default: timestamp)")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    base_url = os.environ.get("AIQ_EVAL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    api_key = os.environ.get("AIQ_EVAL_API_KEY") or None
    job_timeout = float(os.environ.get("AIQ_EVAL_JOB_TIMEOUT_SECONDS", DEFAULT_JOB_TIMEOUT_SECONDS))

    questions = load_questions(args.questions_file)
    try:
        selected = select_questions(questions, args.questions)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = []
    for question in selected:
        if (run_dir / question["id"] / "report.md").exists():
            skipped.append(question["id"])
        else:
            pending.append(question)
    if skipped:
        logger.info("Skipping %d question(s) with existing reports: %s", len(skipped), ", ".join(skipped))

    summaries: list[dict[str, Any]] = [{"id": qid, "status": "skipped_existing"} for qid in skipped]
    if pending:
        workers = max(1, args.concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(run_question, question, run_dir, base_url, api_key, job_timeout, args.depth)
                for question in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                summary = future.result()
                summaries.append(summary)
                logger.info("[%s] done: %s", summary["id"], summary["status"])

    summary_payload = {
        "run_name": run_name,
        "base_url": base_url,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "questions": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    logger.info("Run complete: %s", run_dir)

    failures = [s for s in summaries if s.get("status") not in {"success", "skipped_existing"}]
    if failures:
        logger.warning("%d question(s) did not finish with success: %s", len(failures), [s["id"] for s in failures])
    return 0


if __name__ == "__main__":
    sys.exit(main())

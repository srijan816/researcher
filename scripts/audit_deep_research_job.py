#!/usr/bin/env python3
"""Inspect persisted async deep-research job artifacts.

Usage:
    scripts/audit_deep_research_job.py [job_id] [--db jobs.db]

If job_id is omitted, the most recently created job is audited.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


def _json_value(payload: str, path: list[str], default=None):
    value = json.loads(payload)
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def _latest_job(conn: sqlite3.Connection) -> str:
    row = conn.execute("select job_id from job_info order by created_at desc limit 1").fetchone()
    if not row:
        raise SystemExit("No jobs found.")
    return str(row[0])


def audit_job(db_path: Path, job_id: str | None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    job_id = job_id or _latest_job(conn)

    job = conn.execute(
        "select job_id,status,created_at,updated_at,length(output) as output_len,error from job_info where job_id=?",
        (job_id,),
    ).fetchone()
    if not job:
        raise SystemExit(f"Job not found: {job_id}")

    events = conn.execute(
        "select id,event_type,event_data from job_events where job_id=? order by id",
        (job_id,),
    ).fetchall()

    event_types = Counter(row["event_type"] for row in events)
    artifact_types = Counter()
    output_categories = Counter()
    source_urls: list[str] = []
    cited_urls: list[str] = []
    files: list[tuple[int, str, int]] = []
    failed_workflows: list[tuple[int, str, str]] = []
    search_samples: list[tuple[int, int, int, bool, str]] = []
    scrape_artifacts: list[tuple[int, str, str, str, int]] = []

    for row in events:
        payload = row["event_data"]
        event_type = row["event_type"]
        data_type = _json_value(payload, ["data", "type"])
        if data_type:
            artifact_types[str(data_type)] += 1
        category = _json_value(payload, ["data", "output_category"])
        if category:
            output_categories[str(category)] += 1

        if event_type == "artifact.update" and data_type == "citation_source":
            url = _json_value(payload, ["data", "url"]) or _json_value(payload, ["data", "content"])
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                source_urls.append(url)
        elif event_type == "artifact.update" and data_type == "citation_use":
            url = _json_value(payload, ["data", "url"]) or _json_value(payload, ["data", "content"])
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                cited_urls.append(url)
        elif event_type == "artifact.update" and data_type == "file":
            name = _json_value(payload, ["name"]) or _json_value(payload, ["data", "file_path"]) or ""
            content = _json_value(payload, ["data", "content"], "")
            files.append((row["id"], str(name), len(str(content))))
        elif event_type == "artifact.update" and category == "search_result":
            content = str(_json_value(payload, ["data", "content"], ""))
            docs = len(re.findall(r"<document\b", content))
            urls = len(re.findall(r"<url>https?://", content))
            truncated = "[... search result truncated ...]" in content
            first_title = ""
            match = re.search(r"<title>(.*?)</title>", content, re.S)
            if match:
                first_title = re.sub(r"\s+", " ", match.group(1)).strip()
            search_samples.append((row["id"], docs, urls, truncated, first_title[:100]))
        elif event_type == "artifact.update" and data_type == "scrape_artifact":
            url = str(_json_value(payload, ["data", "url"], ""))
            path = str(_json_value(payload, ["data", "artifact_path"], ""))
            status = str(_json_value(payload, ["data", "extraction_status"], ""))
            length = int(_json_value(payload, ["data", "content_length"], 0) or 0)
            scrape_artifacts.append((row["id"], url, path, status, length))
        elif event_type == "workflow.end":
            output = _json_value(payload, ["data", "output"], "")
            if isinstance(output, str) and "Model call failed" in output:
                failed_workflows.append((row["id"], str(_json_value(payload, ["name"], "")), output[:160]))

    print(f"Job: {job['job_id']}")
    print(f"Status: {job['status']}  Created: {job['created_at']}  Updated: {job['updated_at']}")
    print(f"Output length: {job['output_len'] or 0}")
    if job["error"]:
        print(f"Error: {job['error']}")
    print()

    print("Event types:")
    for key, value in event_types.most_common():
        print(f"  {key}: {value}")
    print("Artifact types:")
    for key, value in artifact_types.most_common():
        print(f"  {key}: {value}")
    print("Output categories:")
    for key, value in output_categories.most_common():
        print(f"  {key}: {value}")
    print()

    print(f"Sources found: {len(set(source_urls))}")
    print(f"Sources cited: {len(set(cited_urls))}")
    if source_urls:
        print("First found sources:")
        for url in list(dict.fromkeys(source_urls))[:15]:
            print(f"  - {url}")
    if cited_urls:
        print("First cited sources:")
        for url in list(dict.fromkeys(cited_urls))[:15]:
            print(f"  - {url}")
    print()

    if files:
        print("File artifacts:")
        for event_id, name, size in files:
            print(f"  {event_id}: {name} ({size} chars)")
    if failed_workflows:
        print("Failed workflow endings:")
        for event_id, name, output in failed_workflows:
            print(f"  {event_id}: {name}: {output}")
    if search_samples:
        print("Search-result snapshots:")
        for event_id, docs, urls, truncated, title in search_samples[:12]:
            marker = "truncated" if truncated else "complete"
            print(f"  {event_id}: docs={docs} urls={urls} {marker} first={title}")
    if scrape_artifacts:
        print("Durable scrape artifacts:")
        print(f"  Total: {len(scrape_artifacts)}")
        for event_id, url, path, status, length in scrape_artifacts[:12]:
            print(f"  {event_id}: {status} chars={length} url={url} path={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--db", default="jobs.db")
    args = parser.parse_args()
    audit_job(Path(args.db), args.job_id)


if __name__ == "__main__":
    main()

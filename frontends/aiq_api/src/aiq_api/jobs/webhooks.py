# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal job webhook delivery for async API clients."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

import httpx

from .event_store import BatchingEventStore
from .event_store import EventStore

logger = logging.getLogger(__name__)


def build_job_webhook_config(
    *,
    url: str | None,
    headers: dict[str, str] | None = None,
    secret: str | None = None,
) -> dict[str, Any] | None:
    """Build the serializable webhook config passed to the Dask worker."""
    if not url:
        return None
    safe_headers = {
        str(key): str(value)
        for key, value in (headers or {}).items()
        if key and value is not None and "\n" not in str(key) and "\n" not in str(value)
    }
    return {
        "url": str(url),
        "headers": safe_headers,
        "secret": secret or None,
    }


def build_terminal_payload(
    *,
    job_id: str,
    status: str,
    agent_config_name: str,
    agent_class_path: str,
    research_depth: str,
    terminal: bool = True,
    has_report: bool = False,
    error: str | None = None,
    quality_warnings: list[str] | None = None,
    recovered: bool = False,
) -> dict[str, Any]:
    """Build the public webhook body clients receive when a job reaches a terminal state."""
    return {
        "event": "job.terminal",
        "job_id": job_id,
        "status": status,
        "terminal": terminal,
        "success": status == "success",
        "agent_config_name": agent_config_name,
        "agent_class_path": agent_class_path,
        "research_depth": research_depth,
        "has_report": has_report,
        "report_ready": has_report,
        "recovered": recovered,
        "error": error,
        "quality_warnings": quality_warnings or [],
        "links": {
            "status_url": f"/v1/jobs/async/job/{job_id}",
            "report_url": f"/v1/jobs/async/job/{job_id}/report",
            "state_url": f"/v1/jobs/async/job/{job_id}/state",
            "stream_url": f"/v1/jobs/async/job/{job_id}/stream",
        },
    }


def _signed_headers(config: dict[str, Any], body: bytes) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AIQ-Deep-Research-Webhook/1.0",
        **dict(config.get("headers") or {}),
    }
    secret = config.get("secret")
    if secret:
        timestamp = str(int(time.time()))
        signature = hmac.new(
            str(secret).encode("utf-8"),
            timestamp.encode("utf-8") + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        headers["X-AIQ-Webhook-Timestamp"] = timestamp
        headers["X-AIQ-Webhook-Signature"] = f"sha256={signature}"
    return headers


async def send_job_webhook(
    *,
    config: dict[str, Any] | None,
    payload: dict[str, Any],
    event_store: EventStore | BatchingEventStore | None = None,
) -> None:
    """POST a terminal job webhook without letting delivery failures fail the job."""
    if not config or not config.get("url"):
        return

    url = str(config["url"])
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(config, body)
    retries = max(1, int(os.environ.get("AIQ_JOB_WEBHOOK_RETRIES", "3")))
    timeout_seconds = max(1.0, float(os.environ.get("AIQ_JOB_WEBHOOK_TIMEOUT_SECONDS", "10")))
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False) as client:
                response = await client.post(url, content=body, headers=headers)
                response.raise_for_status()
            if event_store is not None:
                event_store.store(
                    {
                        "type": "job.webhook.delivered",
                        "data": {
                            "url": url,
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "event": payload.get("event"),
                        },
                    }
                )
            return
        except Exception as exc:  # pragma: no cover - exact httpx exception mix is environment-specific
            last_error = str(exc)
            logger.warning("Job webhook delivery failed for %s on attempt %d/%d: %s", url, attempt, retries, exc)
            if attempt < retries:
                await asyncio_sleep_backoff(attempt)

    if event_store is not None:
        event_store.store(
            {
                "type": "job.webhook.failed",
                "data": {
                    "url": url,
                    "attempts": retries,
                    "event": payload.get("event"),
                    "error": last_error,
                },
            }
        )


async def asyncio_sleep_backoff(attempt: int) -> None:
    """Sleep between webhook retries; isolated for tests."""
    import asyncio

    await asyncio.sleep(min(2.0, 0.25 * attempt))

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Server-side research queue routes.

Routes (mounted at /v1/jobs/queue):
    POST   /v1/jobs/queue              - Add a queued/scheduled/recurring research item
    GET    /v1/jobs/queue              - List queue items (optional ?status= filter)
    POST   /v1/jobs/queue/{id}/approve - Approve a queued item for execution
    POST   /v1/jobs/queue/{id}/cancel  - Cancel a queued/approved item
    DELETE /v1/jobs/queue/{id}         - Delete a non-running item

The durable queue complements the legacy frontend-only batch queue: items
persist across backend restarts and are drained by the background scheduler in
:mod:`aiq_api.jobs.queue_scheduler`, which submits through the existing
internal submit path.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import AnyHttpUrl
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import SecretStr

from aiq_agent.common import DEFAULT_RESEARCH_DEPTH
from aiq_agent.common import ResearchDepthTier

logger = logging.getLogger(__name__)


class QueueItemCreateRequest(BaseModel):
    """Request to add a research item to the server-side queue."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "agent_type": "deep_researcher",
                    "input": "Track weekly developments in EU AI Act enforcement.",
                    "research_depth": "deeper",
                    "priority": 5,
                    "scheduled_for": "2026-06-11T08:00:00Z",
                    "recurrence_seconds": 604800,
                    "auto_approve": True,
                }
            ]
        }
    )

    agent_type: str = Field(..., description="Agent type (e.g., 'deep_researcher')")
    input: str = Field(..., min_length=1, description="Research query for the agent")
    research_depth: ResearchDepthTier = Field(
        DEFAULT_RESEARCH_DEPTH,
        description="Depth/source tier: shallow, medium, deeper, or deep.",
    )
    priority: int = Field(0, ge=-100, le=100, description="Higher priority items run first")
    scheduled_for: datetime | None = Field(
        None,
        description="Optional earliest run time (ISO 8601). Omit to run as soon as a slot is free.",
    )
    recurrence_seconds: int | None = Field(
        None,
        ge=60,
        le=31536000,
        description="Optional recurrence interval; after each run the item re-approves itself at now + interval.",
    )
    auto_approve: bool = Field(True, description="Approve immediately (status=approved) instead of status=queued")
    webhook_url: AnyHttpUrl | None = Field(None, description="Optional terminal webhook for each spawned job")
    webhook_headers: dict[str, str] | None = Field(
        None, max_length=10, description="Optional static headers for the terminal webhook"
    )
    webhook_secret: SecretStr | None = Field(
        None, min_length=8, max_length=512, description="Optional HMAC secret for webhook signing"
    )


class QueueItemResponse(BaseModel):
    """A persisted research queue item."""

    id: int
    status: str
    agent_type: str
    input: str
    research_depth: str
    priority: int = 0
    scheduled_for: str | None = None
    recurrence_seconds: int | None = None
    owner_subject: str | None = None
    owner_display_name: str | None = None
    last_job_id: str | None = None
    last_run_at: str | None = None
    last_error: str | None = None
    webhook_url: str | None = None
    has_webhook_secret: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    job_status_url: str | None = Field(None, description="Status URL for the most recently spawned job")
    job_report_url: str | None = Field(None, description="Report URL for the most recently spawned job")


class QueueListResponse(BaseModel):
    """Queue items visible to the caller."""

    items: list[QueueItemResponse]


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _item_to_response(item: dict[str, Any]) -> QueueItemResponse:
    last_job_id = item.get("last_job_id")
    return QueueItemResponse(
        id=int(item["id"]),
        status=str(item.get("status")),
        agent_type=str(item.get("agent_type")),
        input=str(item.get("input")),
        research_depth=str(item.get("research_depth")),
        priority=int(item.get("priority") or 0),
        scheduled_for=_iso_or_none(item.get("scheduled_for")),
        recurrence_seconds=item.get("recurrence_seconds"),
        owner_subject=item.get("owner_subject"),
        owner_display_name=item.get("owner_email") or item.get("owner_subject"),
        last_job_id=last_job_id,
        last_run_at=_iso_or_none(item.get("last_run_at")),
        last_error=item.get("last_error"),
        webhook_url=item.get("webhook_url"),
        has_webhook_secret=bool(item.get("webhook_secret")),
        created_at=_iso_or_none(item.get("created_at")),
        updated_at=_iso_or_none(item.get("updated_at")),
        job_status_url=f"/v1/jobs/async/job/{last_job_id}" if last_job_id else None,
        job_report_url=f"/v1/jobs/async/job/{last_job_id}/report" if last_job_id else None,
    )


def _auth_enabled() -> bool:
    return os.environ.get("REQUIRE_AUTH", "false").lower() == "true"


def _principal_owns_item(principal, item: dict[str, Any]) -> bool:
    if getattr(principal, "role", None) == "admin":
        return True
    return principal.type == item.get("owner_auth_type") and principal.sub == item.get("owner_subject")


async def register_queue_routes(app: FastAPI, db_url: str) -> None:
    """Register server-side research queue routes and start the scheduler."""
    from ..jobs.access import require_verified_principal
    from ..jobs.queue_scheduler import start_queue_scheduler
    from ..jobs.queue_store import QUEUE_STATUS_APPROVED
    from ..jobs.queue_store import QUEUE_STATUS_CANCELLED
    from ..jobs.queue_store import QUEUE_STATUS_QUEUED
    from ..jobs.queue_store import QUEUE_STATUS_RUNNING
    from ..jobs.queue_store import QUEUE_STATUSES
    from ..jobs.queue_store import QueueStore
    from ..registry import get_agent_config

    loop = asyncio.get_running_loop()
    # Build once so schema creation happens at startup, not first request.
    await loop.run_in_executor(None, QueueStore, db_url)

    def _store() -> QueueStore:
        return QueueStore(db_url)

    async def _load_authorized_item(item_id: int, principal) -> dict[str, Any]:
        item = await asyncio.get_running_loop().run_in_executor(None, lambda: _store().get_item(item_id))
        if item is None:
            raise HTTPException(404, f"Queue item not found: {item_id}")
        if _auth_enabled() and not _principal_owns_item(principal, item):
            raise HTTPException(404, f"Queue item not found: {item_id}")
        return item

    @app.post(
        "/v1/jobs/queue",
        response_model=QueueItemResponse,
        tags=["research queue"],
        summary="Add a research item to the server-side queue",
        responses={400: {"description": "Unknown agent type or invalid request"}},
    )
    async def add_queue_item(req: QueueItemCreateRequest) -> QueueItemResponse:
        try:
            get_agent_config(req.agent_type)
        except KeyError as e:
            raise HTTPException(400, str(e))

        principal = require_verified_principal()
        item = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _store().add_item(
                owner_auth_type=principal.type,
                owner_subject=principal.sub,
                owner_email=principal.email,
                agent_type=req.agent_type,
                input_text=req.input,
                research_depth=str(req.research_depth),
                priority=req.priority,
                scheduled_for=req.scheduled_for,
                recurrence_seconds=req.recurrence_seconds,
                auto_approve=req.auto_approve,
                webhook_url=str(req.webhook_url) if req.webhook_url else None,
                webhook_headers=req.webhook_headers,
                webhook_secret=req.webhook_secret.get_secret_value() if req.webhook_secret else None,
            ),
        )
        logger.info(
            "Queued research item %s (%s, status=%s) for %s:%s",
            item["id"],
            req.agent_type,
            item["status"],
            principal.type,
            principal.sub,
        )
        return _item_to_response(item)

    @app.get(
        "/v1/jobs/queue",
        response_model=QueueListResponse,
        tags=["research queue"],
        summary="List server-side research queue items",
    )
    async def list_queue_items(status: str | None = None, limit: int = 100) -> QueueListResponse:
        if status is not None and status not in QUEUE_STATUSES:
            raise HTTPException(400, f"Unknown status filter: {status}")
        principal = require_verified_principal()

        owner_auth_type = owner_subject = None
        if _auth_enabled() and getattr(principal, "role", None) != "admin":
            owner_auth_type = principal.type
            owner_subject = principal.sub

        items = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _store().list_items(
                status=status,
                owner_auth_type=owner_auth_type,
                owner_subject=owner_subject,
                limit=limit,
            ),
        )
        return QueueListResponse(items=[_item_to_response(item) for item in items])

    @app.post(
        "/v1/jobs/queue/{item_id}/approve",
        response_model=QueueItemResponse,
        tags=["research queue"],
        summary="Approve a queued research item",
        responses={400: {"description": "Item cannot be approved from its current status"}},
    )
    async def approve_queue_item(item_id: int) -> QueueItemResponse:
        principal = require_verified_principal()
        item = await _load_authorized_item(item_id, principal)
        if item["status"] == QUEUE_STATUS_APPROVED:
            return _item_to_response(item)
        if item["status"] not in (QUEUE_STATUS_QUEUED, QUEUE_STATUS_CANCELLED):
            raise HTTPException(400, f"Queue item {item_id} cannot be approved from status {item['status']}")
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _store().update_status(
                item_id,
                QUEUE_STATUS_APPROVED,
                expected_statuses=(QUEUE_STATUS_QUEUED, QUEUE_STATUS_CANCELLED),
            ),
        )
        refreshed = await _load_authorized_item(item_id, principal)
        return _item_to_response(refreshed)

    @app.post(
        "/v1/jobs/queue/{item_id}/cancel",
        response_model=QueueItemResponse,
        tags=["research queue"],
        summary="Cancel a queued or approved research item",
        responses={400: {"description": "Item cannot be cancelled from its current status"}},
    )
    async def cancel_queue_item(item_id: int) -> QueueItemResponse:
        principal = require_verified_principal()
        item = await _load_authorized_item(item_id, principal)
        if item["status"] == QUEUE_STATUS_CANCELLED:
            return _item_to_response(item)
        if item["status"] not in (QUEUE_STATUS_QUEUED, QUEUE_STATUS_APPROVED):
            raise HTTPException(
                400,
                f"Queue item {item_id} cannot be cancelled from status {item['status']}. "
                "Cancel the running job via /v1/jobs/async/job/{job_id}/cancel instead.",
            )
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _store().update_status(
                item_id,
                QUEUE_STATUS_CANCELLED,
                expected_statuses=(QUEUE_STATUS_QUEUED, QUEUE_STATUS_APPROVED),
            ),
        )
        refreshed = await _load_authorized_item(item_id, principal)
        return _item_to_response(refreshed)

    @app.delete(
        "/v1/jobs/queue/{item_id}",
        tags=["research queue"],
        summary="Delete a research queue item",
        responses={400: {"description": "Running items cannot be deleted"}},
    )
    async def delete_queue_item(item_id: int) -> dict:
        principal = require_verified_principal()
        item = await _load_authorized_item(item_id, principal)
        if item["status"] == QUEUE_STATUS_RUNNING:
            raise HTTPException(400, f"Queue item {item_id} is running; cancel its job first")
        deleted = await asyncio.get_running_loop().run_in_executor(None, lambda: _store().delete_item(item_id))
        return {"id": item_id, "deleted": bool(deleted)}

    logger.info("Registered research queue routes at /v1/jobs/queue")

    # The scheduler needs the Dask-backed submit path; without a scheduler
    # address submissions would always fail and drain the queue erroneously.
    if os.environ.get("NAT_DASK_SCHEDULER_ADDRESS"):
        start_queue_scheduler(db_url)
    else:
        logger.info("Research queue scheduler not started: NAT_DASK_SCHEDULER_ADDRESS is not set")

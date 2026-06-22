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

"""Background scheduler that drains the durable research queue.

Polls ``research_queue`` every ``AIQ_QUEUE_POLL_SECONDS`` and, while fewer than
``AIQ_QUEUE_MAX_CONCURRENT`` queue-spawned jobs are running, claims the
highest-priority approved item that is due and submits it through the existing
internal submit path (:func:`aiq_api.jobs.submit.submit_agent_job` — no HTTP
self-calls). Terminal job state is tracked through ``job_info``; recurring
items re-approve themselves with ``scheduled_for = now + recurrence_seconds``.

Kill switch: ``AIQ_QUEUE_ENABLED`` (default ``1``). All failures are logged and
never fatal to the app.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable
from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from typing import Any

from .queue_store import QUEUE_STATUS_APPROVED
from .queue_store import QUEUE_STATUS_COMPLETED
from .queue_store import QUEUE_STATUS_FAILED
from .queue_store import QueueStore
from .queue_store import utc_now

logger = logging.getLogger(__name__)

# Job statuses considered terminal in job_info (NAT JobStatus values plus aliases).
_TERMINAL_SUCCESS = {"success"}
_TERMINAL_FAILURE = {"failure", "failed", "interrupted", "cancelled"}

# Sentinel returned by the status probe when the job row is definitively gone.
JOB_STATUS_MISSING = "__missing__"


def queue_enabled() -> bool:
    """Return True when the server-side queue scheduler is enabled."""
    # @environment_variable AIQ_QUEUE_ENABLED
    # @category Server
    # @type bool
    # @default 1
    # Kill switch for the server-side research queue scheduler.
    return os.environ.get("AIQ_QUEUE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def queue_poll_seconds() -> float:
    # @environment_variable AIQ_QUEUE_POLL_SECONDS
    # @category Server
    # @type float
    # @default 10
    # Poll interval for the research queue scheduler.
    try:
        return max(1.0, float(os.environ.get("AIQ_QUEUE_POLL_SECONDS", "10")))
    except ValueError:
        return 10.0


def queue_max_concurrent() -> int:
    # @environment_variable AIQ_QUEUE_MAX_CONCURRENT
    # @category Server
    # @type int
    # @default 1
    # Maximum queue-spawned research jobs running at once.
    try:
        return max(1, int(os.environ.get("AIQ_QUEUE_MAX_CONCURRENT", "1")))
    except ValueError:
        return 1


def _default_job_status_probe(db_url: str) -> Callable[[str], str | None]:
    """Build a sync ``job_id -> status`` probe over the job_info table.

    Returns ``JOB_STATUS_MISSING`` when the job row is definitively absent and
    ``None`` when the status cannot be determined (probe errors, missing
    table) so callers can skip rather than misclassify.
    """

    def probe(job_id: str) -> str | None:
        try:
            from sqlalchemy import inspect
            from sqlalchemy import text

            from .event_store import EventStore

            engine = EventStore._get_or_create_sync_engine(db_url)
            if not inspect(engine).has_table("job_info"):
                return None
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT status FROM job_info WHERE job_id = :job_id"),
                    {"job_id": job_id},
                ).first()
                if row is None:
                    return JOB_STATUS_MISSING
                return str(row[0]).lower() if row[0] is not None else None
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            logger.warning("Queue scheduler could not probe job %s status: %s", job_id, exc)
            return None

    return probe


async def _default_submit(item: dict[str, Any]) -> str:
    """Submit a queue item through the existing internal submit path."""
    from aiq_agent.auth import Principal

    from .submit import submit_agent_job

    principal = Principal(
        type=str(item.get("owner_auth_type") or "anonymous"),
        sub=str(item.get("owner_subject") or "queue"),
        email=item.get("owner_email"),
    )
    return await submit_agent_job(
        agent_type=str(item["agent_type"]),
        input_text=str(item["input"]),
        owner=item.get("owner_email") or str(item.get("owner_subject") or "queue"),
        principal=principal,
        research_depth=item.get("research_depth") or "deeper",
        webhook_url=item.get("webhook_url"),
        webhook_headers=item.get("webhook_headers") if isinstance(item.get("webhook_headers"), dict) else None,
        webhook_secret=item.get("webhook_secret"),
    )


class QueueScheduler:
    """Async background scheduler for the durable research queue.

    All dependencies (submit callable, job-status probe, clock) are injectable
    for tests; production wiring uses the internal submit path and job_info.
    """

    def __init__(
        self,
        db_url: str,
        *,
        submit_fn: Callable[[dict[str, Any]], Awaitable[str]] | None = None,
        job_status_fn: Callable[[str], str | None] | None = None,
        now_fn: Callable[[], datetime] | None = None,
        poll_seconds: float | None = None,
        max_concurrent: int | None = None,
    ):
        self.db_url = db_url
        self._submit_fn = submit_fn or _default_submit
        self._job_status_fn = job_status_fn or _default_job_status_probe(db_url)
        self._now_fn = now_fn or utc_now
        self._poll_seconds = poll_seconds if poll_seconds is not None else queue_poll_seconds()
        self._max_concurrent = max_concurrent if max_concurrent is not None else queue_max_concurrent()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Research queue scheduler started (poll=%ss, max_concurrent=%d)",
            self._poll_seconds,
            self._max_concurrent,
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Research queue scheduler stopped")

    async def _run_loop(self) -> None:
        # Reconcile immediately on startup so items stuck in `running` whose
        # job already hit a terminal state are repaired before new submits.
        try:
            await self.run_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - scheduler must never die
            logger.warning("Queue scheduler startup cycle failed: %s", exc)

        while True:
            try:
                await asyncio.sleep(self._poll_seconds)
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 - scheduler must never die
                logger.warning("Queue scheduler cycle failed: %s", exc)

    # ------------------------------------------------------------------
    # One scheduling cycle (also the test entry point)
    # ------------------------------------------------------------------

    async def run_once(self) -> None:
        """Run one reconcile + dispatch cycle."""
        loop = asyncio.get_running_loop()
        store = await loop.run_in_executor(None, QueueStore, self.db_url)
        await self._reconcile_running(store)
        await self._dispatch_due(store)

    async def _reconcile_running(self, store: QueueStore) -> None:
        loop = asyncio.get_running_loop()
        running = await loop.run_in_executor(None, store.list_running)
        for item in running:
            try:
                await self._reconcile_item(store, item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to reconcile queue item %s: %s", item.get("id"), exc)

    async def _reconcile_item(self, store: QueueStore, item: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        item_id = int(item["id"])
        job_id = item.get("last_job_id")

        if not job_id:
            # Claimed but never submitted (e.g. backend died mid-claim): re-approve.
            logger.warning("Queue item %d was running without a job id; re-approving", item_id)
            await loop.run_in_executor(
                None,
                lambda: store.update_status(item_id, QUEUE_STATUS_APPROVED, now=self._now_fn()),
            )
            return

        job_status = await loop.run_in_executor(None, self._job_status_fn, str(job_id))
        if job_status is None:
            return  # Undeterminable this cycle; leave running.

        if job_status in _TERMINAL_SUCCESS:
            await self._finish_item(store, item, QUEUE_STATUS_COMPLETED, error=None)
        elif job_status in _TERMINAL_FAILURE or job_status == JOB_STATUS_MISSING:
            error = (
                f"job {job_id} record is missing"
                if job_status == JOB_STATUS_MISSING
                else f"job {job_id} finished with status {job_status}"
            )
            await self._finish_item(store, item, QUEUE_STATUS_FAILED, error=error)
        # Otherwise the job is still active; nothing to do.

    async def _finish_item(
        self,
        store: QueueStore,
        item: dict[str, Any],
        terminal_status: str,
        *,
        error: str | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        item_id = int(item["id"])
        recurrence = item.get("recurrence_seconds")
        now = self._now_fn()

        if recurrence:
            next_run = now + timedelta(seconds=int(recurrence))
            await loop.run_in_executor(
                None,
                lambda: store.reschedule_recurring(item_id, next_run, last_error=error, now=now),
            )
            logger.info(
                "Recurring queue item %d (job %s) %s; next run at %s",
                item_id,
                item.get("last_job_id"),
                terminal_status,
                next_run.isoformat(),
            )
            return

        await loop.run_in_executor(
            None,
            lambda: store.update_status(item_id, terminal_status, last_error=error, now=now),
        )
        logger.info("Queue item %d (job %s) marked %s", item_id, item.get("last_job_id"), terminal_status)

    async def _dispatch_due(self, store: QueueStore) -> None:
        loop = asyncio.get_running_loop()
        while True:
            running_count = await loop.run_in_executor(None, store.count_running)
            if running_count >= self._max_concurrent:
                return

            now = self._now_fn()
            item = await loop.run_in_executor(None, store.claim_next_due, now)
            if item is None:
                return

            item_id = int(item["id"])
            try:
                job_id = await self._submit_fn(item)
            except Exception as exc:  # noqa: BLE001 - submit failures must not kill the loop
                logger.warning("Queue item %d submission failed: %s", item_id, exc)
                await self._handle_submit_failure(store, item, str(exc))
                continue

            await loop.run_in_executor(None, lambda: store.record_submission(item_id, job_id, self._now_fn()))
            logger.info("Queue item %d submitted as job %s", item_id, job_id)

    async def _handle_submit_failure(self, store: QueueStore, item: dict[str, Any], error: str) -> None:
        loop = asyncio.get_running_loop()
        item_id = int(item["id"])
        recurrence = item.get("recurrence_seconds")
        now = self._now_fn()
        if recurrence:
            next_run = now + timedelta(seconds=int(recurrence))
            await loop.run_in_executor(
                None,
                lambda: store.reschedule_recurring(item_id, next_run, last_error=f"submit failed: {error}", now=now),
            )
        else:
            await loop.run_in_executor(
                None,
                lambda: store.update_status(
                    item_id, QUEUE_STATUS_FAILED, last_error=f"submit failed: {error}", now=now
                ),
            )


_scheduler: QueueScheduler | None = None


def start_queue_scheduler(db_url: str) -> QueueScheduler | None:
    """Start the module-level queue scheduler if enabled. Never raises."""
    global _scheduler
    try:
        if not queue_enabled():
            logger.info("Research queue scheduler disabled via AIQ_QUEUE_ENABLED")
            return None
        if _scheduler is not None and _scheduler._task and not _scheduler._task.done():
            return _scheduler
        _scheduler = QueueScheduler(db_url)
        _scheduler.start()
        return _scheduler
    except Exception as exc:  # noqa: BLE001 - startup must never be fatal
        logger.warning("Failed to start research queue scheduler: %s", exc)
        return None


async def stop_queue_scheduler() -> None:
    """Stop the module-level queue scheduler. Call from shutdown handlers."""
    global _scheduler
    if _scheduler is not None:
        try:
            await _scheduler.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to stop research queue scheduler: %s", exc)
        _scheduler = None

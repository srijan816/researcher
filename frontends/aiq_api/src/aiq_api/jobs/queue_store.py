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

"""Durable server-side research queue store.

Persists queued/scheduled/recurring research requests in a ``research_queue``
table so queue state survives backend restarts and works across clients.
Follows the dual SQLite/PostgreSQL pattern used by :mod:`.event_store`:
SQLAlchemy Core with raw-text SQL, sync engines shared through
``EventStore._get_or_create_sync_engine``, and lazily auto-created schema.

Item lifecycle::

    queued -> approved -> running -> completed | failed
       \\-> cancelled        (recurring items: running -> approved again)

Timestamps are stored as naive UTC datetimes so ordering/comparison SQL works
identically on SQLite and PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

QUEUE_STATUS_QUEUED = "queued"
QUEUE_STATUS_APPROVED = "approved"
QUEUE_STATUS_RUNNING = "running"
QUEUE_STATUS_COMPLETED = "completed"
QUEUE_STATUS_FAILED = "failed"
QUEUE_STATUS_CANCELLED = "cancelled"

QUEUE_STATUSES = {
    QUEUE_STATUS_QUEUED,
    QUEUE_STATUS_APPROVED,
    QUEUE_STATUS_RUNNING,
    QUEUE_STATUS_COMPLETED,
    QUEUE_STATUS_FAILED,
    QUEUE_STATUS_CANCELLED,
}

_QUEUE_COLUMNS = (
    "id, owner_auth_type, owner_subject, owner_email, agent_type, input, research_depth, "
    "status, priority, scheduled_for, recurrence_seconds, last_job_id, last_run_at, last_error, "
    "webhook_url, webhook_headers, webhook_secret, created_at, updated_at"
)


def utc_now() -> datetime:
    """Return the current time as a naive UTC datetime (portable across backends)."""
    return datetime.now(UTC).replace(tzinfo=None)


def normalize_queue_datetime(value: datetime | None) -> datetime | None:
    """Convert any aware datetime to naive UTC for stable storage/comparison."""
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


class QueueStore:
    """Synchronous research-queue persistence helper.

    All methods are sync (DB I/O); call from async code via
    ``loop.run_in_executor`` like the other job stores in this package.
    """

    _tables_initialized: set[str] = set()
    _init_lock = threading.Lock()

    def __init__(self, db_url: str):
        from .event_store import EventStore

        self.db_url = db_url
        self._engine = EventStore._get_or_create_sync_engine(db_url)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        with QueueStore._init_lock:
            if self.db_url in QueueStore._tables_initialized:
                return

            from sqlalchemy import Column
            from sqlalchemy import DateTime
            from sqlalchemy import Index
            from sqlalchemy import Integer
            from sqlalchemy import MetaData
            from sqlalchemy import String
            from sqlalchemy import Table
            from sqlalchemy import Text
            from sqlalchemy import inspect
            from sqlalchemy.sql import func

            metadata = MetaData()
            Table(
                "research_queue",
                metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("owner_auth_type", String(64), nullable=False),
                Column("owner_subject", String(256), nullable=False),
                Column("owner_email", String(320), nullable=True),
                Column("agent_type", String(128), nullable=False),
                Column("input", Text, nullable=False),
                Column("research_depth", String(32), nullable=False),
                Column("status", String(16), nullable=False, index=True),
                Column("priority", Integer, nullable=False, server_default="0"),
                Column("scheduled_for", DateTime, nullable=True),
                Column("recurrence_seconds", Integer, nullable=True),
                Column("last_job_id", String(64), nullable=True),
                Column("last_run_at", DateTime, nullable=True),
                Column("last_error", Text, nullable=True),
                Column("webhook_url", Text, nullable=True),
                Column("webhook_headers", Text, nullable=True),
                Column("webhook_secret", Text, nullable=True),
                Column("created_at", DateTime, server_default=func.now()),
                Column("updated_at", DateTime, server_default=func.now()),
                Index("idx_research_queue_status_priority", "status", "priority"),
                Index("idx_research_queue_owner", "owner_auth_type", "owner_subject"),
            )

            inspector = inspect(self._engine)
            if not inspector.has_table("research_queue"):
                metadata.create_all(self._engine)
                logger.info("Created research_queue table in %s", self.db_url[:50])

            QueueStore._tables_initialized.add(self.db_url)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_item(
        self,
        *,
        owner_auth_type: str,
        owner_subject: str,
        owner_email: str | None,
        agent_type: str,
        input_text: str,
        research_depth: str,
        priority: int = 0,
        scheduled_for: datetime | None = None,
        recurrence_seconds: int | None = None,
        auto_approve: bool = True,
        webhook_url: str | None = None,
        webhook_headers: dict[str, str] | None = None,
        webhook_secret: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Insert a new queue item and return it as a dict."""
        from sqlalchemy import text

        now = now or utc_now()
        status = QUEUE_STATUS_APPROVED if auto_approve else QUEUE_STATUS_QUEUED
        params = {
            "owner_auth_type": owner_auth_type,
            "owner_subject": owner_subject,
            "owner_email": owner_email,
            "agent_type": agent_type,
            "input": input_text,
            "research_depth": research_depth,
            "status": status,
            "priority": int(priority),
            "scheduled_for": normalize_queue_datetime(scheduled_for),
            "recurrence_seconds": recurrence_seconds,
            "webhook_url": webhook_url,
            "webhook_headers": json.dumps(webhook_headers) if webhook_headers else None,
            "webhook_secret": webhook_secret,
            "created_at": now,
            "updated_at": now,
        }
        insert_sql = (
            "INSERT INTO research_queue ("
            "owner_auth_type, owner_subject, owner_email, agent_type, input, research_depth, "
            "status, priority, scheduled_for, recurrence_seconds, webhook_url, webhook_headers, "
            "webhook_secret, created_at, updated_at"
            ") VALUES ("
            ":owner_auth_type, :owner_subject, :owner_email, :agent_type, :input, :research_depth, "
            ":status, :priority, :scheduled_for, :recurrence_seconds, :webhook_url, :webhook_headers, "
            ":webhook_secret, :created_at, :updated_at"
            ")"
        )
        with self._engine.connect() as conn:
            if self.db_url.startswith("postgres"):
                result = conn.execute(text(insert_sql + " RETURNING id"), params)
                item_id = result.scalar()
            else:
                result = conn.execute(text(insert_sql), params)
                item_id = result.lastrowid
            conn.commit()
        item = self.get_item(int(item_id))
        if item is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to read back queue item {item_id}")
        return item

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(f"SELECT {_QUEUE_COLUMNS} FROM research_queue WHERE id = :id"),
                    {"id": item_id},
                )
                .mappings()
                .first()
            )
            return self._row_to_dict(row) if row is not None else None

    def list_items(
        self,
        *,
        status: str | None = None,
        owner_auth_type: str | None = None,
        owner_subject: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        from sqlalchemy import text

        clauses = []
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 500))}
        if status:
            clauses.append("status = :status")
            params["status"] = status
        if owner_auth_type is not None and owner_subject is not None:
            clauses.append("owner_auth_type = :owner_auth_type AND owner_subject = :owner_subject")
            params["owner_auth_type"] = owner_auth_type
            params["owner_subject"] = owner_subject
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = text(
            f"SELECT {_QUEUE_COLUMNS} FROM research_queue "
            f"{where}"
            "ORDER BY priority DESC, COALESCE(scheduled_for, created_at) ASC, id ASC "
            "LIMIT :limit"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
            return [self._row_to_dict(row) for row in rows]

    def update_status(
        self,
        item_id: int,
        status: str,
        *,
        expected_statuses: tuple[str, ...] | None = None,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Transition an item to a new status. Returns False if the guard failed."""
        from sqlalchemy import text

        if status not in QUEUE_STATUSES:
            raise ValueError(f"Unknown queue status: {status}")
        now = now or utc_now()
        params: dict[str, Any] = {"id": item_id, "status": status, "updated_at": now}
        guard = ""
        if expected_statuses:
            placeholders = ", ".join(f":expected_{idx}" for idx in range(len(expected_statuses)))
            guard = f" AND status IN ({placeholders})"
            params.update({f"expected_{idx}": value for idx, value in enumerate(expected_statuses)})
        error_clause = ""
        if last_error is not None:
            error_clause = ", last_error = :last_error"
            params["last_error"] = last_error[:4000]
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "UPDATE research_queue SET status = :status, updated_at = :updated_at"
                    f"{error_clause} WHERE id = :id{guard}"
                ),
                params,
            )
            conn.commit()
            return bool(result.rowcount)

    def delete_item(self, item_id: int) -> bool:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            result = conn.execute(text("DELETE FROM research_queue WHERE id = :id"), {"id": item_id})
            conn.commit()
            return bool(result.rowcount)

    # ------------------------------------------------------------------
    # Scheduler helpers
    # ------------------------------------------------------------------

    def count_running(self) -> int:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT COUNT(*) FROM research_queue WHERE status = :status"),
                    {"status": QUEUE_STATUS_RUNNING},
                ).scalar()
                or 0
            )

    def list_running(self) -> list[dict[str, Any]]:
        return self.list_items(status=QUEUE_STATUS_RUNNING, limit=500)

    def claim_next_due(self, now: datetime | None = None) -> dict[str, Any] | None:
        """Atomically claim the highest-priority approved item that is due.

        Marks the item ``running`` (guarded on its previous status so a
        concurrent claimer loses the race cleanly) and returns it, or None
        when nothing is due.
        """
        from sqlalchemy import text

        now = now or utc_now()
        select_sql = text(
            f"SELECT {_QUEUE_COLUMNS} FROM research_queue "
            "WHERE status = :status AND (scheduled_for IS NULL OR scheduled_for <= :now) "
            "ORDER BY priority DESC, COALESCE(scheduled_for, created_at) ASC, id ASC "
            "LIMIT 5"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(select_sql, {"status": QUEUE_STATUS_APPROVED, "now": now}).mappings().all()
        for row in rows:
            item = self._row_to_dict(row)
            claimed = self.update_status(
                int(item["id"]),
                QUEUE_STATUS_RUNNING,
                expected_statuses=(QUEUE_STATUS_APPROVED,),
                now=now,
            )
            if claimed:
                item["status"] = QUEUE_STATUS_RUNNING
                return item
        return None

    def record_submission(self, item_id: int, job_id: str, now: datetime | None = None) -> None:
        """Record the async job spawned for a claimed queue item."""
        from sqlalchemy import text

        now = now or utc_now()
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "UPDATE research_queue SET last_job_id = :job_id, last_run_at = :now, "
                    "last_error = NULL, updated_at = :now WHERE id = :id"
                ),
                {"id": item_id, "job_id": job_id, "now": now},
            )
            conn.commit()

    def reschedule_recurring(
        self,
        item_id: int,
        next_run_at: datetime,
        *,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Re-approve a recurring item for its next run, preserving run history."""
        from sqlalchemy import text

        now = now or utc_now()
        params: dict[str, Any] = {
            "id": item_id,
            "status": QUEUE_STATUS_APPROVED,
            "scheduled_for": normalize_queue_datetime(next_run_at),
            "updated_at": now,
            "last_error": last_error[:4000] if last_error else None,
        }
        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    "UPDATE research_queue SET status = :status, scheduled_for = :scheduled_for, "
                    "last_error = :last_error, updated_at = :updated_at WHERE id = :id"
                ),
                params,
            )
            conn.commit()
            return bool(result.rowcount)

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        item = dict(row)
        headers = item.get("webhook_headers")
        if isinstance(headers, str) and headers.strip():
            try:
                item["webhook_headers"] = json.loads(headers)
            except (TypeError, ValueError):
                item["webhook_headers"] = None
        for key in ("scheduled_for", "last_run_at", "created_at", "updated_at"):
            value = item.get(key)
            if isinstance(value, str):
                try:
                    item[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
        return item

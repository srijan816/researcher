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

"""Tests for the durable server-side research queue store and scheduler."""

from datetime import datetime
from datetime import timedelta

import pytest

from aiq_api.jobs.queue_scheduler import JOB_STATUS_MISSING
from aiq_api.jobs.queue_scheduler import QueueScheduler
from aiq_api.jobs.queue_store import QUEUE_STATUS_APPROVED
from aiq_api.jobs.queue_store import QUEUE_STATUS_CANCELLED
from aiq_api.jobs.queue_store import QUEUE_STATUS_COMPLETED
from aiq_api.jobs.queue_store import QUEUE_STATUS_FAILED
from aiq_api.jobs.queue_store import QUEUE_STATUS_QUEUED
from aiq_api.jobs.queue_store import QUEUE_STATUS_RUNNING
from aiq_api.jobs.queue_store import QueueStore

NOW = datetime(2026, 6, 10, 12, 0, 0)


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/queue.db"


@pytest.fixture
def store(db_url):
    return QueueStore(db_url)


def _add(store: QueueStore, **overrides):
    params = {
        "owner_auth_type": "local",
        "owner_subject": "srijan",
        "owner_email": "srijan@example.com",
        "agent_type": "deep_researcher",
        "input_text": "research something",
        "research_depth": "deeper",
        "now": NOW,
    }
    params.update(overrides)
    return store.add_item(**params)


class TestQueueStoreCrud:
    def test_add_item_auto_approve_default(self, store):
        item = _add(store)
        assert item["status"] == QUEUE_STATUS_APPROVED
        assert item["agent_type"] == "deep_researcher"
        assert item["input"] == "research something"
        assert item["priority"] == 0
        assert item["last_job_id"] is None

    def test_add_item_without_auto_approve_is_queued(self, store):
        item = _add(store, auto_approve=False)
        assert item["status"] == QUEUE_STATUS_QUEUED

    def test_webhook_fields_roundtrip(self, store):
        item = _add(
            store,
            webhook_url="https://example.com/hook",
            webhook_headers={"X-Client": "abc"},
            webhook_secret="super-secret-value",  # pragma: allowlist secret
        )
        fetched = store.get_item(item["id"])
        assert fetched["webhook_url"] == "https://example.com/hook"
        assert fetched["webhook_headers"] == {"X-Client": "abc"}
        assert fetched["webhook_secret"] == "super-secret-value"  # pragma: allowlist secret

    def test_get_missing_item_returns_none(self, store):
        assert store.get_item(99999) is None

    def test_list_items_filters_by_status_and_owner(self, store):
        _add(store, auto_approve=False)
        _add(store)
        _add(store, owner_subject="other-user")

        queued = store.list_items(status=QUEUE_STATUS_QUEUED)
        assert len(queued) == 1

        own = store.list_items(owner_auth_type="local", owner_subject="srijan")
        assert len(own) == 2
        assert all(entry["owner_subject"] == "srijan" for entry in own)

    def test_update_status_with_guard(self, store):
        item = _add(store, auto_approve=False)
        assert store.update_status(item["id"], QUEUE_STATUS_CANCELLED, expected_statuses=(QUEUE_STATUS_QUEUED,))
        # Guard fails once the item is no longer queued.
        assert not store.update_status(item["id"], QUEUE_STATUS_APPROVED, expected_statuses=(QUEUE_STATUS_QUEUED,))
        assert store.get_item(item["id"])["status"] == QUEUE_STATUS_CANCELLED

    def test_update_status_records_error(self, store):
        item = _add(store)
        store.update_status(item["id"], QUEUE_STATUS_FAILED, last_error="boom")
        assert store.get_item(item["id"])["last_error"] == "boom"

    def test_delete_item(self, store):
        item = _add(store)
        assert store.delete_item(item["id"])
        assert store.get_item(item["id"]) is None
        assert not store.delete_item(item["id"])


class TestDueItemSelection:
    def test_claim_prefers_higher_priority(self, store):
        low = _add(store, priority=0)
        high = _add(store, priority=10)

        claimed = store.claim_next_due(now=NOW)
        assert claimed["id"] == high["id"]
        assert claimed["status"] == QUEUE_STATUS_RUNNING

        second = store.claim_next_due(now=NOW)
        assert second["id"] == low["id"]

    def test_future_scheduled_items_are_not_due(self, store):
        _add(store, priority=50, scheduled_for=NOW + timedelta(hours=2))
        due_now = _add(store, priority=0)

        claimed = store.claim_next_due(now=NOW)
        assert claimed["id"] == due_now["id"]
        assert store.claim_next_due(now=NOW) is None

        # Once the clock passes the schedule, the item becomes claimable.
        claimed_later = store.claim_next_due(now=NOW + timedelta(hours=3))
        assert claimed_later is not None

    def test_queued_items_are_not_claimed(self, store):
        _add(store, auto_approve=False)
        assert store.claim_next_due(now=NOW) is None

    def test_count_running(self, store):
        assert store.count_running() == 0
        _add(store)
        store.claim_next_due(now=NOW)
        assert store.count_running() == 1


class TestQueueScheduler:
    def _scheduler(self, db_url, *, submitted=None, job_statuses=None, max_concurrent=1, now=NOW):
        submitted = submitted if submitted is not None else []
        job_statuses = job_statuses or {}

        async def fake_submit(item):
            submitted.append(item)
            return f"job-{item['id']}"

        return QueueScheduler(
            db_url,
            submit_fn=fake_submit,
            job_status_fn=lambda job_id: job_statuses.get(job_id),
            now_fn=lambda: now,
            poll_seconds=0.01,
            max_concurrent=max_concurrent,
        )

    async def test_dispatches_due_item_and_records_job(self, db_url, store):
        item = _add(store)
        submitted = []
        scheduler = self._scheduler(db_url, submitted=submitted)

        await scheduler.run_once()

        assert len(submitted) == 1
        refreshed = store.get_item(item["id"])
        assert refreshed["status"] == QUEUE_STATUS_RUNNING
        assert refreshed["last_job_id"] == f"job-{item['id']}"
        assert refreshed["last_run_at"] is not None

    async def test_respects_max_concurrent(self, db_url, store):
        _add(store)
        _add(store)
        submitted = []
        scheduler = self._scheduler(db_url, submitted=submitted, max_concurrent=1)

        await scheduler.run_once()
        assert len(submitted) == 1

        # Second cycle: first job still running -> no new submission.
        await scheduler.run_once()
        assert len(submitted) == 1
        assert store.count_running() == 1

    async def test_reconcile_marks_completed_and_failed(self, db_url, store):
        ok = _add(store)
        bad = _add(store)
        submitted = []
        scheduler = self._scheduler(db_url, submitted=submitted, max_concurrent=2)
        await scheduler.run_once()
        assert len(submitted) == 2

        scheduler = self._scheduler(
            db_url,
            job_statuses={f"job-{ok['id']}": "success", f"job-{bad['id']}": "failure"},
        )
        await scheduler.run_once()

        assert store.get_item(ok["id"])["status"] == QUEUE_STATUS_COMPLETED
        assert store.get_item(bad["id"])["status"] == QUEUE_STATUS_FAILED
        assert "failure" in store.get_item(bad["id"])["last_error"]

    async def test_reconcile_missing_job_marks_failed(self, db_url, store):
        item = _add(store)
        scheduler = self._scheduler(db_url)
        await scheduler.run_once()

        scheduler = self._scheduler(db_url, job_statuses={f"job-{item['id']}": JOB_STATUS_MISSING})
        await scheduler.run_once()
        refreshed = store.get_item(item["id"])
        assert refreshed["status"] == QUEUE_STATUS_FAILED
        assert "missing" in refreshed["last_error"]

    async def test_reconcile_leaves_unknown_status_running(self, db_url, store):
        item = _add(store)
        scheduler = self._scheduler(db_url)
        await scheduler.run_once()

        # Probe returns None (undeterminable): item stays running.
        scheduler = self._scheduler(db_url, job_statuses={})
        await scheduler.run_once()
        assert store.get_item(item["id"])["status"] == QUEUE_STATUS_RUNNING

    async def test_recurring_item_reenqueues_after_terminal(self, db_url, store):
        item = _add(store, recurrence_seconds=3600)
        scheduler = self._scheduler(db_url)
        await scheduler.run_once()
        job_id = store.get_item(item["id"])["last_job_id"]
        first_run_at = store.get_item(item["id"])["last_run_at"]

        later = NOW + timedelta(minutes=10)
        scheduler = self._scheduler(db_url, job_statuses={job_id: "success"}, now=later)
        await scheduler.run_once()

        refreshed = store.get_item(item["id"])
        assert refreshed["status"] == QUEUE_STATUS_APPROVED
        assert refreshed["scheduled_for"] == later + timedelta(seconds=3600)
        # Run history is preserved on re-enqueue.
        assert refreshed["last_job_id"] == job_id
        assert refreshed["last_run_at"] == first_run_at

        # Not due until the recurrence interval elapses.
        assert store.claim_next_due(now=later) is None
        assert store.claim_next_due(now=later + timedelta(seconds=3601)) is not None

    async def test_startup_reconcile_repairs_stuck_running_without_job(self, db_url, store):
        item = _add(store)
        # Simulate a crash between claim and submit: running with no job id.
        store.update_status(item["id"], QUEUE_STATUS_RUNNING)
        submitted = []
        scheduler = self._scheduler(db_url, submitted=submitted)

        await scheduler.run_once()

        refreshed = store.get_item(item["id"])
        # Re-approved during reconcile, then immediately claimed and dispatched.
        assert refreshed["status"] == QUEUE_STATUS_RUNNING
        assert refreshed["last_job_id"] == f"job-{item['id']}"
        assert len(submitted) == 1

    async def test_submit_failure_marks_failed(self, db_url, store):
        item = _add(store)

        async def failing_submit(_item):
            raise RuntimeError("dask is down")

        scheduler = QueueScheduler(
            db_url,
            submit_fn=failing_submit,
            job_status_fn=lambda job_id: None,
            now_fn=lambda: NOW,
            max_concurrent=1,
        )
        await scheduler.run_once()

        refreshed = store.get_item(item["id"])
        assert refreshed["status"] == QUEUE_STATUS_FAILED
        assert "dask is down" in refreshed["last_error"]

    async def test_submit_failure_recurring_reschedules(self, db_url, store):
        item = _add(store, recurrence_seconds=600)

        async def failing_submit(_item):
            raise RuntimeError("dask is down")

        scheduler = QueueScheduler(
            db_url,
            submit_fn=failing_submit,
            job_status_fn=lambda job_id: None,
            now_fn=lambda: NOW,
            max_concurrent=1,
        )
        await scheduler.run_once()

        refreshed = store.get_item(item["id"])
        assert refreshed["status"] == QUEUE_STATUS_APPROVED
        assert refreshed["scheduled_for"] == NOW + timedelta(seconds=600)
        assert "dask is down" in refreshed["last_error"]

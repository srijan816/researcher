# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave 3 W3.4 — tests for the in-process asyncio job executor.

Covers the public surface of :mod:`aiq_api.jobs.asyncio_executor`:

* Env-flag parsing (``use_inprocess_executor``)
* Submit/run/done lifecycle for a single job
* Exception capture (worker raises → record.status == "failed")
* Concurrency cap (semaphore gates in-flight jobs)
* Cancellation of a running job (task.cancel() + cancel_event)
* Cancellation of a queued job (before the semaphore slot opens)
* Requeue replaces an existing record
* ``snapshot()`` reports the right counts
* Singleton is per-process; ``reset_instance()`` clears state for tests
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from aiq_api.jobs import asyncio_executor
from aiq_api.jobs.asyncio_executor import InProcessJobExecutor
from aiq_api.jobs.asyncio_executor import JobRecord
from aiq_api.jobs.asyncio_executor import use_inprocess_executor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_noop(*_args: Any, **_kwargs: Any) -> str:
    return "ok"


async def _async_sleep(seconds: float, *args: Any, **kwargs: Any) -> str:
    await asyncio.sleep(seconds)
    return f"slept_{seconds}"


async def _async_raise(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("boom from worker")


@pytest.fixture(autouse=True)
def _reset_executor_singleton() -> None:
    """Each test starts from a clean executor state.

    The executor is a process-global singleton; if a previous test left
    records behind, ``submit()`` will refuse the next ``job_id`` with
    ``ValueError``. ``reset_instance()`` clears it.
    """
    InProcessJobExecutor.reset_instance()
    # Make sure we start with a known-off state, even if the test runner
    # has an exported AIQ_USE_INPROCESS_EXECUTOR in its env.
    os.environ.pop("AIQ_USE_INPROCESS_EXECUTOR", None)
    os.environ.pop("AIQ_ASYNC_JOB_MAX_CONCURRENT", None)
    yield
    InProcessJobExecutor.reset_instance()
    os.environ.pop("AIQ_USE_INPROCESS_EXECUTOR", None)
    os.environ.pop("AIQ_ASYNC_JOB_MAX_CONCURRENT", None)


# ---------------------------------------------------------------------------
# Env flag
# ---------------------------------------------------------------------------


class TestUseInprocessExecutor:
    def test_default_is_false(self) -> None:
        assert use_inprocess_executor() is False

    @pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_truthy_values(self, truthy: str) -> None:
        os.environ["AIQ_USE_INPROCESS_EXECUTOR"] = truthy
        assert use_inprocess_executor() is True

    @pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "", "maybe"])
    def test_falsy_values(self, falsy: str) -> None:
        os.environ["AIQ_USE_INPROCESS_EXECUTOR"] = falsy
        assert use_inprocess_executor() is False

    def test_unset_is_false(self) -> None:
        os.environ.pop("AIQ_USE_INPROCESS_EXECUTOR", None)
        assert use_inprocess_executor() is False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_instance_returns_same_object(self) -> None:
        a = InProcessJobExecutor.instance()
        b = InProcessJobExecutor.instance()
        assert a is b

    def test_reset_clears_state(self) -> None:
        a = InProcessJobExecutor.instance()
        InProcessJobExecutor.reset_instance()
        b = InProcessJobExecutor.instance()
        assert a is not b

    def test_max_concurrent_from_env(self) -> None:
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "5"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()
        assert ex._max_concurrent == 5
        assert ex._semaphore._value == 5  # type: ignore[attr-defined]

    def test_max_concurrent_default_when_env_unset(self) -> None:
        # The default is 2 — matches the historical DASK_NWORKERS=2 ceiling.
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()
        assert ex._max_concurrent == 2


# ---------------------------------------------------------------------------
# Submit / run / done
# ---------------------------------------------------------------------------


class TestSubmitRunDone:
    async def test_submit_runs_and_records_success(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.submit(job_id="j1", job_fn=_async_noop, job_args=[])

        rec = ex.get_record("j1")
        assert rec is not None
        # Wait for the task to reach a terminal state.
        await rec.task
        assert rec.status == "success"
        assert rec.finished_at is not None
        assert rec.started_at is not None
        assert rec.exception is None

    async def test_submit_returns_job_id(self) -> None:
        ex = InProcessJobExecutor.instance()
        out = await ex.submit(job_id="j1", job_fn=_async_noop, job_args=[])
        assert out == "j1"

    async def test_submit_duplicate_id_raises(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.submit(job_id="j1", job_fn=_async_noop, job_args=[])
        with pytest.raises(ValueError, match="already submitted"):
            await ex.submit(job_id="j1", job_fn=_async_noop, job_args=[])

    async def test_passes_args_to_worker(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.submit(
            job_id="j1", job_fn=_async_sleep, job_args=[0.01]
        )
        rec = ex.get_record("j1")
        assert await rec.task == "slept_0.01"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestFailurePath:
    async def test_exception_recorded_as_failed(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.submit(job_id="j1", job_fn=_async_raise, job_args=[])
        rec = ex.get_record("j1")
        # Await the task; the exception is re-raised so the test fails
        # if the executor is silently swallowing.
        with pytest.raises(RuntimeError, match="boom from worker"):
            await rec.task
        assert rec.status == "failed"
        assert rec.exception is not None
        assert "boom from worker" in str(rec.exception)
        assert rec.finished_at is not None

    async def test_slot_released_after_failure(self) -> None:
        # If the slot wasn't released, subsequent submits would queue.
        # Set the cap via env so the singleton picks it up.
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "1"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()
        await ex.submit(job_id="j1", job_fn=_async_raise, job_args=[])
        rec = ex.get_record("j1")
        with pytest.raises(RuntimeError):
            await rec.task
        # The semaphore is back to 1.
        assert ex._semaphore._value == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


class TestConcurrencyCap:
    async def test_semaphore_caps_in_flight_jobs(self) -> None:
        # max_concurrent=1 → only one job runs at a time; the second is
        # queued until the first finishes. Cap is set via env so the
        # singleton picks it up; passing max_concurrent to .instance()
        # is a no-op because the classmethod only invokes __init__ when
        # the singleton is None.
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "1"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()

        started = asyncio.Event()
        proceed = asyncio.Event()

        async def gated_worker(*args: Any) -> str:
            started.set()
            await proceed.wait()
            return "done"

        await ex.submit(job_id="j1", job_fn=gated_worker, job_args=[])
        # Wait until the first job is in `running`.
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert ex.is_running("j1")

        # The second submit is queued (semaphore full).
        await ex.submit(job_id="j2", job_fn=_async_noop, job_args=[])
        rec2 = ex.get_record("j2")
        # Yield so j2 has a chance to start and reach the semaphore
        # acquire. Without this, the task may not have run at all and
        # rec2.status is still the default "queued" for the wrong reason.
        await asyncio.sleep(0)
        assert rec2.status == "queued"
        assert not ex.is_running("j2")

        # Let j1 finish, j2 should pick up the slot and run.
        proceed.set()
        await asyncio.gather(ex.get_record("j1").task, rec2.task)
        assert ex.get_record("j2").status == "success"

    async def test_cap_from_env(self) -> None:
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "1"
        ex = InProcessJobExecutor.instance()
        assert ex._max_concurrent == 1

    async def test_many_jobs_eventually_all_complete(self) -> None:
        # 5 jobs through a cap-2 executor — all must finish successfully.
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "2"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()
        ids = [f"j{i}" for i in range(5)]
        for jid in ids:
            await ex.submit(job_id=jid, job_fn=_async_sleep, job_args=[0.01])
        # Wait for all to finish.
        await asyncio.gather(*(ex.get_record(jid).task for jid in ids))
        for jid in ids:
            assert ex.get_record(jid).status == "success"

        snap = ex.snapshot()
        assert snap["completed_total"] == 5
        assert snap["failed_total"] == 0
        assert snap["active"] == 0
        assert snap["queued"] == 0


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_cancel_running_job(self) -> None:
        ex = InProcessJobExecutor.instance()
        started = asyncio.Event()
        proceed = asyncio.Event()

        async def long_running(*args: Any) -> str:
            started.set()
            await proceed.wait()
            return "done"

        await ex.submit(job_id="j1", job_fn=long_running, job_args=[])
        await asyncio.wait_for(started.wait(), timeout=1.0)

        cancelled = await ex.cancel("j1", timeout=1.0)
        assert cancelled is True
        rec = ex.get_record("j1")
        assert rec.status == "cancelled"
        assert rec.cancel_event.is_set()

    async def test_cancel_unknown_job_returns_false(self) -> None:
        ex = InProcessJobExecutor.instance()
        assert await ex.cancel("does-not-exist") is False

    async def test_cancel_queued_job(self) -> None:
        # max_concurrent=1, two submits: the first blocks on a held
        # semaphore, the second is queued. Cancelling the queued one
        # should release its place without running the worker.
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "1"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()

        started = asyncio.Event()
        proceed = asyncio.Event()

        async def blocker(*args: Any) -> str:
            started.set()
            await proceed.wait()
            return "done"

        await ex.submit(job_id="blocker", job_fn=blocker, job_args=[])
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await ex.submit(job_id="queued", job_fn=_async_noop, job_args=[])
        # Confirm the second is actually queued.
        assert ex.is_queued("queued")

        # Yield once to let the queued task actually start and reach
        # `await self._semaphore.acquire()` so its exception handlers are
        # installed before cancel() is called. Without this, task.cancel()
        # can transition the task to done+cancelled before its body has
        # run at all, skipping the record-status update.
        await asyncio.sleep(0)

        cancelled = await ex.cancel("queued", timeout=1.0)
        assert cancelled is True
        # Wait for the task to finish propagating the cancellation into
        # the record's status. Awaiting raises CancelledError; expect that
        # and assert on the record directly.
        rec = ex.get_record("queued")
        with pytest.raises(asyncio.CancelledError):
            await rec.task
        assert rec.status == "cancelled"

        # Cleanup: let the blocker finish so the test teardown is clean.
        proceed.set()
        await ex.get_record("blocker").task

    async def test_cancel_after_done_returns_done_state(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.submit(job_id="j1", job_fn=_async_noop, job_args=[])
        await ex.get_record("j1").task
        # Already terminal; cancel() should return True (the task is done).
        assert await ex.cancel("j1") is True


# ---------------------------------------------------------------------------
# Requeue (resume path)
# ---------------------------------------------------------------------------


class TestRequeue:
    async def test_requeue_replaces_existing_record(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.submit(job_id="j1", job_fn=_async_noop, job_args=[])
        await ex.get_record("j1").task
        old = ex.get_record("j1")
        assert old.status == "success"

        await ex.requeue(job_id="j1", job_fn=_async_sleep, job_args=[0.01])
        new = ex.get_record("j1")
        # The record was replaced; the new task reflects the re-submit.
        assert new is not old
        assert new.status in ("queued", "running")
        await new.task
        assert new.status == "success"

    async def test_requeue_with_no_prior_record_just_submits(self) -> None:
        ex = InProcessJobExecutor.instance()
        await ex.requeue(job_id="fresh", job_fn=_async_noop, job_args=[])
        await ex.get_record("fresh").task
        assert ex.get_record("fresh").status == "success"

    async def test_requeue_cancels_in_flight_old_task(self) -> None:
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "1"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()
        started = asyncio.Event()
        proceed = asyncio.Event()

        async def blocker(*args: Any) -> str:
            started.set()
            await proceed.wait()
            return "done"

        await ex.submit(job_id="j1", job_fn=blocker, job_args=[])
        await asyncio.wait_for(started.wait(), timeout=1.0)

        # Requeue should cancel the in-flight task and start a fresh one.
        await ex.requeue(job_id="j1", job_fn=_async_noop, job_args=[])
        rec = ex.get_record("j1")
        await rec.task
        assert rec.status == "success"

        # The original blocker should have been cancelled.
        # (We don't assert on its record — requeue replaced it.)


# ---------------------------------------------------------------------------
# Snapshot / shutdown
# ---------------------------------------------------------------------------


class TestSnapshotAndShutdown:
    async def test_snapshot_counts(self) -> None:
        os.environ["AIQ_ASYNC_JOB_MAX_CONCURRENT"] = "2"
        InProcessJobExecutor.reset_instance()
        ex = InProcessJobExecutor.instance()

        proceed = asyncio.Event()
        started_count = 0

        async def gated(*args: Any) -> str:
            nonlocal started_count
            started_count += 1
            await proceed.wait()
            return "done"

        await ex.submit(job_id="running1", job_fn=gated, job_args=[])
        # Wait for running1 to actually be running (gated inside the worker).
        while started_count < 1:
            await asyncio.sleep(0.01)

        await ex.submit(job_id="running2", job_fn=gated, job_args=[])
        # Wait for running2 to also be running.
        while started_count < 2:
            await asyncio.sleep(0.01)

        # Submit a third that will queue behind the cap of 2.
        await ex.submit(job_id="queued1", job_fn=_async_noop, job_args=[])
        # Yield a tick so the queued task is observable in the dict.
        await asyncio.sleep(0.01)

        snap = ex.snapshot()
        assert snap["max_concurrent"] == 2
        assert snap["active"] == 2
        assert snap["queued"] == 1
        assert snap["tracked_jobs"] == 3

        # Cleanup.
        proceed.set()
        await asyncio.gather(
            ex.get_record("running1").task,
            ex.get_record("running2").task,
            ex.get_record("queued1").task,
        )

    async def test_shutdown_cancels_in_flight(self) -> None:
        ex = InProcessJobExecutor.instance()
        started = asyncio.Event()
        proceed = asyncio.Event()

        async def long(*args: Any) -> str:
            started.set()
            await proceed.wait()
            return "done"

        await ex.submit(job_id="j1", job_fn=long, job_args=[])
        await asyncio.wait_for(started.wait(), timeout=1.0)
        ex.shutdown()
        # Records cleared.
        assert ex.get_record("j1") is None


# ---------------------------------------------------------------------------
# JobRecord shape
# ---------------------------------------------------------------------------


class TestJobRecord:
    def test_default_status_is_queued(self) -> None:
        r = JobRecord(job_id="j1", submitted_at=0.0)
        assert r.status == "queued"
        assert r.started_at is None
        assert r.finished_at is None
        assert r.exception is None
        assert r.task is None
        assert not r.cancel_event.is_set()

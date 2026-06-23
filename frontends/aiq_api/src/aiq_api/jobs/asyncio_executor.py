# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave 3 W3.4 — in-process asyncio job executor.

Replaces the Dask scheduler+worker pair for *async deep-research jobs* with
a singleton that runs the worker function (``run_agent_job``) on the
agent's existing uvicorn event loop. Each submitted job becomes a single
``asyncio.Task``; concurrency is bounded by an ``asyncio.Semaphore`` so the
total in-flight deep-research jobs never exceeds the configured cap.

Why this exists
---------------

The Dask path was introduced when the chat-researcher was first wired to
escalate turns to deep research. The deep-research agent itself is pure
asyncio (LangGraph async nodes + ``asyncio.gather`` in the ToolNode), so
shipping work out to a Dask worker process was added complexity for no
clear throughput win on an I/O-bound workload (LLM HTTP, websurfx HTTP,
postgres).

The executor reuses everything we already had:

* Status persistence — still goes through NAT's ``JobStore`` (postgres or
  sqlite). The Dask ``scheduler_address`` is irrelevant to status writes,
  which are pure SQLAlchemy.
* Cancellation polling — ``CancellationMonitor`` already polls the
  ``JobStore`` for ``JobStatus.INTERRUPTED``. We pass an empty scheduler
  address; the monitor still works.
* ``EventStore`` — the runner already writes ``job.cancelled``,
  ``job.error``, ``job.metrics`` events to the same store. The executor
  adds nothing; it just owns the lifecycle of the task that runs the
  runner.

What this gives us
------------------

* Lower baseline RSS — no ``dask-scheduler`` (~80 MB) and no
  ``dask-worker`` (~200 MB) subprocesses per agent container.
* No scheduler-to-worker TCP socket, no job-submission serialization
  across that boundary.
* Cancellation via ``task.cancel()`` propagates immediately to whatever
  the agent is awaiting, instead of having to wait for the next 1-second
  poll.
* Lifecycle visibility — the executor logs ``submit / start / done /
  cancelled / failed`` transitions in the same structured style as the
  W2.1 ``aiq.metrics`` instrumentation.

What this does NOT do
---------------------

* The executor does **not** change per-tier parallel-researcher caps
  (``research_depth.py``: shallow=1, medium=3, deeper=4, deep=3). Those
  govern how many parallel *researcher tasks* a single deep-research job
  spawns. This executor only caps how many deep-research jobs are
  in-flight at once.
* The executor does **not** replace NAT's ``JobStore``. The ``JobStore``
  still owns the durable job metadata, status rows, and EventStore
  history. We only stop using its ``submit_job`` and ``dask_client.submit``
  methods.
* The executor does **not** scale across processes. The default
  ``AIQ_ASYNC_JOB_MAX_CONCURRENT=2`` matches the current
  ``DASK_NWORKERS=2`` ceiling; if more parallelism is needed later the
  right move is more uvicorn workers or a dedicated job-loop thread,
  not bringing Dask back.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: Environment variable that opts into the in-process executor. When unset
#: or false, the existing Dask submission path is used unchanged.
_USE_INPROCESS_ENV = "AIQ_USE_INPROCESS_EXECUTOR"

#: Concurrency cap for the in-process executor. Defaults to 2 to match
#: the historical ``DASK_NWORKERS=2`` ceiling — we are *not* unlocking
#: more parallelism in this change.
_MAX_CONCURRENT_ENV = "AIQ_ASYNC_JOB_MAX_CONCURRENT"
_DEFAULT_MAX_CONCURRENT = 2


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def use_inprocess_executor() -> bool:
    """Return True if the in-process executor is enabled for this process."""
    return _env_flag(_USE_INPROCESS_ENV, default=False)


# ---------------------------------------------------------------------------
# JobRecord — per-job state held in-memory
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    """Mutable per-job state owned by the executor.

    Attributes:
        job_id: Stable identifier (matches the ``JobStore.job_id``).
        task: The asyncio.Task running the worker coroutine, once started.
        submitted_at: ``time.monotonic()`` at submit time.
        started_at: ``time.monotonic()`` when the task transitioned to
            running. ``None`` while still queued behind the semaphore.
        finished_at: ``time.monotonic()`` on terminal transition.
        status: One of ``queued``, ``running``, ``success``, ``failed``,
            ``cancelled``.
        exception: Captured exception when ``status == failed``; ``None``
            otherwise.
        cancel_event: Set by :meth:`InProcessJobExecutor.cancel` to
            request cooperative cancellation. ``run_agent_job``'s
            ``CancellationMonitor`` polls a separate job-store flag for
            user-initiated cancellation; this event is a backstop for
            executor-initiated cancellation.
    """

    job_id: str
    submitted_at: float
    task: asyncio.Task[Any] | None = None
    started_at: float | None = None
    finished_at: float | None = None
    status: str = "queued"
    exception: BaseException | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


# ---------------------------------------------------------------------------
# InProcessJobExecutor
# ---------------------------------------------------------------------------


class InProcessJobExecutor:
    """Singleton executor that runs async jobs on the current event loop.

    Lifetime: the executor is process-global. The asyncio loop it runs on
    is whichever loop ``submit()`` is called from — typically the agent's
    uvicorn loop. There is no separate scheduler process; the executor
    itself is just an in-memory dict of ``JobRecord`` and a semaphore.

    Concurrency: capped by ``asyncio.Semaphore(max_concurrent)``. The
    default ``max_concurrent=2`` matches the historical
    ``DASK_NWORKERS=2`` ceiling; raise ``AIQ_ASYNC_JOB_MAX_CONCURRENT`` to
    change it. A job that arrives when the semaphore is full is *queued*,
    not rejected — ``JobRecord.status == "queued"`` until it gets a slot.

    Cancellation: :meth:`cancel` sets the per-job ``cancel_event`` *and*
    calls ``task.cancel()`` on the asyncio task. The worker coroutine
    sees the cancel via the asyncio cancellation protocol at its next
    ``await`` point, and can also poll ``cancel_event`` for cooperative
    cleanup.

    Thread safety: the executor is designed to be used from a single
    event loop. The semaphore and the dict are not locked; concurrent
    ``submit()`` calls from different threads would race. The agent
    process uses one loop per uvicorn worker, so this matches the
    existing call pattern.
    """

    _instance: "InProcessJobExecutor | None" = None

    def __init__(self, max_concurrent: int | None = None) -> None:
        self._max_concurrent = max_concurrent or _env_int(
            _MAX_CONCURRENT_ENV, _DEFAULT_MAX_CONCURRENT
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._records: dict[str, JobRecord] = {}
        self._counter_lock = asyncio.Lock()  # for serializing record mutation
        self._completed_total = 0
        self._failed_total = 0
        self._cancelled_total = 0

    @classmethod
    def instance(cls) -> "InProcessJobExecutor":
        """Return the process-global executor, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (tests only)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def submit(
        self,
        *,
        job_id: str,
        job_fn: Callable[..., Awaitable[Any]],
        job_args: list[Any],
    ) -> str:
        """Submit an async job. Returns the ``job_id`` once accepted.

        The call is non-blocking: it returns as soon as the
        ``asyncio.Task`` is created. If the semaphore is saturated the
        task sits in ``queued`` state until a slot frees up.

        Args:
            job_id: Stable identifier. Must be unique within the
                executor's process lifetime — repeated submits with the
                same ID raise :class:`ValueError`.
            job_fn: An async callable. The executor will call
                ``await job_fn(*job_args)`` once the job gets a slot.
            job_args: Positional arguments to pass to ``job_fn``.

        Returns:
            The ``job_id``, mirrored back for caller convenience.

        Raises:
            ValueError: If ``job_id`` is already in the executor.
        """
        if job_id in self._records:
            raise ValueError(f"job_id {job_id!r} already submitted to executor")

        record = JobRecord(job_id=job_id, submitted_at=time.monotonic())
        self._records[job_id] = record

        # Wrap job_fn so we can observe lifecycle transitions and store
        # the result/exception on the record.
        async def _runner() -> Any:
            try:
                # Acquire a slot; this awaits if the semaphore is full,
                # which is the "queued" state.
                await self._semaphore.acquire()
            except asyncio.CancelledError:
                record.status = "cancelled"
                record.finished_at = time.monotonic()
                self._cancelled_total += 1
                raise
            try:
                record.status = "running"
                record.started_at = time.monotonic()
                logger.info(
                    "aiq.jobs.lifecycle job_id=%s status=start "
                    "queue_wait=%.2fs executor_max_concurrent=%d",
                    job_id,
                    (record.started_at - record.submitted_at),
                    self._max_concurrent,
                )
                result = await job_fn(*job_args)
                record.status = "success"
                record.finished_at = time.monotonic()
                self._completed_total += 1
                logger.info(
                    "aiq.jobs.lifecycle job_id=%s status=success "
                    "ran=%.2fs",
                    job_id,
                    (record.finished_at - (record.started_at or record.finished_at)),
                )
                return result
            except asyncio.CancelledError:
                record.status = "cancelled"
                record.finished_at = time.monotonic()
                self._cancelled_total += 1
                logger.info(
                    "aiq.jobs.lifecycle job_id=%s status=cancelled ran=%.2fs",
                    job_id,
                    (record.finished_at - (record.started_at or record.finished_at)),
                )
                raise
            except Exception as exc:  # noqa: BLE001 - capture all, re-raise
                record.status = "failed"
                record.exception = exc
                record.finished_at = time.monotonic()
                self._failed_total += 1
                logger.exception(
                    "aiq.jobs.lifecycle job_id=%s status=failed "
                    "ran=%.2fs error_type=%s",
                    job_id,
                    (record.finished_at - (record.started_at or record.finished_at)),
                    type(exc).__name__,
                )
                raise
            finally:
                self._semaphore.release()

        record.task = asyncio.create_task(_runner(), name=f"inproc-job-{job_id}")
        logger.info(
            "aiq.jobs.lifecycle job_id=%s status=submit "
            "executor_max_concurrent=%d",
            job_id,
            self._max_concurrent,
        )
        return job_id

    async def cancel(self, job_id: str, *, timeout: float = 1.0) -> bool:
        """Request cancellation of a queued or running job.

        Sets the per-job :attr:`JobRecord.cancel_event` *and* calls
        ``task.cancel()`` on the asyncio task. Returns ``True`` if the
        task transitioned to a terminal state within ``timeout`` seconds.

        For running jobs, ``task.cancel()`` raises ``CancelledError`` at
        the task's next ``await`` point — the deep-research agent
        awaits many things, so this is effectively immediate. The
        runner's existing ``except asyncio.CancelledError`` block writes
        the ``job.cancelled`` event to the EventStore, so the chat UI
        sees the cancellation exactly as it did on the Dask path.

        For queued jobs, the task hasn't started yet; cancellation sets
        the event and cancels the task, which prevents the semaphore
        acquire from completing.
        """
        record = self._records.get(job_id)
        if record is None:
            return False
        record.cancel_event.set()
        task = record.task
        if task is None or task.done():
            return task.done() if task else False
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            return task.done()
        return task.done()

    async def requeue(
        self,
        *,
        job_id: str,
        job_fn: Callable[..., Awaitable[Any]],
        job_args: list[Any],
    ) -> str:
        """Replace any existing record for ``job_id`` with a fresh task.

        Used by the resume path. The Dask path's resume uses
        ``dask_client.submit(...)`` with a new key, so behavior is
        identical: the same job_id runs a new task.
        """
        async with self._counter_lock:
            old = self._records.pop(job_id, None)
        if old is not None and old.task is not None and not old.task.done():
            old.task.cancel()
            # Don't await — the caller is replacing the job and the old
            # task will clean up via its own exception handlers.
        return await self.submit(job_id=job_id, job_fn=job_fn, job_args=job_args)

    def get_record(self, job_id: str) -> JobRecord | None:
        """Return the record for ``job_id`` (read-only access)."""
        return self._records.get(job_id)

    def is_running(self, job_id: str) -> bool:
        record = self._records.get(job_id)
        return bool(record and record.status == "running")

    def is_queued(self, job_id: str) -> bool:
        record = self._records.get(job_id)
        return bool(record and record.status == "queued")

    def snapshot(self) -> dict[str, Any]:
        """Return a small dict of executor-level metrics for logging."""
        active = sum(1 for r in self._records.values() if r.status == "running")
        queued = sum(1 for r in self._records.values() if r.status == "queued")
        return {
            "max_concurrent": self._max_concurrent,
            "active": active,
            "queued": queued,
            "completed_total": self._completed_total,
            "failed_total": self._failed_total,
            "cancelled_total": self._cancelled_total,
            "tracked_jobs": len(self._records),
        }

    def shutdown(self) -> None:
        """Cancel all in-flight tasks. Idempotent.

        Called from process-shutdown paths to ensure no orphan tasks
        keep the loop alive at exit. Tests may call this in
        ``tearDown`` to make a fresh executor state.
        """
        for record in self._records.values():
            if record.task is not None and not record.task.done():
                record.task.cancel()
        self._records.clear()


__all__ = [
    "InProcessJobExecutor",
    "JobRecord",
    "use_inprocess_executor",
]
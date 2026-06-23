# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa: E501
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

"""Launch a local Dask cluster (or skip it) and the web server.

Wave 3 W3.4 — when ``AIQ_USE_INPROCESS_EXECUTOR=1`` is set we skip the
Dask scheduler and worker subprocesses entirely. The deep-research jobs
run on the uvicorn event loop via :mod:`aiq_api.jobs.asyncio_executor`.
The web server's lifecycle (signal handlers, graceful shutdown) stays
identical; only the Dask-specific child processes are not spawned.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _install_signal_handlers(
    web_proc: subprocess.Popen[str],
    dask_procs: list[subprocess.Popen[str]] | None = None,
) -> None:
    """Install SIGTERM/SIGINT handlers that shut down ``web_proc`` and any
    Dask child processes (when Dask is in use)."""
    dask_procs = dask_procs or []

    def _handle_signal(_signum: int, _frame: object) -> None:
        print("Shutting down...", flush=True)
        _terminate_process(web_proc)
        for proc in reversed(dask_procs):
            _terminate_process(proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def _wait_for_scheduler(port: int) -> None:
    from distributed import Client

    print("Waiting for scheduler to start...", flush=True)
    for attempt in range(1, 31):
        try:
            Client(f"tcp://localhost:{port}", timeout="2s").close()
            print("Scheduler ready.", flush=True)
            return
        except Exception as exc:
            if attempt == 30:
                raise RuntimeError("Scheduler failed to start") from exc
            time.sleep(1)


def main() -> int:
    if len(sys.argv) > 1:
        os.execvp(sys.argv[1], sys.argv[1:])

    config_file = os.getenv(
        "CONFIG_FILE",
        "/app/configs/config_web_default_llamaindex.yml",
    )
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # Wave 3 W3.4 — opt into the in-process asyncio executor. When set,
    # we skip the Dask scheduler and worker subprocesses entirely.
    use_inprocess = os.getenv("AIQ_USE_INPROCESS_EXECUTOR", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    print("============================================", flush=True)
    if use_inprocess:
        print("NVIDIA NeMo Agent toolkit - InProcess Executor Mode", flush=True)
    else:
        print("NVIDIA NeMo Agent toolkit - Local Dask Mode", flush=True)
    print("============================================", flush=True)
    print("", flush=True)
    print(f"Config: {config_file}", flush=True)
    print(f"API:    http://{host}:{port}", flush=True)
    if not use_inprocess:
        scheduler_port = int(os.getenv("DASK_SCHEDULER_PORT", "8786"))
        print(f"Dask:   tcp://localhost:{scheduler_port}", flush=True)
    print("", flush=True)

    dask_procs: list[subprocess.Popen[str]] = []

    if not use_inprocess:
        scheduler_port = int(os.getenv("DASK_SCHEDULER_PORT", "8786"))
        nworkers = os.getenv("DASK_NWORKERS", "1")
        nthreads = os.getenv("DASK_NTHREADS", "4")
        memory_limit = os.getenv("DASK_MEMORY_LIMIT")
        lifetime = os.getenv("DASK_LIFETIME")

        scheduler_proc = subprocess.Popen(
            [
                "dask-scheduler",
                "--port",
                str(scheduler_port),
                "--dashboard-address",
                ":8787",
            ],
        )
        dask_procs.append(scheduler_proc)

        try:
            _wait_for_scheduler(scheduler_port)
        except RuntimeError as exc:
            _terminate_process(scheduler_proc)
            raise SystemExit(str(exc)) from exc

        worker_args = [
            "dask-worker",
            f"tcp://localhost:{scheduler_port}",
            "--nworkers",
            str(nworkers),
            "--nthreads",
            str(nthreads),
            "--no-dashboard",
        ]
        if memory_limit:
            worker_args += ["--memory-limit", memory_limit]
        if lifetime:
            lifetime_restart = os.getenv("DASK_LIFETIME_RESTART", "true").lower() != "false"
            worker_args += ["--lifetime", lifetime]
            if lifetime_restart:
                worker_args += ["--lifetime-restart"]

        worker_proc = subprocess.Popen(worker_args)
        dask_procs.append(worker_proc)

        print("Waiting for worker to connect...", flush=True)
        time.sleep(3)

        os.environ["NAT_DASK_SCHEDULER_ADDRESS"] = f"tcp://localhost:{scheduler_port}"

        print("", flush=True)
        print("--------------------------------------------", flush=True)
        print("  Dask cluster ready", flush=True)
        print("  Starting web server...", flush=True)
        print("--------------------------------------------", flush=True)
    else:
        # In-process executor: deep-research jobs run on the uvicorn loop
        # via :mod:`aiq_api.jobs.asyncio_executor`. No NAT_DASK_SCHEDULER_ADDRESS
        # is set; ``JobStore`` calls inside the runner pass a sentinel
        # address that is never connected to (Dask is only used for job
        # submission, not for status or cancellation).
        print("", flush=True)
        print("--------------------------------------------", flush=True)
        print("  InProcess executor ready (no Dask subprocesses)", flush=True)
        print("  Starting web server...", flush=True)
        print("--------------------------------------------", flush=True)

    print("", flush=True)

    web_proc = subprocess.Popen(["python", "/app/deploy/start_web.py"])
    _install_signal_handlers(web_proc, dask_procs=dask_procs if dask_procs else None)

    try:
        return web_proc.wait()
    finally:
        _terminate_process(web_proc)
        for proc in reversed(dask_procs):
            _terminate_process(proc)


if __name__ == "__main__":
    sys.exit(main())

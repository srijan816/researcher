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

"""
Agent-agnostic job runner.

Provides the Dask task function for running any registered agent with:
- NAT's JobStore for job metadata and status
- SSE event streaming for real-time UI updates
- Cancellation monitoring for graceful job termination
- Phoenix/OpenTelemetry observability via NAT's ExporterManager
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import uuid
from typing import Any

from aiq_agent.agents.chat_researcher.utils import coerce_content_text
from aiq_agent.common import DEFAULT_RESEARCH_DEPTH
from aiq_agent.common import ResearchDepthTier
from aiq_agent.common.claim_table import summarize_claim_table
from aiq_agent.common.fact_ledger import summarize_fact_ledger
from aiq_agent.common.report_quality import is_model_failure_report
from aiq_agent.common.report_quality import report_matches_request_scope
from aiq_agent.common.source_classification import normalize_source_class
from aiq_agent.common.source_quality_gates import evaluate_source_quality

from .callbacks import AgentEventCallback
from .event_store import BatchingEventStore
from .event_store import EventStore
from .webhooks import build_terminal_payload
from .webhooks import send_job_webhook

logger = logging.getLogger(__name__)


def _normalize_trace_id(trace_id: int | str | None) -> int | None:
    """Convert trace ID to integer format.

    Args:
        trace_id: Trace ID as int, hex string, or None.

    Returns:
        Integer trace ID or None.
    """
    if trace_id is None:
        return None
    if isinstance(trace_id, int):
        return trace_id
    try:
        return int(trace_id, 16)
    except ValueError:
        return int(trace_id)


class CancellationMonitor:
    """
    Monitors job status for cancellation requests.

    Polls the job store at regular intervals and sets an asyncio.Event
    when the job status changes to INTERRUPTED.
    """

    def __init__(
        self,
        scheduler_address: str,
        db_url: str,
        job_id: str,
        poll_interval: float = 1.0,
    ):
        self.scheduler_address = scheduler_address
        self.db_url = db_url
        self.job_id = job_id
        self.poll_interval = poll_interval
        self._cancelled = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def _poll_job_status(self) -> None:
        """Poll job status and set cancelled event if interrupted."""
        from nat.front_ends.fastapi.async_jobs.job_store import JobStatus
        from nat.front_ends.fastapi.async_jobs.job_store import JobStore

        job_store = JobStore(scheduler_address=self.scheduler_address, db_url=self.db_url)

        while not self._cancelled.is_set():
            try:
                job = await job_store.get_job(self.job_id)
                if job and job.status == JobStatus.INTERRUPTED.value:
                    logger.info("Cancellation detected for job %s", self.job_id)
                    self._cancelled.set()
                    break
            except Exception as e:
                logger.warning("Error checking job status for %s: %s", self.job_id, e)

            await asyncio.sleep(self.poll_interval)

    def start(self) -> None:
        """Start the cancellation monitor background task."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._poll_job_status())
            logger.debug("Started cancellation monitor for job %s", self.job_id)

    def stop(self) -> None:
        """Stop the cancellation monitor."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None
            logger.debug("Stopped cancellation monitor for job %s", self.job_id)

    def check(self) -> None:
        """Check if cancelled and raise CancelledError if so."""
        if self._cancelled.is_set():
            raise asyncio.CancelledError("Job cancelled by user")


# Interval for emitting heartbeat events
HEARTBEAT_INTERVAL_SECONDS = 30


async def run_with_cancellation(
    coro,
    monitor: CancellationMonitor,
    event_store: EventStore | BatchingEventStore | None = None,
) -> Any:
    """
    Run a coroutine with cancellation monitoring and periodic heartbeats.

    Emits job.heartbeat events every 30s so the SSE stream stays alive
    and the ghost job reaper can detect dead workers.
    Raises asyncio.CancelledError if the monitor detects cancellation.
    """
    import time

    task = asyncio.create_task(coro)
    monitor.start()
    start_time = time.monotonic()
    last_heartbeat = start_time

    try:
        while not task.done():
            if monitor.is_cancelled:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError("Job cancelled by user")

            now = time.monotonic()
            if event_store and (now - last_heartbeat) >= HEARTBEAT_INTERVAL_SECONDS:
                last_heartbeat = now
                event_store.store(
                    {
                        "type": "job.heartbeat",
                        "data": {"uptime_seconds": int(now - start_time)},
                    }
                )

            await asyncio.sleep(0.1)

        return task.result()
    finally:
        monitor.stop()


def _load_agent_class(agent_class_path: str) -> type:
    """
    Dynamically load an agent class from its module path.

    Args:
        agent_class_path: Full path like 'aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent'

    Returns:
        The agent class.

    Raises:
        ImportError: If the module or class cannot be found.
    """
    module_path, class_name = agent_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def _notify_terminal_webhook(
    *,
    webhook_config: dict[str, Any] | None,
    event_store: EventStore | BatchingEventStore | None,
    job_id: str,
    status: str,
    agent_config_name: str,
    agent_class_path: str,
    research_depth: str,
    has_report: bool,
    error: str | None = None,
    quality_warnings: list[str] | None = None,
    recovered: bool = False,
) -> None:
    """Send a best-effort terminal webhook and never fail the job on delivery errors."""
    if not webhook_config:
        return
    payload = build_terminal_payload(
        job_id=job_id,
        status=status,
        agent_config_name=agent_config_name,
        agent_class_path=agent_class_path,
        research_depth=research_depth,
        has_report=has_report,
        error=error,
        quality_warnings=quality_warnings,
        recovered=recovered,
    )
    try:
        await send_job_webhook(config=webhook_config, payload=payload, event_store=event_store)
    except Exception as exc:  # pragma: no cover - defensive guard around best-effort callback
        logger.warning("Ignoring terminal webhook failure for job %s: %s", job_id, exc)


async def run_agent_job(
    configure_logging: bool,
    log_level: int,
    scheduler_address: str | None,
    db_url: str,
    config_file_path: str,
    job_id: str,
    input_text: str,
    agent_class_path: str,
    agent_config_name: str,
    parent_span_id: str | None = None,
    parent_function_id: str | None = None,
    parent_function_name: str | None = None,
    parent_workflow_run_id: str | None = None,
    parent_workflow_trace_id: int | str | None = None,
    parent_conversation_id: str | None = None,
    request_trace_tags: dict[str, str] | None = None,
    available_documents: list[dict] | None = None,
    data_sources: list[str] | None = None,
    auth_token: str | None = None,
    research_depth: ResearchDepthTier = DEFAULT_RESEARCH_DEPTH,
    resume_files: dict[str, Any] | None = None,
    webhook_config: dict[str, Any] | None = None,
    include_images: bool = False,
):
    """
    Dask task to run any registered agent with cancellation support and telemetry.

    Wave 3 W3.4 — this function is the same one that used to run inside a
    Dask worker process; it now runs in-process on the agent's uvicorn
    event loop when ``AIQ_USE_INPROCESS_EXECUTOR=1`` is set, and inside
    a Dask worker otherwise. ``scheduler_address`` is unused on the
    in-process path (``JobStore`` only needs it for ``dask_client.submit``)
    but is preserved as a positional argument for backwards compatibility
    with existing Dask-task call sites.

    Responsibilities:
    - Uses NAT's JobStore for status tracking (postgres/sqlite — not Dask)
    - Monitors for cancellation requests and gracefully terminates the agent
    - Exports telemetry to Phoenix/OpenTelemetry via NAT's ExporterManager
    - Propagates trace context from parent workflow for nested spans

    Args:
        configure_logging: Whether to set up logging in the worker.
        log_level: Logging level to use.
        scheduler_address: Dask scheduler address. Optional on the
            in-process path; required on the Dask path. The Dask
            scheduler is only consulted at job submission time — once
            the worker is running, status updates and cancellation polls
            go through NAT's ``JobStore`` (postgres/sqlite), not Dask.
        db_url: Database URL for job store and event store.
        config_file_path: Path to NAT config file.
        job_id: Unique job identifier.
        input_text: User input/query to run.
        agent_class_path: Full module path to agent class.
        agent_config_name: NAT config function name for the agent.
        parent_span_id: Parent span ID for trace continuity (from caller context).
        parent_function_id: Parent function ID for span hierarchy.
        parent_function_name: Parent function name for span metadata.
        parent_workflow_run_id: Parent workflow run ID for trace grouping.
        parent_workflow_trace_id: Parent trace ID (int or hex string) for trace continuity.
        parent_conversation_id: Conversation ID for session grouping in Phoenix.
        request_trace_tags: Request trace tags captured at async submission time.
        available_documents: Optional list of document dicts with file_name and summary.
        data_sources: Optional list of allowed data sources to enforce in the worker.
        auth_token: Optional auth token propagated from the HTTP request for
            data sources that require authentication (requires_auth: true).
        research_depth: Source/depth tier for deep research workloads.
        resume_files: Optional virtual filesystem snapshot recovered from a
            previous failed attempt for the same job.
        webhook_config: Optional terminal-status webhook config passed from
            the API submit request.
        include_images: Opt-in flag to embed up to three generated images in
            the final research report.
    """

    # Propagate auth token into the current async task's context so tools
    # can retrieve it via get_auth_token(). Uses a ContextVar so concurrent
    # jobs in the same Dask worker process don't leak tokens across tasks.
    _auth_token_reset = None
    if auth_token:
        from ._auth_context import job_auth_token

        _auth_token_reset = job_auth_token.set(auth_token)

    from aiq_api.auth.request_trace import install_request_trace_span_injection
    from aiq_api.auth.request_trace import request_trace_tag_context

    install_request_trace_span_injection()

    from aiq_agent.common import LLMProvider
    from aiq_agent.common import LLMRole
    from aiq_agent.common import VerboseTraceCallback
    from aiq_agent.common import is_verbose
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.front_ends.fastapi.async_jobs.job_store import JobStatus
    from nat.front_ends.fastapi.async_jobs.job_store import JobStore
    from nat.runtime.loader import load_config

    if configure_logging:
        try:
            from nat.utils.log_utils import setup_logging

            setup_logging(log_level)
        except ImportError:
            import logging as std_logging

            std_logging.basicConfig(level=log_level)

    job_store: JobStore | None = None
    cancellation_monitor: CancellationMonitor | None = None
    event_store: EventStore | BatchingEventStore | None = None
    agent_event_callback: AgentEventCallback | None = None
    logger.info(
        "Dask worker received: agent=%s, config=%s, job_id=%s",
        agent_class_path,
        agent_config_name,
        job_id,
    )

    try:
        # Wave 3 W3.4 — when running in-process the executor passes a
        # sentinel scheduler_address (or None). The runner only uses
        # ``JobStore.update_status`` / ``get_job`` and ``CancellationMonitor``,
        # both of which are pure SQLAlchemy. The Dask client is only
        # instantiated lazily on first access (see ``DaskClientMixin``), and
        # this code path never accesses it, so passing any non-empty
        # placeholder is safe.
        effective_scheduler_address = scheduler_address or "in-process"
        job_store = JobStore(scheduler_address=effective_scheduler_address, db_url=db_url)
        await job_store.update_status(job_id, JobStatus.RUNNING)

        cancellation_monitor = CancellationMonitor(
            scheduler_address=effective_scheduler_address,
            db_url=db_url,
            job_id=job_id,
            poll_interval=1.0,
        )

        config = load_config(config_file_path)

        # Dynamically load the agent class
        agent_cls = _load_agent_class(agent_class_path)

        async with WorkflowBuilder.from_config(config=config) as builder:
            fn_config = builder.get_function_config(agent_config_name)

            # Get LLMs - handle both deep_researcher (orchestrator_llm) and shallow/other agents (llm)
            orchestrator_llm = None
            if hasattr(fn_config, "orchestrator_llm") and fn_config.orchestrator_llm:
                orchestrator_llm = await builder.get_llm(
                    fn_config.orchestrator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
                )
            planner_llm = None
            researcher_llm = None
            medium_orchestrator_llm = None
            deeper_orchestrator_llm = None
            deeper_planner_llm = None
            deeper_researcher_llm = None
            if hasattr(fn_config, "medium_orchestrator_llm") and fn_config.medium_orchestrator_llm:
                medium_orchestrator_llm = await builder.get_llm(
                    fn_config.medium_orchestrator_llm,
                    wrapper_type=LLMFrameworkEnum.LANGCHAIN,
                )
            if hasattr(fn_config, "deeper_orchestrator_llm") and fn_config.deeper_orchestrator_llm:
                deeper_orchestrator_llm = await builder.get_llm(
                    fn_config.deeper_orchestrator_llm,
                    wrapper_type=LLMFrameworkEnum.LANGCHAIN,
                )
            if hasattr(fn_config, "deeper_planner_llm") and fn_config.deeper_planner_llm:
                deeper_planner_llm = await builder.get_llm(
                    fn_config.deeper_planner_llm,
                    wrapper_type=LLMFrameworkEnum.LANGCHAIN,
                )
            if hasattr(fn_config, "deeper_researcher_llm") and fn_config.deeper_researcher_llm:
                deeper_researcher_llm = await builder.get_llm(
                    fn_config.deeper_researcher_llm,
                    wrapper_type=LLMFrameworkEnum.LANGCHAIN,
                )
            if hasattr(fn_config, "planner_llm") and fn_config.planner_llm:
                planner_llm = await builder.get_llm(fn_config.planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            if hasattr(fn_config, "researcher_llm") and fn_config.researcher_llm:
                researcher_llm = await builder.get_llm(
                    fn_config.researcher_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN
                )

            llm = orchestrator_llm
            if llm is None and hasattr(fn_config, "llm") and fn_config.llm:
                llm = await builder.get_llm(fn_config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

            # Resolve tools: use explicit list or auto-inherit from data_source_registry.
            # Not every async agent is LangChain-tool bound. The Claude Code
            # research bridge uses its own repo-local subprocess/tool contract,
            # so its config intentionally has no ``tools`` field.
            tool_refs = _resolve_tool_refs(fn_config)
            tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

            # Apply per-agent exclusions (e.g. deep_research excludes web_search_tool)
            if hasattr(fn_config, "exclude_tools") and fn_config.exclude_tools:
                excluded = set(fn_config.exclude_tools)
                tools = [t for t in tools if getattr(t, "name", "") not in excluded]

            if data_sources is not None:
                from aiq_agent.common import filter_tools_by_sources

                tools = filter_tools_by_sources(tools, data_sources)

            # Set up telemetry/observability for Phoenix and OpenTelemetry
            from nat.builder.context import Context
            from nat.builder.context import ContextState
            from nat.data_models.intermediate_step import IntermediateStepPayload
            from nat.data_models.intermediate_step import IntermediateStepType
            from nat.data_models.intermediate_step import StreamEventData
            from nat.data_models.intermediate_step import TraceMetadata
            from nat.data_models.invocation_node import InvocationNode
            from nat.observability.exporter_manager import ExporterManager
            from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
            from nat.utils.reactive.subject import Subject

            telemetry_exporters = {
                name: configured.instance for name, configured in builder._telemetry_exporters.items()
            }
            exporter_manager = ExporterManager.from_exporters(telemetry_exporters)

            # Initialize context state with trace propagation from parent
            context_state = ContextState.get()
            context_state.workflow_run_id.set(job_id)
            if parent_conversation_id:
                context_state.conversation_id.set(parent_conversation_id)

            workflow_trace_id = _normalize_trace_id(parent_workflow_trace_id) or uuid.uuid4().int
            context_state.workflow_trace_id.set(workflow_trace_id)

            # Event stream for exporters to subscribe to
            event_stream = Subject()
            context_state.event_stream.set(event_stream)

            # Initialize span stack (triggers default ["root"])
            _ = context_state.active_span_id_stack

            # Set up span hierarchy metadata
            workflow_span_name = f"async_job:{agent_config_name}"
            context_state.active_function.set(
                InvocationNode(
                    function_name=workflow_span_name,
                    function_id=job_id,
                    parent_id=parent_function_id,
                    parent_name=parent_function_name,
                )
            )

            context = Context(context_state)

            workflow_metadata = TraceMetadata(
                provided_metadata={
                    "workflow_run_id": job_id,
                    "workflow_trace_id": f"{workflow_trace_id:032x}",
                    "conversation_id": parent_conversation_id,
                    "agent": agent_class_path,
                    "parent_workflow_run_id": parent_workflow_run_id,
                    "parent_workflow_name": parent_function_name,
                }
            )

            # Run with telemetry - exporter must start before pushing events
            with request_trace_tag_context(request_trace_tags or {}):
                async with exporter_manager.start(context_state=context_state):
                    # Link to parent span if provided (for nested trace continuity)
                    parent_metadata: TraceMetadata | None = None
                    if parent_span_id and parent_span_id != "root":
                        parent_metadata = TraceMetadata(
                            provided_metadata={
                                "workflow_run_id": parent_workflow_run_id,
                                "workflow_trace_id": f"{workflow_trace_id:032x}",
                                "conversation_id": parent_conversation_id,
                                "workflow_name": parent_function_name,
                            }
                        )
                        context.intermediate_step_manager.push_intermediate_step(
                            IntermediateStepPayload(
                                UUID=parent_span_id,
                                event_type=IntermediateStepType.SPAN_START,
                                name=parent_function_name or "parent_workflow",
                                metadata=parent_metadata,
                            )
                        )

                    # Push WORKFLOW_START first so LLM/tool events become children
                    context.intermediate_step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            UUID=job_id,
                            event_type=IntermediateStepType.WORKFLOW_START,
                            name=workflow_span_name,
                            metadata=workflow_metadata,
                            data=StreamEventData(input=input_text),
                        )
                    )

                    # Create profiler callback AFTER workflow starts (ensures correct parent)
                    nat_profiler_callback = LangchainProfilerHandler()

                    # Set up LLM provider
                    provider = LLMProvider()
                    provider.set_default(llm)
                    if orchestrator_llm:
                        provider.configure(LLMRole.ORCHESTRATOR, orchestrator_llm)
                    if medium_orchestrator_llm:
                        provider.configure(LLMRole.MEDIUM_ORCHESTRATOR, medium_orchestrator_llm)
                    if deeper_orchestrator_llm:
                        provider.configure(LLMRole.DEEPER_ORCHESTRATOR, deeper_orchestrator_llm)
                    if deeper_planner_llm:
                        provider.configure(LLMRole.DEEPER_PLANNER, deeper_planner_llm)
                    if deeper_researcher_llm:
                        provider.configure(LLMRole.DEEPER_RESEARCHER, deeper_researcher_llm)
                    if planner_llm:
                        provider.configure(LLMRole.PLANNER, planner_llm)
                    if researcher_llm:
                        provider.configure(LLMRole.RESEARCHER, researcher_llm)

                    verbose = is_verbose(getattr(fn_config, "verbose", False))
                    callbacks = [VerboseTraceCallback()] if verbose else []

                    raw_event_store = EventStore(db_url, job_id)
                    event_store = BatchingEventStore(raw_event_store)
                    agent_event_callback = AgentEventCallback(event_store)
                    # Wave 3 W3.2 — opt-in section-by-section final-report
                    # emission. The frontend's setReportContent replaces, so
                    # cumulative section emissions make the report appear to
                    # grow on screen. Default off to preserve current
                    # behaviour; flip on in deploy/.env when ready.
                    if os.getenv("AIQ_FINAL_REPORT_SECTION_STREAM", "false").lower() in ("1", "true", "yes"):
                        agent_event_callback.final_report_section_streaming = True
                    callbacks.append(agent_event_callback)
                    callbacks.append(nat_profiler_callback)

                    # Instantiate agent with callbacks
                    agent = _create_agent_instance(
                        agent_cls=agent_cls,
                        llm_provider=provider,
                        llm=llm,
                        tools=tools,
                        fn_config=fn_config,
                        verbose=verbose,
                        callbacks=callbacks,
                        job_id=job_id,
                    )

                    # Run agent - LLM/tool events will be nested under workflow span
                    result = await _run_agent(
                        agent=agent,
                        input_text=input_text,
                        monitor=cancellation_monitor,
                        available_documents=available_documents,
                        data_sources=data_sources,
                        research_depth=research_depth,
                        include_images=include_images,
                        resume_files=resume_files,
                        event_store=event_store,
                    )

                    # Emit WORKFLOW_END event for Phoenix
                    context.intermediate_step_manager.push_intermediate_step(
                        IntermediateStepPayload(
                            UUID=job_id,
                            event_type=IntermediateStepType.WORKFLOW_END,
                            name=workflow_span_name,
                            metadata=workflow_metadata,
                            data=StreamEventData(output=_extract_result(result)),
                        )
                    )

                    if parent_metadata:
                        context.intermediate_step_manager.push_intermediate_step(
                            IntermediateStepPayload(
                                UUID=parent_span_id,
                                event_type=IntermediateStepType.SPAN_END,
                                name=parent_function_name or "parent_workflow",
                                metadata=parent_metadata,
                            )
                        )

                    # Signal event stream completion
                    event_stream.on_complete()

                    # Flush any buffered events before updating status
                    if hasattr(event_store, "flush"):
                        event_store.flush()

                    # Extract report and update status inside the context manager
                    # so the UI sees completion before exporter flush and cleanup
                    result_report = _extract_result(result)
                    artifact_report = _extract_report_from_events(db_url, job_id)
                    report = _prefer_report(result_report, artifact_report)
                    is_deep_research_job = "deep_researcher" in agent_class_path.lower()
                    if is_deep_research_job:
                        scope_ok, scope_reason = report_matches_request_scope(report, input_text)
                        if not scope_ok:
                            raise RuntimeError("Deep research report drifted from the requested topic: " + scope_reason)
                    if is_model_failure_report(report) or not report.strip():
                        raise RuntimeError(report.strip() or "Deep research completed without a report")
                    quality_warnings: list[str] = []
                    if is_deep_research_job:
                        quality_warnings = _evaluate_post_run_quality(db_url, job_id)
                        _emit_quality_verification_artifact(event_store, quality_warnings)
                        if quality_warnings:
                            _emit_quality_audit_artifacts(event_store, quality_warnings)
                            logger.warning(
                                "Job %s completed with deep-research quality warning(s): %s",
                                job_id,
                                "; ".join(quality_warnings),
                            )
                    _persist_job_metrics(event_store, agent_event_callback)
                    await job_store.update_status(
                        job_id,
                        JobStatus.SUCCESS,
                        output=_build_success_output(report, quality_warnings=quality_warnings),
                    )
                    await _notify_terminal_webhook(
                        webhook_config=webhook_config,
                        event_store=event_store,
                        job_id=job_id,
                        status=JobStatus.SUCCESS.value,
                        agent_config_name=agent_config_name,
                        agent_class_path=agent_class_path,
                        research_depth=str(research_depth),
                        has_report=True,
                        quality_warnings=quality_warnings,
                    )
                    logger.info(
                        "Job %s completed (report: %d chars, quality_warnings=%d)",
                        job_id,
                        len(report),
                        len(quality_warnings),
                    )

    except asyncio.CancelledError:
        logger.info("Job %s received cancellation", job_id)
        was_user_cancelled = False
        if job_store:
            try:
                job = await job_store.get_job(job_id)
                was_user_cancelled = bool(job and job.status == JobStatus.INTERRUPTED.value)
                if not was_user_cancelled:
                    await job_store.update_status(
                        job_id,
                        JobStatus.FAILURE,
                        error="Worker task cancelled unexpectedly",
                    )
            except (ConnectionError, TimeoutError, RuntimeError):
                pass

        if event_store is None:
            event_store = BatchingEventStore(EventStore(db_url, job_id))

        _persist_job_metrics(event_store, agent_event_callback)
        if was_user_cancelled:
            event_store.store(
                {
                    "type": "job.cancelled",
                    "data": {"reason": "cancelled by user"},
                }
            )
        else:
            event_store.store(
                {
                    "type": "job.error",
                    "data": {
                        "error": "Worker task cancelled unexpectedly",
                        "error_type": "WorkerCancelled",
                    },
                }
            )
        await _notify_terminal_webhook(
            webhook_config=webhook_config,
            event_store=event_store,
            job_id=job_id,
            status=JobStatus.INTERRUPTED.value if was_user_cancelled else JobStatus.FAILURE.value,
            agent_config_name=agent_config_name,
            agent_class_path=agent_class_path,
            research_depth=str(research_depth),
            has_report=False,
            error=None if was_user_cancelled else "Worker task cancelled unexpectedly",
        )
        if hasattr(event_store, "flush"):
            event_store.flush()

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, type(e).__name__)
        recovered_report = _recover_report_from_events(db_url, job_id)
        if recovered_report and job_store:
            await job_store.update_status(
                job_id,
                JobStatus.SUCCESS,
                output=_build_success_output(
                    recovered_report,
                    recovered_from_error=str(e),
                    quality_warnings=[
                        "Recovered a usable report from persisted artifacts after the original run failed."
                    ],
                ),
            )
            if event_store is None:
                event_store = BatchingEventStore(EventStore(db_url, job_id))
            _persist_job_metrics(event_store, agent_event_callback)
            _emit_recovered_report_artifacts(event_store, recovered_report)
            _emit_quality_audit_artifacts(
                event_store,
                ["Recovered a usable report from persisted artifacts after the original run failed."],
                recovered=True,
            )
            event_store.store(
                {
                    "type": "job.recovered",
                    "data": {
                        "reason": "Recovered a usable report from persisted research artifacts after worker failure.",
                        "original_error": str(e),
                    },
                }
            )
            await _notify_terminal_webhook(
                webhook_config=webhook_config,
                event_store=event_store,
                job_id=job_id,
                status=JobStatus.SUCCESS.value,
                agent_config_name=agent_config_name,
                agent_class_path=agent_class_path,
                research_depth=str(research_depth),
                has_report=True,
                error=str(e),
                quality_warnings=["Recovered a usable report from persisted artifacts after the original run failed."],
                recovered=True,
            )
            if hasattr(event_store, "flush"):
                event_store.flush()
            logger.info(
                "Job %s recovered from artifacts after failure (report: %d chars)",
                job_id,
                len(recovered_report),
            )
            return
        if job_store:
            await job_store.update_status(job_id, JobStatus.FAILURE, error=str(e))

        if event_store is None:
            event_store = BatchingEventStore(EventStore(db_url, job_id))

        _persist_job_metrics(event_store, agent_event_callback)
        event_store.store(
            {
                "type": "job.error",
                "data": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }
        )
        await _notify_terminal_webhook(
            webhook_config=webhook_config,
            event_store=event_store,
            job_id=job_id,
            status=JobStatus.FAILURE.value,
            agent_config_name=agent_config_name,
            agent_class_path=agent_class_path,
            research_depth=str(research_depth),
            has_report=False,
            error=str(e),
        )
        if hasattr(event_store, "flush"):
            event_store.flush()

    finally:
        # Ensure terminal-path events are not left in the batch buffer.
        if event_store is not None and hasattr(event_store, "flush"):
            event_store.flush()
        if cancellation_monitor:
            cancellation_monitor.stop()
        # Clean up job-scoped auth token
        if _auth_token_reset is not None:
            from ._auth_context import job_auth_token

            job_auth_token.reset(_auth_token_reset)


def _persist_job_metrics(
    event_store: EventStore | BatchingEventStore | None,
    agent_event_callback: AgentEventCallback | None,
) -> None:
    """Persist a best-effort ``job.metrics`` summary event before the terminal event.

    Persist-then-notify semantics come from EventStore.store; this helper must
    never raise — metrics can never fail a job.
    """
    if event_store is None or agent_event_callback is None:
        return
    try:
        summary = agent_event_callback.metrics.summary()
        event_store.store({"type": "job.metrics", "data": summary})
    except Exception as exc:  # noqa: BLE001 - metrics are best-effort
        logger.warning("Failed to persist job metrics: %s", exc)


def _create_agent_instance(
    agent_cls: type,
    llm_provider,
    llm,
    tools: list,
    fn_config,
    verbose: bool,
    callbacks: list,
    job_id: str | None = None,
):
    """
    Create an agent instance, supporting different constructor patterns.

    Tries in order:
    1. llm_provider + tools pattern (DeepResearcherAgent style)
    2. llm + tools pattern (simpler agents)
    """
    # Try async deep_researcher pattern with generic function config and job-scoped runtime state.
    try:
        return agent_cls(
            llm_provider=llm_provider,
            tools=tools,
            max_loops=getattr(fn_config, "max_loops", 3),
            verbose=verbose,
            callbacks=callbacks,
            config=fn_config,
            job_id=job_id,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise

    # Try original deep_researcher pattern (llm_provider + tools + max_loops + verbose)
    try:
        return agent_cls(
            llm_provider=llm_provider,
            tools=tools,
            max_loops=getattr(fn_config, "max_loops", 3),
            verbose=verbose,
            callbacks=callbacks,
        )
    except TypeError:
        pass

    # Try llm_provider + tools pattern (ShallowResearcherAgent style)
    try:
        return agent_cls(
            llm_provider=llm_provider,
            tools=tools,
            max_tool_iterations=getattr(fn_config, "max_tool_iterations", 5),
            callbacks=callbacks,
        )
    except TypeError:
        pass

    # Try simpler llm + tools pattern
    try:
        return agent_cls(
            llm=llm,
            tools=tools,
            callbacks=callbacks,
        )
    except TypeError:
        pass

    # Fallback: just callbacks
    return agent_cls(callbacks=callbacks)


def _resolve_tool_refs(fn_config) -> list[str]:
    """Resolve configured tool refs for async jobs.

    Standard AIQ research configs expose a ``tools`` field and auto-inherit all
    data-source tools when that field is empty. Tool-less async agents, such as
    the Claude Code research bridge, intentionally omit the field and should not
    trigger LangChain tool construction.
    """
    if not hasattr(fn_config, "tools"):
        return []
    tool_refs = fn_config.tools
    if tool_refs:
        return list(tool_refs)

    from aiq_agent.common import get_all_tool_refs

    return list(get_all_tool_refs())


async def _run_agent(
    agent,
    input_text: str,
    monitor: CancellationMonitor,
    available_documents: list[dict] | None = None,
    data_sources: list[str] | None = None,
    research_depth: ResearchDepthTier = DEFAULT_RESEARCH_DEPTH,
    include_images: bool = False,
    resume_files: dict[str, Any] | None = None,
    event_store: EventStore | None = None,
) -> Any:
    """
    Run the agent, supporting different run() signatures.

    Tries:
    1. run(input_text: str) -> str (simple protocol)
    2. run(state) where state has messages (LangGraph pattern)
    """
    from langchain_core.messages import HumanMessage

    # Check if agent has a simple run(input_text) method
    if hasattr(agent, "run"):
        import inspect

        sig = inspect.signature(agent.run)
        params = list(sig.parameters.keys())

        # If first param is 'input_text' or 'query', use simple pattern
        if params and params[0] in ("input_text", "query", "input"):
            return await run_with_cancellation(
                agent.run(input_text),
                monitor,
                event_store=event_store,
            )

        # Otherwise assume state-based pattern
        # Try to find the agent's state class
        state_cls = _get_agent_state_class(agent)
        if state_cls:
            # Build state with available_documents if the class supports it
            state_kwargs = {"messages": [HumanMessage(content=input_text)]}
            if data_sources is not None:
                state_kwargs["data_sources"] = data_sources
            state_kwargs["research_depth"] = research_depth
            state_kwargs["include_images"] = include_images
            if resume_files:
                state_kwargs["files"] = resume_files
            if available_documents:
                # Convert dicts to AvailableDocument if the state class expects them
                try:
                    from aiq_agent.knowledge import AvailableDocument

                    state_kwargs["available_documents"] = [AvailableDocument(**doc) for doc in available_documents]
                    logger.debug(
                        "Dask worker passing %d available documents to agent state",
                        len(available_documents),
                    )
                except (ImportError, TypeError):
                    # AvailableDocument not available or state doesn't support it
                    pass
            state = state_cls(**state_kwargs)
        else:
            # Fallback: create a simple dict state
            state = {"messages": [HumanMessage(content=input_text)]}
            if data_sources is not None:
                state["data_sources"] = data_sources
            state["research_depth"] = research_depth
            state["include_images"] = include_images
            if resume_files:
                state["files"] = resume_files
            if available_documents:
                state["available_documents"] = available_documents

        return await run_with_cancellation(
            agent.run(state),
            monitor,
            event_store=event_store,
        )

    raise TypeError(f"Agent {type(agent).__name__} does not have a run method")


def _get_agent_state_class(agent) -> type | None:
    """Try to find the state class for an agent."""
    agent_module = type(agent).__module__
    agent_name = type(agent).__name__

    # Try common patterns for state class names
    # e.g., DeepResearcherAgent -> DeepResearchAgentState, DeepResearcherAgentState
    state_name_patterns = [
        "AgentState",
        f"{agent_name}State",
        f"{agent_name.replace('Agent', '')}AgentState",  # DeepResearcher -> DeepResearcherAgentState
        f"{agent_name.replace('erAgent', '')}AgentState",  # DeepResearcherAgent -> DeepResearchAgentState
        "State",
    ]

    # Try models submodule first
    try:
        models_module = importlib.import_module(agent_module.replace(".agent", ".models"))
        for state_name in state_name_patterns:
            if hasattr(models_module, state_name):
                return getattr(models_module, state_name)

        # Also scan for any class ending with "State" that has a messages field
        for name in dir(models_module):
            if name.endswith("State") and not name.startswith("_"):
                cls = getattr(models_module, name)
                if isinstance(cls, type) and hasattr(cls, "model_fields"):
                    if "messages" in cls.model_fields:
                        return cls
    except (ImportError, AttributeError):
        pass

    # Try same module
    try:
        module = importlib.import_module(agent_module)
        for state_name in state_name_patterns:
            if hasattr(module, state_name):
                return getattr(module, state_name)
    except ImportError:
        pass

    return None


def _extract_result(result: Any) -> str:
    """Extract string result from various result formats."""
    # Direct string
    if isinstance(result, str):
        return result

    # State with messages
    if hasattr(result, "messages") and result.messages:
        last_msg = result.messages[-1]
        if hasattr(last_msg, "content"):
            return _coerce_report_text(last_msg.content)

    # Dict with messages
    if isinstance(result, dict):
        if "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                return _coerce_report_text(last_msg.content)
        if "report" in result:
            return _coerce_report_text(result["report"])
        if "output" in result:
            return _coerce_report_text(result["output"])

    return str(result) if result else ""


def _prefer_report(result_report: str, artifact_report: str | None) -> str:
    """Prefer the substantive report over short terminal messages."""
    if _is_usable_report(artifact_report):
        return artifact_report
    if _is_usable_report(result_report):
        return result_report
    return artifact_report or result_report or ""


def _coerce_report_text(value: Any) -> str:
    """Convert provider output to text while dropping raw thinking/tool blocks."""
    if _looks_like_provider_content_block(value):
        return ""
    return coerce_content_text(value)


def _is_usable_report(content: str | None, *, min_chars: int = 1) -> bool:
    """Return true when text is plausible user-facing report content."""
    if not isinstance(content, str):
        return False
    text = content.strip()
    return bool(len(text) >= min_chars and not is_model_failure_report(text) and not _looks_like_tool_call_draft(text))


def _build_success_output(
    report: str,
    *,
    quality_warnings: list[str] | None = None,
    recovered_from_error: str | None = None,
) -> dict[str, Any]:
    """Build a stable success payload for final reports and quality metadata."""
    warnings = [str(item) for item in (quality_warnings or []) if str(item).strip()]
    output: dict[str, Any] = {"report": report}
    if warnings:
        output["quality_status"] = "warning"
        output["quality_warnings"] = warnings
    else:
        output["quality_status"] = "pass"
        output["quality_warnings"] = []
    if recovered_from_error:
        output["recovered_from_error"] = recovered_from_error
    return output


def _emit_quality_audit_artifacts(
    event_store: Any,
    quality_warnings: list[str],
    *,
    recovered: bool = False,
) -> None:
    """Persist a non-fatal quality audit artifact for UI/API clients."""
    warnings = [str(item) for item in quality_warnings if str(item).strip()]
    if not warnings:
        return
    event_store.store(
        {
            "type": "artifact.update",
            "name": "quality_audit",
            "data": {
                "type": "quality_audit",
                "content": {
                    "status": "warning",
                    "warnings": warnings,
                    "recovered": recovered,
                },
                "quality_status": "warning",
                "warnings": warnings,
                "recovered": recovered,
            },
        }
    )
    event_store.store(
        {
            "type": "quality.warning",
            "data": {
                "warnings": warnings,
                "recovered": recovered,
            },
        }
    )


def _emit_quality_verification_artifact(event_store: Any, quality_warnings: list[str]) -> None:
    """Persist a stable quality-verification JSON artifact for every completed report."""
    warnings = [str(item) for item in quality_warnings if str(item).strip()]
    artifact = {
        "schema_version": "1.0",
        "overall_status": "warning" if warnings else "pass",
        "hard_failure": False,
        "warnings": warnings,
        "metrics": {
            "warning_count": len(warnings),
        },
    }
    event_store.store(
        {
            "type": "artifact.update",
            "name": "quality_verification.json",
            "data": {
                "type": "quality_verification",
                "content": artifact,
                "quality_status": artifact["overall_status"],
                "warnings": warnings,
            },
        }
    )


def _evaluate_post_run_quality(db_url: str, job_id: str) -> list[str]:
    """Return post-run quality gate failures for deep-research artifacts."""
    events = EventStore.get_events(db_url, job_id, 0, 10000)
    collected_sources: set[str] = set()
    cited_sources: set[str] = set()
    cited_source_classes: list[str] = []
    has_report_file = False
    failed_researcher_tasks: list[str] = []
    plan_content: str | None = None
    fact_ledger_content: str | None = None
    claim_table_content: str | None = None

    for event in events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        event_type = event.get("type")
        artifact_type = data.get("type")
        content = data.get("content")

        if event_type == "artifact.update" and artifact_type == "file":
            name = str(event.get("name") or data.get("file_path") or data.get("path") or data.get("filename") or "")
            name_lower = name.lower()
            if name_lower.endswith("plan.json") and isinstance(content, str) and content.strip():
                plan_content = content
            if "fact_ledger" in name_lower and name_lower.endswith(".json") and isinstance(content, str):
                fact_ledger_content = content
            if "claim_table" in name_lower and name_lower.endswith(".json") and isinstance(content, str):
                claim_table_content = content
            if name.lower().endswith("report.md") and isinstance(content, str) and content.strip():
                has_report_file = True
            continue

        if event_type == "artifact.update" and artifact_type == "citation_source":
            url = data.get("url") or content
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                collected_sources.add(_clean_recovered_url(url))
            continue

        if event_type == "artifact.update" and artifact_type == "citation_use":
            url = data.get("url") or content
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                cited_sources.add(_clean_recovered_url(url))
                source_class = data.get("source_class")
                if isinstance(source_class, str):
                    cited_source_classes.append(normalize_source_class(source_class))
            continue

        if event_type == "workflow.end":
            name = str(event.get("name") or "").lower()
            output = coerce_content_text(data.get("output"))
            if "researcher" in name and _looks_like_failed_researcher_output(output):
                failed_researcher_tasks.append(str(event.get("name") or "researcher"))

    problems = []
    if not has_report_file:
        problems.append("no /report.md file artifact was produced")
    if not cited_sources:
        problems.append("final report emitted no citation_use events")
    if failed_researcher_tasks:
        failed = ", ".join(dict.fromkeys(failed_researcher_tasks))
        problems.append(f"researcher task(s) failed: {failed}")
    if collected_sources:
        # Search/extraction can collect hundreds or thousands of candidate URLs,
        # especially with parallel M3 researcher lanes. A readable report should
        # cite the sources it actually uses, not one quarter of every candidate.
        expected_min = min(25, max(5, len(collected_sources) // 10))
        if len(collected_sources) >= 10 and len(cited_sources) < expected_min:
            problems.append(
                f"cited sources too sparse ({len(cited_sources)} cited vs "
                f"{len(collected_sources)} collected; expected at least {expected_min})"
            )
    if len(cited_sources) >= 4:
        source_quality = evaluate_source_quality(sorted(cited_sources), tier="deep")
        if source_quality.overall_status == "fail":
            problems.extend(source_quality.failure_reasons)
    central_entities = _central_entities_from_plan(plan_content)
    if central_entities:
        if not fact_ledger_content:
            problems.append(
                f"missing /shared/fact_ledger.json for entity-heavy plan ({len(central_entities)} central entities)"
            )
        else:
            summary = summarize_fact_ledger(fact_ledger_content)
            if not summary.get("valid"):
                problems.append(f"invalid fact ledger: {'; '.join(summary.get('errors') or [])}")
            else:
                entity_summary = summary.get("entities") or {}
                missing = [
                    entity
                    for entity in central_entities
                    if not isinstance(entity_summary.get(entity), dict)
                    or int(entity_summary[entity].get("verified", 0)) == 0
                ]
                if missing:
                    problems.append("central entit(ies) have zero verified ledger facts: " + ", ".join(missing[:8]))
        if len(cited_sources) >= 6:
            first_party_count = cited_source_classes.count("first_party")
            first_party_ratio = first_party_count / max(1, len(cited_source_classes))
            if cited_source_classes and first_party_ratio < 0.30:
                problems.append(
                    f"first-party citation share too low for entity-heavy report "
                    f"({first_party_count}/{len(cited_source_classes)} cited classified sources)"
                )
    if claim_table_content:
        claim_summary = summarize_claim_table(claim_table_content)
        if not claim_summary.get("valid"):
            problems.append(f"invalid claim table: {'; '.join(claim_summary.get('errors') or [])}")
        else:
            total_claims = int(claim_summary.get("total") or 0)
            by_status = claim_summary.get("by_status") if isinstance(claim_summary.get("by_status"), dict) else {}
            verified = int(by_status.get("verified", 0))
            partial = int(by_status.get("partially_verified", 0))
            if total_claims >= 8 and verified + partial < max(3, total_claims // 3):
                problems.append(
                    f"claim table under-resolved ({verified} verified, {partial} partially verified, "
                    f"{total_claims} total)"
                )
    return problems


def _central_entities_from_plan(plan_content: str | None) -> list[str]:
    """Extract central entity names from /shared/plan.json, if present."""
    if not plan_content:
        return []
    try:
        plan = json.loads(plan_content)
    except json.JSONDecodeError:
        return []
    task_analysis = plan.get("task_analysis") if isinstance(plan, dict) else None
    entities = task_analysis.get("entities") if isinstance(task_analysis, dict) else None
    names: list[str] = []
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            centrality = str(entity.get("centrality") or "").lower()
            name = entity.get("name")
            if centrality == "central" and isinstance(name, str) and name.strip():
                names.append(name.strip())
    targets = plan.get("fact_ledger_targets") if isinstance(plan, dict) else None
    if isinstance(targets, dict):
        for name in targets:
            if isinstance(name, str) and name.strip() and name.strip() not in names:
                names.append(name.strip())
    return names if len(names) >= 3 else []


def _looks_like_failed_researcher_output(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "model call failed",
            "tool budget exhausted",
            "traceback",
            "recursion limit",
            "failed after",
            "timed out",
        )
    )


def _extract_report_from_events(db_url: str, job_id: str) -> str | None:
    """Recover the final report from persisted artifact events.

    DeepAgents often writes the real report to `/shared/report.md` before the
    final chat message is produced. If the final message is just a short status
    line, relying only on the returned message makes `/report` look empty even
    though the report exists in artifacts.
    """
    from ..jobs.event_store import EventStore

    events = EventStore.get_events(db_url, job_id, 0, 10000)
    candidates: list[tuple[int, str]] = []
    for event in events:
        if event.get("type") != "artifact.update":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if not _is_usable_report(content):
            continue

        artifact_type = data.get("type")
        output_category = data.get("output_category")
        name = str(event.get("name") or data.get("file_path") or data.get("path") or data.get("filename") or "")
        name_lower = name.lower()

        if artifact_type == "output" and output_category == "final_report":
            candidates.append((3, content))
        elif artifact_type == "file" and name_lower.endswith("report.md"):
            candidates.append((2, content))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], len(item[1])))[1]


def _append_recovered_source_inventory(content: str, source_urls: list[str], *, limit: int = 100) -> str:
    """Append collected source inventory when recovery falls back to notes/consolidation."""
    if not source_urls:
        return content

    existing_urls = set(re.findall(r"https?://[^\s<>'\")\]}]+", content))
    missing_urls = [url for url in source_urls if url not in existing_urls]
    if not missing_urls:
        return content

    appendix = "\n".join(f"- {url}" for url in missing_urls[:limit])
    return f"{content.rstrip()}\n\n## Additional Sources Collected During Run\n\n{appendix}\n"


def _extract_report_reference_urls(content: str) -> list[str]:
    """Extract URLs that appear in the recovered report body, excluding source-inventory appendix."""
    report_body = content.split("## Additional Sources Collected During Run", maxsplit=1)[0]
    urls = re.findall(r"https?://[^\s<>'\")\]}]+", report_body)
    cleaned = [url.rstrip(".,;:!?)'\"}>") for url in urls]
    return list(dict.fromkeys(url for url in cleaned if url.startswith(("http://", "https://"))))


def _lesson_scope_terms(text: str | None) -> tuple[list[str], list[str]]:
    """Extract broad-topic and motion terms from lesson-style request text."""
    if not text:
        return [], []

    def _extract(patterns: list[str]) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group("value").strip().rstrip(".")
                if value:
                    return value
        return None

    broad_topic = _extract(
        [
            r"^\s*Broad topic:\s*(?P<value>.+?)\s*$",
            r"^\s*Exact lesson topic:\s*(?P<value>.+?)\s*$",
            r"^\s*Topic:\s*(?P<value>.+?)\s*$",
        ]
    )
    motion = _extract(
        [
            r"^\s*Final debate motion:\s*(?P<value>.+?)\s*$",
            r"^\s*Motion:\s*(?P<value>.+?)\s*$",
        ]
    )

    def _tokens(value: str | None) -> list[str]:
        if not value:
            return []
        return [
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9&-]+", value.lower())
            if len(token) > 2 and token not in {"the", "and", "for", "with", "this", "that"}
        ]

    return _tokens(broad_topic), _tokens(motion)


def _artifact_scope_score(
    name: str, content: str, topic_tokens: list[str], motion_tokens: list[str]
) -> tuple[int, int, int, int]:
    """Score an artifact by how much it preserves the broad lesson topic."""
    lower = content.lower()
    if not topic_tokens and not motion_tokens:
        return (0, 0, 0, len(content))

    topic_hits = sum(lower.count(token) for token in topic_tokens)
    motion_hits = sum(lower.count(token) for token in motion_tokens)
    breadth_hits = sum(
        lower.count(marker)
        for marker in (
            "definitions",
            "background",
            "stakeholders",
            "system",
            "systems",
            "examples",
            "case study",
            "case studies",
            "key terms",
            "foundations",
        )
    )
    name_bonus = sum(lower.count(token) for token in topic_tokens if token in name.lower())
    return (topic_hits + name_bonus, breadth_hits, -motion_hits, len(content))


def _emit_recovered_report_artifacts(event_store: Any, recovered_report: str) -> None:
    """Persist recovered final report and citation-use artifacts for refreshed clients."""
    event_store.store(
        {
            "type": "artifact.update",
            "name": "/report.md",
            "data": {
                "type": "output",
                "output_category": "final_report",
                "content": recovered_report,
            },
        }
    )
    for url in _extract_report_reference_urls(recovered_report):
        event_store.store(
            {
                "type": "artifact.update",
                "name": url,
                "data": {
                    "type": "citation_use",
                    "content": url,
                    "url": url,
                },
            }
        )


def _get_submission_data_from_events(db_url: str, job_id: str) -> dict[str, Any]:
    """Return the original job.submitted payload for a job, if available."""
    events = EventStore.get_events(db_url, job_id, 0, 10000)
    for event in events:
        if event.get("type") != "job.submitted":
            continue
        data = event.get("data")
        if isinstance(data, dict):
            return data
    return {}


def _get_request_text_from_events(db_url: str, job_id: str) -> str | None:
    """Return the original submitted request text for a job, if available."""
    input_text = _get_submission_data_from_events(db_url, job_id).get("input")
    return input_text if isinstance(input_text, str) else None


def _is_deep_research_submission(submission_data: dict[str, Any]) -> bool:
    """Return true for deep-research async jobs, including shallow/deeper/deep tiers."""
    agent_type = submission_data.get("agent_type")
    return isinstance(agent_type, str) and "deep_researcher" in agent_type.lower()


def _compile_final_report_from_research_artifacts(
    request_text: str | None,
    file_candidates: list[tuple[int, str, str, bool]],
    source_urls: list[str],
) -> str | None:
    """Build a final report from persisted research files without serving raw notes.

    This is a last line of defense for provider finalizer failures: if the
    researchers finished and the final model emitted only reasoning blocks, the
    API should still be able to provide a coherent synthesis artifact. It keeps
    the output visibly separate from the old raw "Recovered Research Report"
    bundle that confused downstream lesson generation.
    """
    if not file_candidates:
        return None

    topic_tokens, motion_tokens = _lesson_scope_terms(request_text)

    scoped_candidates = [
        candidate
        for candidate in file_candidates
        if report_matches_request_scope(candidate[2], request_text, candidate_name=candidate[1])[0]
    ]
    if not scoped_candidates:
        return None

    if topic_tokens or motion_tokens:
        ordered = sorted(
            scoped_candidates,
            key=lambda item: (
                _artifact_scope_score(item[1], item[2], topic_tokens, motion_tokens),
                item[0],
                len(item[2]),
            ),
            reverse=True,
        )[:8]
    else:
        ordered = sorted(scoped_candidates, key=lambda item: (item[0], len(item[2])), reverse=True)[:8]
    title = "Research Report"
    if request_text:
        title = next((line.strip().lstrip("#").strip() for line in request_text.splitlines() if line.strip()), title)
        title = re.sub(r"^you are\s+.*?deep research.*?\.\s*", "", title, flags=re.IGNORECASE) or "Research Report"
        if len(title) > 90:
            title = title[:87].rstrip() + "..."

    sections = []
    for _priority, name, content, _is_intermediate in ordered:
        clean_name = name.strip("/").split("/")[-1].replace("_", " ").replace("-", " ") or "research notes"
        snippet = content.strip()
        if len(snippet) > 18000:
            snippet = snippet[:12600].rstrip() + "\n\n[... middle truncated ...]\n\n" + snippet[-5400:].lstrip()
        sections.append(f"## {clean_name}\n\n{snippet}")

    source_lines = "\n".join(f"- {url}" for url in source_urls[:100])
    sources = source_lines or "- No validated source URLs were captured in the event log."
    report = (
        f"# {title}\n\n"
        "## Synthesis Status\n\n"
        "The model finalizer did not write `/report.md`, so this report was compiled from the persisted "
        "researcher outputs for the same job. It is a synthesized final artifact, not a raw "
        "intermediate-note bundle.\n\n" + "\n\n".join(sections) + "\n\n## Sources\n\n" + sources + "\n"
    )
    if len(report) < 1500:
        return None
    scope_ok, scope_reason = report_matches_request_scope(report, request_text)
    if not scope_ok:
        logger.warning("Skipping compiled event-artifact report because it drifted from request: %s", scope_reason)
        return None
    return report


def _recover_report_from_events(db_url: str, job_id: str) -> str | None:
    """Recover a usable report from final-report or research-note artifacts."""
    submission_data = _get_submission_data_from_events(db_url, job_id)
    request_text = submission_data.get("input") if isinstance(submission_data.get("input"), str) else None
    is_deep_research_submission = _is_deep_research_submission(submission_data)
    report = _extract_report_from_events(db_url, job_id)
    if _is_usable_report(report):
        scope_ok, _scope_reason = report_matches_request_scope(report, request_text)
        if scope_ok:
            return report

    if report and not _is_usable_report(report):
        report = None

    events = EventStore.get_events(db_url, job_id, 0, 10000)
    file_candidates: list[tuple[int, str, str, bool]] = []
    note_blocks: list[str] = []
    source_urls: list[str] = []
    seen_sources: set[str] = set()

    for event in events:
        if event.get("type") != "artifact.update":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        artifact_type = data.get("type")
        content = data.get("content")

        if artifact_type == "citation_source":
            url = data.get("url") or content
            if isinstance(url, str):
                clean_url = _clean_recovered_url(url)
                if clean_url and clean_url.startswith(("http://", "https://")) and clean_url not in seen_sources:
                    seen_sources.add(clean_url)
                    source_urls.append(clean_url)
            continue

        if artifact_type == "file" and isinstance(content, str):
            name = str(event.get("name") or data.get("file_path") or data.get("path") or data.get("filename") or "")
            name_lower = name.lower()
            if any(part in name_lower for part in ("resume_sources", "resume_instructions", "plan.json")):
                continue
            if not _is_usable_report(content, min_chars=1000):
                continue
            priority = 3 if name_lower.endswith("report.md") else 2 if "consolidated" in name_lower else 1
            file_candidates.append(
                (priority, name or "research_notes.md", content.strip(), not name_lower.endswith("report.md"))
            )
            continue

        if artifact_type != "output" or not isinstance(content, str):
            continue
        if not _is_usable_report(content, min_chars=300):
            continue
        output_category = data.get("output_category")
        if output_category not in ("research_notes", "draft", "final_report", None):
            continue
        note_blocks.append(content.strip())

    report_file_candidates = [candidate for candidate in file_candidates if not candidate[3]]
    if report_file_candidates:
        if request_text:
            topic_tokens, motion_tokens = _lesson_scope_terms(request_text)
        else:
            topic_tokens, motion_tokens = [], []

        def _candidate_score(item: tuple[int, str, str, bool]) -> tuple[int, int, int, int]:
            if not topic_tokens and not motion_tokens:
                return (item[0], len(item[2]), 0, 0)
            return (*_artifact_scope_score(item[1], item[2], topic_tokens, motion_tokens)[:3], len(item[2]))

        ordered_candidates = sorted(report_file_candidates, key=_candidate_score, reverse=True)
        for _priority, name, selected_content, _should_append_sources in ordered_candidates:
            scope_ok, scope_reason = report_matches_request_scope(selected_content, request_text, candidate_name=name)
            if scope_ok:
                return selected_content
            logger.warning("Skipping recovered report candidate %s: %s", name, scope_reason)
        return None

    if file_candidates:
        if is_deep_research_submission:
            compiled = _compile_final_report_from_research_artifacts(request_text, file_candidates, source_urls)
            return _append_recovered_source_inventory(compiled, source_urls) if compiled else None
        recovered_files = []
        for _priority, name, content, _should_append_sources in sorted(
            file_candidates,
            key=lambda item: item[1].lower(),
        ):
            recovered_files.append(f"### {name.lstrip('/')}\n\n{content}")
        recovered = (
            "# Recovered Research Report\n\n"
            "The original job collected research artifacts but did not produce `/report.md`. "
            "This report was recovered from the persisted intermediate research files.\n\n"
            "## Recovered Findings\n\n" + "\n\n---\n\n".join(recovered_files)
        )
        if report_matches_request_scope(recovered, request_text)[0]:
            return _append_recovered_source_inventory(recovered, source_urls)
        logger.warning("Skipping recovered file bundle because it drifted from the submitted request")
        return None

    if not note_blocks:
        return None

    if is_deep_research_submission:
        logger.warning(
            "Skipping recovered intermediate note bundle for deep-research job %s; no /report.md was produced",
            job_id,
        )
        return None

    sources = "\n".join(f"- {url}" for url in source_urls[:100])
    sources_section = f"\n\n## Sources\n\n{sources}" if sources else ""
    recovered = (
        "# Recovered Research Report\n\n"
        "The original job failed after collecting research artifacts. This report was recovered from persisted "
        "research notes and sources so the work is not lost.\n\n"
        "## Recovered Findings\n\n" + "\n\n---\n\n".join(note_blocks) + sources_section
    )
    if report_matches_request_scope(recovered, request_text)[0]:
        return recovered if len(recovered) >= 1500 else None
    logger.warning("Skipping recovered note bundle because it drifted from the submitted request")
    return None


def _clean_recovered_url(url: str) -> str:
    """Clean URLs recovered from event payloads."""
    import html

    clean = html.unescape(str(url)).replace("\\n", "\n").splitlines()[0].strip()
    return clean.rstrip(".,;:!?)'\"]}>")


def _looks_like_tool_call_draft(content: str) -> bool:
    """Detect raw model/tool-call payloads that should never be served as reports."""
    stripped = content.lstrip()
    if (
        stripped.startswith("[{'signature'")
        or stripped.startswith('[{"signature"')
        or stripped.startswith("[{'thinking'")
        or stripped.startswith('[{"thinking"')
        or stripped.startswith("{'thinking'")
        or stripped.startswith('{"thinking"')
    ):
        return True
    lowered = content.lower()
    return any(
        marker in lowered
        for marker in (
            "<minimax:tool_call>",
            "<invoke name=",
            "'type': 'tool_use'",
            '"type": "tool_use"',
            "'type': 'thinking'",
            '"type": "thinking"',
            "call_function_",
        )
    )


def _looks_like_provider_content_block(value: Any) -> bool:
    """Return true for raw provider block payloads rather than final prose."""
    if isinstance(value, list) and value:
        return all(
            isinstance(item, dict) and item.get("type") in {"thinking", "tool_use", "input_json_delta"}
            for item in value
        )
    if isinstance(value, dict):
        return value.get("type") in {"thinking", "tool_use", "input_json_delta"}
    return False


# Backwards compatibility alias
async def run_deep_research(
    configure_logging: bool,
    log_level: int,
    scheduler_address: str,
    db_url: str,
    config_file_path: str,
    job_id: str,
    input_text: str,
):
    """
    Legacy function for running deep research jobs.

    Preserved for backwards compatibility. New code should use run_agent_job directly.
    """
    await run_agent_job(
        configure_logging=configure_logging,
        log_level=log_level,
        scheduler_address=scheduler_address,
        db_url=db_url,
        config_file_path=config_file_path,
        job_id=job_id,
        input_text=input_text,
        agent_class_path="aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        agent_config_name="deep_research_agent",
    )

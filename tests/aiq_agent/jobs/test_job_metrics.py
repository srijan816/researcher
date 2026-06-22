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

"""Tests for per-job telemetry accumulation (job.metrics)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from aiq_api.jobs.callbacks import AgentEventCallback
from aiq_api.jobs.callbacks import JobMetrics
from aiq_api.jobs.runner import _persist_job_metrics


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestJobMetrics:
    def test_tool_and_search_counters(self):
        metrics = JobMetrics(clock=FakeClock())
        metrics.record_tool_start("web_search_tool", is_search=True)
        metrics.record_tool_start("web_search_tool", is_search=True)
        metrics.record_tool_start("write_file", is_search=False)

        summary = metrics.summary()
        assert summary["tool_calls"] == {"web_search_tool": 2, "write_file": 1}
        assert summary["tool_calls_total"] == 3
        assert summary["search_calls"] == 2

    def test_cache_hits_and_documents(self):
        metrics = JobMetrics(clock=FakeClock())
        metrics.record_tool_end(
            "web_search_tool",
            "<result><cached>true</cached>page body</result>",
            is_search=True,
            sources_seen=4,
            scrape_artifacts=2,
        )
        metrics.record_tool_end("web_search_tool", "fresh results", is_search=True, sources_seen=1)

        summary = metrics.summary()
        assert summary["cache_hits"] == 1
        assert summary["documents_seen"] == 5
        assert summary["scrape_artifacts"] == 2

    def test_error_counters(self):
        metrics = JobMetrics(clock=FakeClock())
        metrics.record_error("TimeoutError")
        metrics.record_error("TimeoutError")
        metrics.record_error("ValueError")

        summary = metrics.summary()
        assert summary["errors"] == {"TimeoutError": 2, "ValueError": 1}
        assert summary["errors_total"] == 3

    def test_phase_durations_from_synthetic_timeline(self):
        clock = FakeClock(start=0.0)
        metrics = JobMetrics(clock=clock)

        clock.advance(30)  # planning takes 30s
        metrics.record_tool_end("write_plan", "Plan successfully written")

        clock.advance(5)
        metrics.record_tool_start("web_search_tool", is_search=True)
        clock.advance(115)
        metrics.record_tool_end("web_search_tool", "results", is_search=True)

        clock.advance(60)  # synthesis takes 60s
        summary = metrics.summary()

        phases = summary["phases"]
        assert phases["planning_seconds"] == 30.0
        assert phases["researching_seconds"] == 120.0
        assert phases["synthesis_seconds"] == 60.0
        assert phases["total_seconds"] == 210.0

    def test_phase_durations_without_plan_uses_first_search(self):
        clock = FakeClock(start=0.0)
        metrics = JobMetrics(clock=clock)

        clock.advance(10)
        metrics.record_tool_start("advanced_web_search_tool", is_search=True)
        clock.advance(40)
        metrics.record_tool_end("advanced_web_search_tool", "results", is_search=True)
        clock.advance(20)

        phases = metrics.summary()["phases"]
        assert phases["planning_seconds"] == 10.0
        assert phases["researching_seconds"] == 40.0
        assert phases["synthesis_seconds"] == 20.0

    def test_summary_with_no_activity(self):
        clock = FakeClock(start=0.0)
        metrics = JobMetrics(clock=clock)
        clock.advance(7)

        summary = metrics.summary()
        assert summary["phases"]["planning_seconds"] is None
        assert summary["phases"]["researching_seconds"] is None
        assert summary["phases"]["synthesis_seconds"] is None
        assert summary["phases"]["total_seconds"] == 7.0


class TestCallbackMetricsIntegration:
    def _make_callback(self, monkeypatch):
        # Avoid real scrape-artifact persistence and citation parsing side effects.
        monkeypatch.setattr("aiq_api.jobs.callbacks.persist_scrape_artifacts", lambda **kwargs: [])
        fake_sources = [
            SimpleNamespace(url="https://example.com/a", citation_key="a", title="A", source_class="news"),
            SimpleNamespace(url="https://example.com/b", citation_key="b", title="B", source_class="news"),
        ]
        monkeypatch.setattr(
            "aiq_api.jobs.callbacks.extract_sources_from_tool_result",
            lambda tool_name, text: fake_sources,
        )
        return AgentEventCallback(event_store=None)

    def test_search_tool_lifecycle_accumulates_metrics(self, monkeypatch):
        callback = self._make_callback(monkeypatch)

        callback.on_tool_start({"name": "web_search_tool"}, "query", run_id="run-1")
        callback.on_tool_end("<cached>true</cached> results here", run_id="run-1")

        summary = callback.metrics.summary()
        assert summary["tool_calls"] == {"web_search_tool": 1}
        assert summary["search_calls"] == 1
        assert summary["documents_seen"] == 2
        assert summary["cache_hits"] == 1

    def test_llm_and_error_hooks(self, monkeypatch):
        callback = self._make_callback(monkeypatch)

        callback.on_llm_start({"name": "minimax-m3"}, ["prompt"], run_id="run-2")
        callback.on_chat_model_start({"name": "minimax-m3"}, [["msg"]], run_id="run-3")
        callback.on_chain_error(TimeoutError("slow"), run_id="run-2")
        callback.on_llm_error(ValueError("bad output"), run_id="run-2")
        callback.on_tool_error(RuntimeError("tool broke"), run_id="run-4")

        summary = callback.metrics.summary()
        assert summary["llm_calls"] == 2
        assert summary["errors"] == {"TimeoutError": 1, "ValueError": 1, "RuntimeError": 1}

    def test_metrics_never_raise_on_weird_inputs(self, monkeypatch):
        callback = self._make_callback(monkeypatch)
        callback.metrics.record_tool_start(None)  # type: ignore[arg-type]
        callback.metrics.record_tool_end(None, None)  # type: ignore[arg-type]
        callback.metrics.record_error(None)  # type: ignore[arg-type]
        summary = callback.metrics.summary()
        assert summary["tool_calls_total"] == 1


class TestPersistJobMetrics:
    def test_persists_job_metrics_event(self):
        event_store = MagicMock()
        callback = AgentEventCallback(event_store=None)
        callback.metrics.record_tool_start("web_search_tool", is_search=True)

        _persist_job_metrics(event_store, callback)

        assert event_store.store.call_count == 1
        event = event_store.store.call_args[0][0]
        assert event["type"] == "job.metrics"
        assert event["data"]["search_calls"] == 1

    def test_never_fails_job_on_metrics_error(self):
        event_store = MagicMock()
        event_store.store.side_effect = RuntimeError("db down")
        callback = AgentEventCallback(event_store=None)

        # Must not raise.
        _persist_job_metrics(event_store, callback)

    def test_noop_without_callback_or_store(self):
        _persist_job_metrics(None, None)
        _persist_job_metrics(MagicMock(), None)
        _persist_job_metrics(None, AgentEventCallback(event_store=None))

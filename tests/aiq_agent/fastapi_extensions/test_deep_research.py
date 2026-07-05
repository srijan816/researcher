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
Tests for the async job API routes.

Module under test: frontends/aiq_api/src/aiq_api/routes/jobs.py

API endpoints tested:
    GET  /v1/jobs/async/agents                         - List available agent types
    POST /v1/jobs/async/submit                         - Submit a new job
    GET  /v1/jobs/async/job/{id}                       - Get job status
    GET  /v1/jobs/async/job/{id}/stream                - SSE stream from beginning
    POST /v1/jobs/async/job/{id}/cancel                - Cancel a running job
    GET  /v1/jobs/async/job/{id}/state                 - Get artifacts from event store
    GET  /v1/jobs/async/job/{id}/report                - Get final report

Test coverage:
    TestJobSubmitRequest:
        - Valid request with defaults
        - Custom job_id and expiry_seconds
        - Empty input rejected (min_length=1)
        - Expiry validation (ge=600, le=604800)

    TestJobStatusResponse:
        - Minimal response (job_id, status)
        - Full response with error, created_at

    TestJobStateResponse:
        - Response without state (has_state=False)
        - Response with artifacts

    TestJobReportResponse:
        - Response without report (has_report=False)
        - Response with report content

    TestRegisterRoutes:
        - Routes not registered when Dask unavailable
        - Routes not registered without job_store
        - Routes registered when infrastructure available
"""

import json
from unittest.mock import MagicMock

import pytest

from aiq_api.routes.jobs import JobReportResponse
from aiq_api.routes.jobs import JobStateResponse
from aiq_api.routes.jobs import JobStatusResponse
from aiq_api.routes.jobs import JobSubmitRequest
from aiq_api.routes.jobs import _get_final_report_for_job


class TestJobSubmitRequest:
    """Tests for the JobSubmitRequest model."""

    def test_valid_request(self):
        """Test creating a valid submit request."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="What is CUDA?")

        assert req.input == "What is CUDA?"
        assert req.agent_type == "deep_researcher"
        assert req.job_id is None
        assert req.expiry_seconds is None
        assert req.research_depth == "deeper"

    def test_with_research_depth(self):
        """Test submit request with a research-depth tier."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", research_depth="deep")

        assert req.research_depth == "deep"

    def test_with_terminal_webhook(self):
        """Test submit request accepts a terminal webhook endpoint."""
        req = JobSubmitRequest(
            agent_type="deep_researcher",
            input="query",
            webhook_url="https://example.com/aiq/hook",
            webhook_headers={"X-Client": "abc"},
            webhook_secret="secret-value",  # pragma: allowlist secret
        )

        assert str(req.webhook_url) == "https://example.com/aiq/hook"
        assert req.webhook_headers == {"X-Client": "abc"}
        assert req.webhook_secret is not None
        assert req.webhook_secret.get_secret_value() == "secret-value"  # pragma: allowlist secret

    def test_with_custom_job_id(self):
        """Test submit request with custom job ID."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", job_id="custom-123")

        assert req.job_id == "custom-123"

    def test_with_custom_expiry(self):
        """Test submit request with custom expiry."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", expiry_seconds=7200)

        assert req.expiry_seconds == 7200

    def test_empty_input_rejected(self):
        """Test that empty input is rejected."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="")

    def test_image_count_defaults_to_none(self):
        """image_count is optional and absent by default (backend defaults to 3 when images are on)."""
        req = JobSubmitRequest(agent_type="deep_researcher", input="query", include_images=True)

        assert req.image_count is None

    def test_image_count_accepts_valid_range(self):
        """image_count accepts 1 through 4."""
        for count in (1, 2, 3, 4):
            req = JobSubmitRequest(
                agent_type="deep_researcher", input="query", include_images=True, image_count=count
            )
            assert req.image_count == count

    def test_image_count_zero_rejected(self):
        """image_count=0 fails validation (422 at the API boundary)."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="query", include_images=True, image_count=0)

    def test_image_count_five_rejected(self):
        """image_count=5 fails validation (422 at the API boundary)."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="query", include_images=True, image_count=5)

    def test_expiry_too_low_rejected(self):
        """Test that expiry below 600 is rejected."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="query", expiry_seconds=300)

    def test_expiry_too_high_rejected(self):
        """Test that expiry above 604800 is rejected."""
        with pytest.raises(ValueError):
            JobSubmitRequest(agent_type="deep_researcher", input="query", expiry_seconds=700000)


class TestJobStatusResponse:
    """Tests for the JobStatusResponse model."""

    def test_minimal_response(self):
        """Test minimal job response."""
        resp = JobStatusResponse(job_id="123", status="running")

        assert resp.job_id == "123"
        assert resp.status == "running"
        assert resp.error is None
        assert resp.created_at is None
        assert resp.has_report is False
        assert resp.report_ready is False
        assert resp.terminal is False
        assert resp.poll_after_seconds is None
        assert resp.report_url is None

    def test_full_response(self):
        """Test full job response."""
        resp = JobStatusResponse(
            job_id="123",
            status="success",
            error="some error",
            created_at="2026-01-20T10:00:00",
            updated_at="2026-01-20T10:05:00",
            has_report=True,
            report_ready=True,
            terminal=True,
            message="Final report is ready.",
            status_url="/v1/jobs/async/job/123",
            report_url="/v1/jobs/async/job/123/report",
        )

        assert resp.job_id == "123"
        assert resp.status == "success"
        assert resp.error == "some error"
        assert resp.created_at == "2026-01-20T10:00:00"
        assert resp.updated_at == "2026-01-20T10:05:00"
        assert resp.has_report is True
        assert resp.report_ready is True
        assert resp.terminal is True
        assert resp.message == "Final report is ready."
        assert resp.status_url == "/v1/jobs/async/job/123"
        assert resp.report_url == "/v1/jobs/async/job/123/report"


class TestJobStateResponse:
    """Tests for the JobStateResponse model."""

    def test_without_state(self):
        """Test state response without state."""
        resp = JobStateResponse(job_id="123", has_state=False)

        assert resp.job_id == "123"
        assert resp.has_state is False
        assert resp.state is None

    def test_with_artifacts(self):
        """Test state response with artifacts."""
        artifacts = {"tools": [], "outputs": []}
        resp = JobStateResponse(job_id="123", has_state=True, artifacts=artifacts)

        assert resp.has_state is True
        assert resp.artifacts == artifacts


class TestJobReportResponse:
    """Tests for the JobReportResponse model."""

    def test_without_report(self):
        """Test report response without report."""
        resp = JobReportResponse(job_id="123", has_report=False)

        assert resp.job_id == "123"
        assert resp.has_report is False
        assert resp.report_ready is False
        assert resp.terminal is False
        assert resp.report is None
        assert resp.report_markdown is None

    def test_with_report(self):
        """Test report response with report."""
        resp = JobReportResponse(
            job_id="123",
            has_report=True,
            report_ready=True,
            terminal=True,
            report="# Report\n\nContent here",
            report_markdown="# Report\n\nContent here",
            status_url="/v1/jobs/async/job/123",
            report_url="/v1/jobs/async/job/123/report",
            sources_found=10,
            sources_cited=3,
        )

        assert resp.has_report is True
        assert resp.report == "# Report\n\nContent here"
        assert resp.report_markdown == "# Report\n\nContent here"
        assert resp.content_type == "text/markdown"
        assert resp.report_ready is True
        assert resp.terminal is True
        assert resp.status_url == "/v1/jobs/async/job/123"
        assert resp.report_url == "/v1/jobs/async/job/123/report"
        assert resp.sources_found == 10
        assert resp.sources_cited == 3


def test_final_report_helper_rejects_off_topic_report(tmp_path):
    """A completed deep-research job should not expose a drifted final report."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    job_id = "job-off-topic"
    store = EventStore(db_url, job_id)
    store.store(
        {
            "type": "job.submitted",
            "data": {
                "agent_type": "deep_researcher",
                "input": "University education costs and outcomes",
                "owner": "tester@example.com",
                "data_sources": [],
                "research_depth": "deeper",
            },
        }
    )
    job = MagicMock(status="success", output='{"report": "# Alternate Ways to Get to Your Career\\n\\nBody"}')

    assert _get_final_report_for_job(job, db_url, job_id) is None


def test_final_report_helper_rejects_recovered_intermediate_report(tmp_path):
    """A stale recovered-notes payload in job.output should not count as a final synthesis."""
    from aiq_api.jobs.event_store import EventStore

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    job_id = "job-recovered-intermediate"
    store = EventStore(db_url, job_id)
    store.store(
        {
            "type": "job.submitted",
            "data": {
                "agent_type": "deep_researcher",
                "input": "University education value debate for primary students",
                "owner": "tester@example.com",
                "data_sources": [],
                "research_depth": "shallow",
            },
        }
    )
    recovered = (
        "# Recovered Research Report\n\n"
        "The original job collected research artifacts but did not produce `/report.md`. "
        "This report was recovered from the persisted intermediate research files.\n\n"
        "## Recovered Findings\n\n"
        "### shared/university_education_debate_research.txt\n\n"
        "University Education Value Debate - Primary Student Research Notes"
    )
    job = MagicMock(status="success", output=json.dumps({"report": recovered}))

    assert _get_final_report_for_job(job, db_url, job_id) is None


def test_resume_files_strip_shared_route_prefix(tmp_path):
    """Persisted /shared files must resume at routed state keys, not /shared/shared paths."""
    from aiq_api.jobs.event_store import EventStore
    from aiq_api.routes.jobs import _build_resume_files_from_events
    from aiq_api.routes.jobs import _format_resume_input

    db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
    job_id = "job-resume-paths"
    store = EventStore(db_url, job_id)
    store.store(
        {
            "type": "artifact.update",
            "name": "/shared/researcher_task1.md",
            "data": {
                "type": "file",
                "file_path": "/shared/researcher_task1.md",
                "content": "# Notes\n\nForeign policy evidence",
            },
        }
    )

    resume_files = _build_resume_files_from_events(db_url, job_id)
    resume_input = _format_resume_input("Continue the report", resume_files)

    assert "/researcher_task1.md" in resume_files
    assert "/shared/researcher_task1.md" in resume_input
    assert "/shared/shared/researcher_task1.md" not in resume_input


class TestRegisterRoutes:
    """Tests for the register_routes function."""

    @pytest.mark.asyncio
    async def test_routes_not_registered_without_dask(self):
        """Test that routes are not registered when Dask is not available."""
        from aiq_api.routes.jobs import register_job_routes

        mock_app = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_function_config.side_effect = KeyError("Not found")
        mock_worker = MagicMock()
        mock_worker._dask_available = False
        mock_worker._job_store = None

        await register_job_routes(mock_app, mock_builder, mock_worker)

        mock_app.post.assert_not_called()
        assert mock_app.get.call_count == 2

    @pytest.mark.asyncio
    async def test_routes_not_registered_without_job_store(self):
        """Test that routes are not registered without job store."""
        from aiq_api.routes.jobs import register_job_routes

        mock_app = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_function_config.side_effect = KeyError("Not found")
        mock_worker = MagicMock()
        mock_worker._dask_available = True
        mock_worker._job_store = None

        await register_job_routes(mock_app, mock_builder, mock_worker)

        mock_app.post.assert_not_called()
        assert mock_app.get.call_count == 2

    @pytest.mark.asyncio
    async def test_routes_registered_with_dask(self):
        """Test that routes are registered when Dask is available."""
        from aiq_api.routes.jobs import register_job_routes

        mock_app = MagicMock()
        mock_builder = MagicMock()
        mock_builder.get_function_config.side_effect = KeyError("Not found")
        mock_worker = MagicMock()
        mock_worker._dask_available = True
        mock_worker._job_store = MagicMock()
        mock_worker._scheduler_address = "tcp://localhost:8786"
        mock_worker._db_url = "sqlite:///./test.db"
        mock_worker._config_file_path = "/path/to/config.yml"
        mock_worker._log_level = 20
        mock_worker._use_dask_threads = False
        mock_worker._front_end_config = MagicMock(expiry_seconds=86400)

        await register_job_routes(mock_app, mock_builder, mock_worker)

        assert mock_app.post.call_count >= 2
        assert mock_app.get.call_count >= 6


class TestJobLifecycleRecovery:
    """Tests for restart/worker-loss job recovery helpers."""

    def _seed_job_info(self, db_url: str, rows: list[tuple[str, str, int]]) -> None:
        from sqlalchemy import text

        from aiq_api.jobs.event_store import EventStore

        EventStore._ensure_table_exists(db_url)
        engine = EventStore._get_or_create_sync_engine(db_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE job_info ("
                    "job_id TEXT PRIMARY KEY, "
                    "status TEXT, "
                    "error TEXT, "
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                    "is_expired BOOLEAN DEFAULT FALSE"
                    ")"
                )
            )
            for job_id, status, is_expired in rows:
                conn.execute(
                    text("INSERT INTO job_info (job_id, status, is_expired) VALUES (:job_id, :status, :is_expired)"),
                    {"job_id": job_id, "status": status, "is_expired": is_expired},
                )

    def test_find_jobs_by_status_excludes_terminal_and_expired_jobs(self, tmp_path):
        from aiq_api.routes.jobs import _find_jobs_by_status

        db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
        self._seed_job_info(
            db_url,
            [
                ("running-job", "running", 0),
                ("submitted-job", "submitted", 0),
                ("success-job", "success", 0),
                ("expired-running-job", "running", 1),
            ],
        )

        rows = _find_jobs_by_status(db_url, ["submitted", "running"])

        assert {row["job_id"] for row in rows} == {"running-job", "submitted-job"}

    @pytest.mark.asyncio
    async def test_interrupt_orphaned_startup_jobs_marks_active_jobs_recoverable(self, tmp_path):
        import asyncio

        from aiq_api.jobs.event_store import EventStore
        from aiq_api.routes.jobs import _resume_orphaned_startup_jobs

        db_url = f"sqlite:///{tmp_path / 'jobs.db'}"
        self._seed_job_info(
            db_url,
            [
                ("running-job", "running", 0),
                ("submitted-job", "submitted", 0),
                ("success-job", "success", 0),
            ],
        )

        class FakeJobStore:
            def __init__(self) -> None:
                self.calls = []

            async def update_status(self, job_id, status, error=None, output=None):
                self.calls.append((job_id, status.value, error, output))

        job_store = FakeJobStore()

        await _resume_orphaned_startup_jobs(job_store, db_url)
        # Yield control to the event loop so background tasks can execute
        await asyncio.sleep(0.1)

        assert {(job_id, status) for job_id, status, _, _ in job_store.calls} == {
            ("running-job", "interrupted"),
            ("submitted-job", "interrupted"),
        }
        events = EventStore.get_events(db_url, "running-job", 0, 100)
        assert events[-1]["type"] == "job.interrupted"
        assert events[-1]["data"]["error_type"] == "BackendRestartInterrupted"
        assert events[-1]["data"]["recoverable"] is True


class TestArtifactHelpers:
    """Tests for artifact extraction helper functions."""

    def test_extract_event_metadata_with_data(self):
        """Test extracting metadata from event with data dict."""
        from aiq_api.routes.jobs import _extract_event_metadata

        event = {
            "type": "tool.start",
            "data": {"name": "test", "input": "query"},
            "metadata": {"workflow": "agent-1"},
        }

        data, metadata = _extract_event_metadata(event)

        assert data == {"name": "test", "input": "query"}
        assert metadata == {"workflow": "agent-1"}

    def test_extract_event_metadata_fallback_to_nested(self):
        """Test extracting metadata from nested data.metadata."""
        from aiq_api.routes.jobs import _extract_event_metadata

        event = {
            "type": "tool.start",
            "data": {"name": "test", "metadata": {"workflow": "nested"}},
        }

        data, metadata = _extract_event_metadata(event)

        assert metadata == {"workflow": "nested"}

    def test_extract_event_metadata_handles_non_dict(self):
        """Test extracting metadata handles non-dict data."""
        from aiq_api.routes.jobs import _extract_event_metadata

        event = {"type": "test", "data": "string_data"}

        data, metadata = _extract_event_metadata(event)

        assert data == {}
        assert metadata == {}

    def test_process_tool_start(self):
        """Test processing tool.start event."""
        from aiq_api.routes.jobs import _process_tool_start

        event = {"timestamp": "2026-01-22T10:00:00"}
        data = {"id": "tool-1", "name": "search", "data": {"input": "query"}}
        metadata = {"workflow": "agent-1"}
        tool_call_map: dict = {}

        _process_tool_start(event, data, metadata, tool_call_map)

        assert "tool-1" in tool_call_map
        assert tool_call_map["tool-1"]["name"] == "search"
        assert tool_call_map["tool-1"]["status"] == "running"

    def test_process_tool_end_updates_existing(self):
        """Test processing tool.end updates existing tool."""
        from aiq_api.routes.jobs import _process_tool_end

        event = {"timestamp": "2026-01-22T10:00:01"}
        data = {"id": "tool-1", "name": "search", "data": {"output": "result"}}
        metadata = {"workflow": "agent-1"}
        tool_call_map = {
            "tool-1": {
                "id": "tool-1",
                "name": "search",
                "input": "query",
                "output": None,
                "status": "running",
            }
        }

        _process_tool_end(event, data, metadata, tool_call_map)

        assert tool_call_map["tool-1"]["output"] == "result"
        assert tool_call_map["tool-1"]["status"] == "completed"

    def test_process_tool_end_creates_new(self):
        """Test processing tool.end creates new entry if missing."""
        from aiq_api.routes.jobs import _process_tool_end

        event = {"timestamp": "2026-01-22T10:00:01"}
        data = {"id": "tool-2", "name": "other", "data": {"output": "result"}}
        metadata = {"workflow": "agent-1"}
        tool_call_map: dict = {}

        _process_tool_end(event, data, metadata, tool_call_map)

        assert "tool-2" in tool_call_map
        assert tool_call_map["tool-2"]["status"] == "completed"

    def test_process_artifact_update(self):
        """Test processing artifact.update event."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "output.md", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "output", "content": "Report content", "extra": "value"}
        metadata = {"workflow": "agent-1"}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(outputs) == 1
        assert outputs[0]["type"] == "output"
        assert outputs[0]["content"] == "Report content"
        assert outputs[0]["extra"] == "value"

    def test_process_artifact_update_skips_empty_content(self):
        """Test that empty content is skipped."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "empty.md", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "output", "content": None}
        metadata = {}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(outputs) == 0

    def test_process_artifact_update_tracks_citation_source(self):
        """Test that citation_source events are tracked."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "https://example.com", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "citation_source", "content": "https://example.com", "url": "https://example.com"}
        metadata = {}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(sources_found) == 1
        assert "https://example.com" in sources_found
        assert len(sources_cited) == 0

    def test_process_artifact_update_tracks_citation_use(self):
        """Test that citation_use events are tracked."""
        from aiq_api.routes.jobs import _process_artifact_update

        event = {"name": "https://example.com", "timestamp": "2026-01-22T10:00:00"}
        data = {"type": "citation_use", "content": "https://example.com", "url": "https://example.com"}
        metadata = {}
        outputs: list = []
        sources_found: set = set()
        sources_cited: set = set()

        _process_artifact_update(event, data, metadata, outputs, sources_found, sources_cited)

        assert len(sources_cited) == 1
        assert "https://example.com" in sources_cited
        assert len(sources_found) == 0

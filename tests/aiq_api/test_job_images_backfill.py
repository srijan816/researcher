# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the image-backfill endpoint core (POST /v1/jobs/async/job/{id}/images/generate)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from aiq_api.routes.jobs import _backfill_job_images

_DB_URL = "sqlite:///unused.db"
_REPORT = "# Title\n\n## Section One\n\nBody.\n"
_ILLUSTRATED = _REPORT + "\n![Cap](/api/jobs/async/job/j1/images/image-1.jpg)\n"


def _job(status: str = "success", report: str | None = _REPORT, **extra):
    output = json.dumps({"report": report, "quality_status": "pass"}) if report else None
    return SimpleNamespace(status=status, error=None, output=output, output_path=None, **extra)


def _job_store():
    store = SimpleNamespace()
    store.update_status = AsyncMock()
    return store


def _run(job, store, *, report=None, backfill=None):
    """Run _backfill_job_images with the report accessor and backfill helper patched."""
    with (
        patch("aiq_api.routes.jobs._get_final_report_for_job", return_value=report),
        patch("aiq_agent.common.report_images.backfill_report_images", backfill or AsyncMock()),
    ):
        return asyncio.run(_backfill_job_images(store, _DB_URL, "j1", job))


def test_returns_409_when_job_not_terminal():
    with pytest.raises(HTTPException) as exc:
        _run(_job(status="running"), _job_store(), report=_REPORT)
    assert exc.value.status_code == 409
    assert "not finished" in exc.value.detail


def test_returns_409_when_no_report_text_available():
    store = _job_store()
    with pytest.raises(HTTPException) as exc:
        _run(_job(report=None), store, report=None)
    assert exc.value.status_code == 409
    assert "no usable final report" in exc.value.detail
    store.update_status.assert_not_called()


def test_idempotent_when_report_already_has_images():
    store = _job_store()
    backfill = AsyncMock()

    result = _run(_job(report=_ILLUSTRATED), store, report=_ILLUSTRATED, backfill=backfill)

    assert result == {"ok": True, "images_added": 0, "already_had_images": True}
    backfill.assert_not_called()
    store.update_status.assert_not_called()


def test_success_persists_illustrated_report_via_job_store():
    store = _job_store()
    backfill = AsyncMock(return_value=(_ILLUSTRATED, 1))

    result = _run(_job(), store, report=_REPORT, backfill=backfill)

    assert result == {"ok": True, "images_added": 1, "already_had_images": False}
    backfill.assert_awaited_once()
    assert backfill.call_args.kwargs["image_url_prefix"] == "/api/jobs/async/job/j1/images"
    store.update_status.assert_awaited_once()
    call = store.update_status.call_args
    assert call.args[0] == "j1"
    assert call.args[1] == "success"
    persisted_output = call.kwargs["output"]
    assert persisted_output["report"] == _ILLUSTRATED
    # Pre-existing output fields (e.g. quality metadata) are preserved.
    assert persisted_output["quality_status"] == "pass"


def test_no_persistence_when_zero_images_inserted():
    store = _job_store()
    backfill = AsyncMock(return_value=(_REPORT, 0))

    result = _run(_job(), store, report=_REPORT, backfill=backfill)

    assert result == {"ok": True, "images_added": 0, "already_had_images": False}
    store.update_status.assert_not_called()


def test_backfill_errors_surface_as_500():
    from aiq_agent.common.report_images import ReportImagesError
    from aiq_agent.common.report_images import ReportImagesTimeoutError

    store = _job_store()
    with pytest.raises(HTTPException) as exc:
        _run(_job(), store, report=_REPORT, backfill=AsyncMock(side_effect=ReportImagesError("MiniMax down")))
    assert exc.value.status_code == 500
    assert "MiniMax down" in exc.value.detail

    with pytest.raises(HTTPException) as exc:
        _run(
            _job(),
            store,
            report=_REPORT,
            backfill=AsyncMock(side_effect=ReportImagesTimeoutError("exceeded 90s time budget")),
        )
    assert exc.value.status_code == 500
    assert "timed out" in exc.value.detail
    store.update_status.assert_not_called()

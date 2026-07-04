# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pluggable image providers (xAI/MiniMax) and the non-fail-open backfill path."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_agent.common import report_images as ri
from aiq_agent.common.report_images import ReportImagesError
from aiq_agent.common.report_images import ReportImagesTimeoutError
from aiq_agent.common.report_images import ReportImageSpec
from aiq_agent.common.report_images import backfill_report_images
from aiq_agent.common.report_images import plan_image_specs_via_minimax
from aiq_agent.common.report_images import report_contains_generated_images
from aiq_agent.common.report_images import resolve_image_provider

_REPORT = "# Title\n\nIntro.\n\n## Section One\n\nBody text.\n"

_PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("XAI_API_KEY", "AIQ_IMAGE_PROVIDER", "MINIMAX_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _mock_response(payload: dict | None = None, *, content: bytes = b"") -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload or {})
    response.content = content
    return response


def _mock_client(post_response=None, get_response=None, post_side_effect=None) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=post_response, side_effect=post_side_effect)
    client.get = AsyncMock(return_value=get_response)
    return client


# ------------------------------------------------------------ resolve_image_provider


def test_provider_defaults_to_minimax_without_xai_key():
    assert resolve_image_provider() == "minimax"


def test_provider_prefers_xai_when_key_present(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xk")
    assert resolve_image_provider() == "xai"


@pytest.mark.parametrize("forced", ["minimax", "xai"])
def test_provider_env_override_forces_choice(monkeypatch, forced):
    monkeypatch.setenv("XAI_API_KEY", "xk")
    monkeypatch.setenv("AIQ_IMAGE_PROVIDER", forced)
    assert resolve_image_provider() == forced


def test_provider_ignores_unknown_override():
    import os

    os.environ["AIQ_IMAGE_PROVIDER"] = "dall-e"
    try:
        assert resolve_image_provider() == "minimax"
    finally:
        del os.environ["AIQ_IMAGE_PROVIDER"]


# ---------------------------------------------------------------- _generate_xai_image


def test_xai_image_url_flow_downloads_returned_url():
    post = _mock_response({"data": [{"url": "https://img.x.ai/1.png"}]})
    client = _mock_client(post_response=post, get_response=_mock_response(content=_PNG))

    result = asyncio.run(ri._generate_xai_image(client, "a scene", "xk"))

    assert result == _PNG
    _, kwargs = client.post.call_args
    assert kwargs["json"]["model"] == "grok-2-image"
    assert kwargs["json"]["n"] == 1
    assert kwargs["json"]["response_format"] == "url"
    assert kwargs["headers"]["Authorization"] == "Bearer xk"
    assert client.post.call_args[0][0] == "https://api.x.ai/v1/images/generations"


def test_xai_image_b64_flow_decodes_inline_bytes():
    post = _mock_response({"data": [{"b64_json": base64.b64encode(_PNG).decode()}]})
    client = _mock_client(post_response=post)

    assert asyncio.run(ri._generate_xai_image(client, "p", "xk")) == _PNG
    client.get.assert_not_called()


def test_xai_image_raises_on_empty_data():
    client = _mock_client(post_response=_mock_response({"data": []}))
    with pytest.raises(RuntimeError, match="no image data"):
        asyncio.run(ri._generate_xai_image(client, "p", "xk"))


# -------------------------------------------------------------------- _generate_image


def test_generate_image_forced_xai_without_key_raises(monkeypatch):
    monkeypatch.setenv("AIQ_IMAGE_PROVIDER", "xai")
    with pytest.raises(ReportImagesError, match="XAI_API_KEY"):
        asyncio.run(ri._generate_image(_mock_client(), "p", minimax_api_key="mk"))


def test_generate_image_forced_xai_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("AIQ_IMAGE_PROVIDER", "xai")
    monkeypatch.setenv("XAI_API_KEY", "xk")
    client = _mock_client(post_side_effect=RuntimeError("xai down"))
    with pytest.raises(RuntimeError, match="xai down"):
        asyncio.run(ri._generate_image(client, "p", minimax_api_key="mk"))


def test_generate_image_auto_xai_falls_back_to_minimax_on_error(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xk")
    minimax_post = _mock_response({"base_resp": {"status_code": 0}, "data": {"image_urls": ["https://mm/1.jpg"]}})
    client = _mock_client(get_response=_mock_response(content=_PNG))
    client.post = AsyncMock(side_effect=[RuntimeError("xai down"), minimax_post])

    result = asyncio.run(ri._generate_image(client, "p", minimax_api_key="mk"))

    assert result == _PNG
    assert client.post.call_count == 2
    assert client.post.call_args_list[1][0][0] == "https://api.minimax.io/v1/image_generation"


def test_generate_image_minimax_only_without_xai_key():
    minimax_post = _mock_response({"base_resp": {"status_code": 0}, "data": {"image_urls": ["https://mm/1.jpg"]}})
    client = _mock_client(post_response=minimax_post, get_response=_mock_response(content=_PNG))

    assert asyncio.run(ri._generate_image(client, "p", minimax_api_key="mk")) == _PNG
    assert client.post.call_args[0][0] == "https://api.minimax.io/v1/image_generation"


def test_generate_image_minimax_without_key_raises():
    with pytest.raises(ReportImagesError, match="MINIMAX_API_KEY"):
        asyncio.run(ri._generate_image(_mock_client(), "p", minimax_api_key=None))


# --------------------------------------------------- report_contains_generated_images


def test_detects_existing_generated_image_markdown():
    report = _REPORT + "\n![Cap](/api/jobs/async/job/abc-123/images/image-1.jpg)\n"
    assert report_contains_generated_images(report) is True


def test_ignores_reports_without_generated_images():
    assert report_contains_generated_images(_REPORT) is False
    assert report_contains_generated_images("") is False
    assert report_contains_generated_images("![ext](https://example.com/x.png)") is False


# ------------------------------------------------------- plan_image_specs_via_minimax


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient as an async context manager."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self._error:
            raise self._error
        return self._response


def test_planning_call_parses_anthropic_style_response(monkeypatch):
    specs_json = json.dumps([{"after_heading": "Section One", "prompt": "a scene", "caption": "Cap"}])
    response = _mock_response({"content": [{"type": "text", "text": specs_json}]})
    fake = _FakeAsyncClient(response=response)
    monkeypatch.setattr(ri.httpx, "AsyncClient", lambda **kw: fake)

    specs = asyncio.run(plan_image_specs_via_minimax(_REPORT, api_key="mk"))

    assert specs == [ReportImageSpec(after_heading="Section One", prompt="a scene", caption="Cap")]
    url, kwargs = fake.calls[0]
    assert url == "https://api.minimax.io/anthropic/v1/messages"
    assert kwargs["json"]["model"] == "MiniMax-M2.7-highspeed"
    assert kwargs["json"]["max_tokens"] <= 2048
    assert kwargs["headers"]["Authorization"] == "Bearer mk"


def test_planning_call_wraps_http_errors(monkeypatch):
    import httpx

    fake = _FakeAsyncClient(error=httpx.ConnectError("boom"))
    monkeypatch.setattr(ri.httpx, "AsyncClient", lambda **kw: fake)

    with pytest.raises(ReportImagesError, match="planning LLM call failed"):
        asyncio.run(plan_image_specs_via_minimax(_REPORT, api_key="mk"))


# ------------------------------------------------------------- backfill_report_images


def test_backfill_requires_minimax_key():
    with pytest.raises(ReportImagesError, match="MINIMAX_API_KEY"):
        asyncio.run(backfill_report_images(report=_REPORT, job_id="j1", image_url_prefix="/api/x"))


def test_backfill_times_out_with_dedicated_error(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mk")

    async def _slow(**kwargs):
        await asyncio.sleep(5)
        return _REPORT, 0

    monkeypatch.setattr(ri, "_run_backfill", _slow)
    with pytest.raises(ReportImagesTimeoutError, match="time budget"):
        asyncio.run(
            backfill_report_images(report=_REPORT, job_id="j1", image_url_prefix="/api/x", budget_seconds=0.05)
        )


def test_backfill_inserts_images_and_returns_count(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "mk")
    monkeypatch.setenv("AIQ_REPORT_IMAGES_DIR", str(tmp_path))
    spec = ReportImageSpec(after_heading="Section One", prompt="p", caption="Cap")

    async def _plan(report, *, api_key, **kw):
        assert api_key == "mk"
        return [spec]

    minimax_post = _mock_response({"base_resp": {"status_code": 0}, "data": {"image_urls": ["https://mm/1"]}})
    client = _mock_client(post_response=minimax_post, get_response=_mock_response(content=_PNG))
    monkeypatch.setattr(ri, "plan_image_specs_via_minimax", _plan)

    class _ClientCM:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(ri.httpx, "AsyncClient", _ClientCM)

    updated, inserted = asyncio.run(
        backfill_report_images(report=_REPORT, job_id="j1", image_url_prefix="/api/jobs/async/job/j1/images")
    )

    assert inserted == 1
    assert "![Cap](/api/jobs/async/job/j1/images/image-1.png)" in updated
    assert (tmp_path / "j1" / "image-1.png").read_bytes() == _PNG
    assert report_contains_generated_images(updated)


def test_backfill_returns_unchanged_when_planner_picks_nothing(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mk")

    async def _plan(report, *, api_key, **kw):
        return []

    monkeypatch.setattr(ri, "plan_image_specs_via_minimax", _plan)
    updated, inserted = asyncio.run(
        backfill_report_images(report=_REPORT, job_id="j1", image_url_prefix="/api/x")
    )
    assert (updated, inserted) == (_REPORT, 0)


def test_backfill_raises_when_all_generations_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "mk")
    monkeypatch.setenv("AIQ_REPORT_IMAGES_DIR", str(tmp_path))

    async def _plan(report, *, api_key, **kw):
        return [ReportImageSpec(after_heading="Section One", prompt="p", caption="c")]

    async def _no_images(specs, **kw):
        return []

    monkeypatch.setattr(ri, "plan_image_specs_via_minimax", _plan)
    monkeypatch.setattr(ri, "_generate_and_save_images", _no_images)
    with pytest.raises(ReportImagesError, match="All image generations failed"):
        asyncio.run(backfill_report_images(report=_REPORT, job_id="j1", image_url_prefix="/api/x"))

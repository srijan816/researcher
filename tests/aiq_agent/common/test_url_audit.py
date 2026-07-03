# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cited-URL liveness audit. All HTTP is mocked."""

import httpx
import pytest

from aiq_agent.common.url_audit import DEAD_LINK_ANNOTATION
from aiq_agent.common.url_audit import annotate_dead_references
from aiq_agent.common.url_audit import audit_reference_urls
from aiq_agent.common.url_audit import extract_reference_urls
from aiq_agent.common.url_audit import url_audit_enabled

_REPORT = """# Findings

Revenue grew 12% [1] while margins held [2].

## References
[1] Earnings report [primary_issuer]: https://alive.example.com/q3
[2] Analyst note: https://dead.example.com/gone
[3] Archived brief: https://blocked.example.com/head-hostile
"""


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestEnvKnob:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AIQ_URL_AUDIT_ENABLED", raising=False)
        assert url_audit_enabled() is True

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AIQ_URL_AUDIT_ENABLED", "0")
        assert url_audit_enabled() is False


class TestAuditReferenceUrls:
    async def test_200_head_is_alive(self):
        result = await audit_reference_urls(
            ["https://a.example.com/x"],
            transport=_transport(lambda request: httpx.Response(200)),
        )
        assert result == {"https://a.example.com/x": True}

    @pytest.mark.parametrize("status", [404, 410])
    async def test_definitive_gone_status_is_dead(self, status):
        result = await audit_reference_urls(
            ["https://a.example.com/x"],
            transport=_transport(lambda request: httpx.Response(status)),
        )
        assert result == {"https://a.example.com/x": False}

    @pytest.mark.parametrize("status", [500, 503, 401, 429])
    async def test_other_error_statuses_count_as_alive(self, status):
        result = await audit_reference_urls(
            ["https://a.example.com/x"],
            transport=_transport(lambda request: httpx.Response(status)),
        )
        assert result == {"https://a.example.com/x": True}

    @pytest.mark.parametrize("head_status", [403, 405])
    async def test_head_rejection_falls_back_to_get(self, head_status):
        methods: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            return httpx.Response(head_status if request.method == "HEAD" else 200)

        result = await audit_reference_urls(["https://a.example.com/x"], transport=_transport(handler))
        assert result == {"https://a.example.com/x": True}
        assert methods == ["HEAD", "GET"]

    async def test_get_fallback_404_is_dead(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(405 if request.method == "HEAD" else 404)

        result = await audit_reference_urls(["https://a.example.com/x"], transport=_transport(handler))
        assert result == {"https://a.example.com/x": False}

    async def test_network_errors_fail_open_to_alive(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow", request=request)

        result = await audit_reference_urls(["https://a.example.com/x"], transport=_transport(handler))
        assert result == {"https://a.example.com/x": True}

    async def test_disabled_skips_all_requests(self, monkeypatch):
        monkeypatch.setenv("AIQ_URL_AUDIT_ENABLED", "0")

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no requests should be made when disabled")

        result = await audit_reference_urls(["https://a.example.com/x"], transport=_transport(handler))
        assert result == {"https://a.example.com/x": True}

    async def test_non_http_urls_default_to_alive_without_requests(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no requests expected for non-http URLs")

        result = await audit_reference_urls(["ftp://a.example.com/x"], transport=_transport(handler))
        assert result == {"ftp://a.example.com/x": True}

    async def test_empty_input(self):
        assert await audit_reference_urls([]) == {}

    async def test_sends_browser_user_agent(self):
        agents: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            agents.append(request.headers.get("User-Agent", ""))
            return httpx.Response(200)

        await audit_reference_urls(["https://a.example.com/x"], transport=_transport(handler))
        assert agents and agents[0].startswith("Mozilla/5.0")


class TestExtractReferenceUrls:
    def test_extracts_urls_from_reference_lines_only(self):
        urls = extract_reference_urls(_REPORT)
        assert urls == [
            "https://alive.example.com/q3",
            "https://dead.example.com/gone",
            "https://blocked.example.com/head-hostile",
        ]

    def test_no_reference_section(self):
        assert extract_reference_urls("# Report\n\nBody text only.") == []
        assert extract_reference_urls("") == []

    def test_deduplicates_preserving_order(self):
        report = "## References\n[1] A: https://x.example.com/a\n[2] B: https://x.example.com/a\n"
        assert extract_reference_urls(report) == ["https://x.example.com/a"]


class TestAnnotateDeadReferences:
    def test_annotates_only_dead_lines_without_renumbering(self):
        liveness = {
            "https://alive.example.com/q3": True,
            "https://dead.example.com/gone": False,
            "https://blocked.example.com/head-hostile": True,
        }
        annotated, count = annotate_dead_references(_REPORT, liveness)
        assert count == 1
        assert f"https://dead.example.com/gone{DEAD_LINK_ANNOTATION}" in annotated
        assert f"https://alive.example.com/q3{DEAD_LINK_ANNOTATION}" not in annotated
        # Citation numbering and body text untouched.
        assert "[1] Earnings report" in annotated
        assert "[2] Analyst note" in annotated
        assert "Revenue grew 12% [1] while margins held [2]." in annotated

    def test_idempotent(self):
        liveness = {"https://dead.example.com/gone": False}
        once, first_count = annotate_dead_references(_REPORT, liveness)
        twice, second_count = annotate_dead_references(once, liveness)
        assert first_count == 1
        assert second_count == 0
        assert twice == once

    def test_all_alive_returns_report_unchanged(self):
        annotated, count = annotate_dead_references(_REPORT, {"https://alive.example.com/q3": True})
        assert annotated == _REPORT
        assert count == 0

    def test_missing_reference_section_is_a_noop(self):
        report = "No references here."
        annotated, count = annotate_dead_references(report, {"https://dead.example.com/gone": False})
        assert annotated == report
        assert count == 0

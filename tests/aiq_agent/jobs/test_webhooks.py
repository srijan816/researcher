"""Tests for async job terminal webhooks."""

from aiq_api.jobs.webhooks import _signed_headers
from aiq_api.jobs.webhooks import build_job_webhook_config
from aiq_api.jobs.webhooks import build_terminal_payload


def test_build_job_webhook_config_sanitizes_headers():
    config = build_job_webhook_config(
        url="https://example.com/hook",
        headers={"X-Client": "abc", "Bad\nHeader": "nope", "X-Bad": "line\nbreak"},
        secret="secret-value",  # pragma: allowlist secret
    )

    assert config == {
        "url": "https://example.com/hook",
        "headers": {"X-Client": "abc"},
        "secret": "secret-value",  # pragma: allowlist secret
    }


def test_terminal_payload_exposes_fetch_links_without_full_report():
    payload = build_terminal_payload(
        job_id="job-123",
        status="success",
        agent_config_name="deep_researcher",
        agent_class_path="aiq_agent.agents.deep_researcher.agent.DeepResearcherAgent",
        research_depth="deeper",
        has_report=True,
        quality_warnings=["weak sources"],
    )

    assert payload["event"] == "job.terminal"
    assert payload["success"] is True
    assert payload["report_ready"] is True
    assert payload["quality_warnings"] == ["weak sources"]
    assert payload["links"]["report_url"] == "/v1/jobs/async/job/job-123/report"


def test_signed_headers_include_hmac_signature(monkeypatch):
    monkeypatch.setattr("aiq_api.jobs.webhooks.time.time", lambda: 1234567890)

    headers = _signed_headers(  # pragma: allowlist secret
        {"headers": {"X-Client": "abc"}, "secret": "secret-value"},  # pragma: allowlist secret
        b'{"ok":true}',
    )

    assert headers["X-Client"] == "abc"
    assert headers["X-AIQ-Webhook-Timestamp"] == "1234567890"
    assert headers["X-AIQ-Webhook-Signature"].startswith("sha256=")

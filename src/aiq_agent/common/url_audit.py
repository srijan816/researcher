# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cited-URL liveness audit for the final references list.

Citation verification proves each reference resolves to a URL the run actually
captured in the source registry — it does not prove that URL still responds at
publication time. Pages get taken down between scrape and synthesis, and a
report shipping dead links erodes trust in the whole citation chain. This
module probes the final reference URLs and lets the caller annotate dead links
in place, never deleting or renumbering citations (the claim/evidence chain
behind the citation remains valid even when the page has since vanished).

Fail-open philosophy: transient network trouble (timeouts, DNS failures, TLS
errors, bot-blocking) is NOT proof of death, so those URLs count as alive.
Only a definitive 404/410 status marks a URL dead. The audit must never break
or delay report finalization beyond its bounded per-URL timeout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEAD_LINK_ANNOTATION = " [link unavailable as of publication]"

_AUDIT_CONCURRENCY = 8
_AUDIT_TIMEOUT_SECONDS = 6.0
# Definitive "gone" statuses; anything else (including 5xx and bot-blocks) is
# treated as alive because it may be transient or client-specific.
_DEAD_STATUS_CODES = frozenset({404, 410})
# Servers that reject HEAD (405) or bot-filter it (403) often serve a normal
# GET, so retry before trusting the status.
_HEAD_FALLBACK_STATUS_CODES = frozenset({403, 405})
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_REFERENCE_HEADING_RE = re.compile(r"^#{1,6}\s*References\b.*$", re.MULTILINE | re.IGNORECASE)
_REFERENCE_LINE_RE = re.compile(r"^\[\d+\].*$", re.MULTILINE)
_URL_RE = re.compile(r"https?://[^\s\])>]+")
_URL_TRIM_CHARS = ".,;:!?\"'"

_FALSEY = {"0", "false", "no", "off"}


def url_audit_enabled() -> bool:
    """Kill switch: AIQ_URL_AUDIT_ENABLED (default on)."""
    return os.getenv("AIQ_URL_AUDIT_ENABLED", "1").strip().lower() not in _FALSEY


async def _check_url(client: httpx.AsyncClient, url: str) -> bool:
    """Probe one URL. True unless the server definitively says it is gone."""
    try:
        response = await client.head(url)
        if response.status_code in _HEAD_FALLBACK_STATUS_CODES:
            # Streaming GET reads only the status line + headers, not the body.
            async with client.stream("GET", url) as get_response:
                return get_response.status_code not in _DEAD_STATUS_CODES
        return response.status_code not in _DEAD_STATUS_CODES
    except Exception:  # noqa: BLE001 - network trouble is not proof of death
        return True


async def audit_reference_urls(urls: list[str], *, transport: Any | None = None) -> dict[str, bool]:
    """Check liveness of cited URLs. Returns {url: alive} for every input URL.

    Bounded concurrency, per-URL timeout, and fully fail-open: on any error
    (or when disabled via AIQ_URL_AUDIT_ENABLED=0) every URL reports alive so
    the report ships unannotated rather than falsely flagged.

    ``transport`` is a test hook (httpx.MockTransport); production callers
    omit it.
    """
    results: dict[str, bool] = {str(url): True for url in urls if url}
    if not results or not url_audit_enabled():
        return results
    candidates = [url for url in results if url.lower().startswith(("http://", "https://"))]
    if not candidates:
        return results

    try:
        semaphore = asyncio.Semaphore(_AUDIT_CONCURRENCY)
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(_AUDIT_TIMEOUT_SECONDS),
            headers={"User-Agent": _BROWSER_USER_AGENT},
            transport=transport,
        )

        async def _bounded(url: str) -> tuple[str, bool]:
            async with semaphore:
                return url, await _check_url(client, url)

        async with client:
            checked = await asyncio.gather(*(_bounded(url) for url in candidates))
        results.update(dict(checked))
        dead_count = sum(1 for alive in results.values() if not alive)
        logger.info("aiq.url_audit audited=%d dead=%d", len(candidates), dead_count)
    except Exception:  # noqa: BLE001 - fail-open: treat everything as alive
        logger.debug("URL liveness audit failed; treating all URLs as alive", exc_info=True)
    return results


def _reference_section_start(report_text: str) -> int | None:
    """Offset of the last References heading, or None when absent."""
    start: int | None = None
    for match in _REFERENCE_HEADING_RE.finditer(report_text):
        start = match.start()
    return start


def extract_reference_urls(report_text: str) -> list[str]:
    """Pull unique URLs from the reference lines of the final References section."""
    start = _reference_section_start(str(report_text or ""))
    if start is None:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for line_match in _REFERENCE_LINE_RE.finditer(report_text[start:]):
        for url_match in _URL_RE.finditer(line_match.group(0)):
            url = url_match.group(0).rstrip(_URL_TRIM_CHARS)
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
    return ordered


def annotate_dead_references(report_text: str, liveness: dict[str, bool]) -> tuple[str, int]:
    """Append the dead-link annotation to reference lines whose URL is dead.

    Citations are never removed or renumbered — the reference stays in place
    so inline [n] markers keep resolving; readers just learn the link died.
    Idempotent: already-annotated lines are left untouched.
    """
    text = str(report_text or "")
    dead_urls = {url for url, alive in (liveness or {}).items() if not alive}
    start = _reference_section_start(text)
    if not dead_urls or start is None:
        return text, 0

    annotated_count = 0

    def _annotate(match: re.Match[str]) -> str:
        nonlocal annotated_count
        line = match.group(0)
        if DEAD_LINK_ANNOTATION in line:
            return line
        line_urls = {url_match.group(0).rstrip(_URL_TRIM_CHARS) for url_match in _URL_RE.finditer(line)}
        if line_urls & dead_urls:
            annotated_count += 1
            return line.rstrip() + DEAD_LINK_ANNOTATION
        return line

    section = _REFERENCE_LINE_RE.sub(_annotate, text[start:])
    return text[:start] + section, annotated_count

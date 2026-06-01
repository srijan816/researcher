# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable storage for fetched web documents."""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
import re
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .source_classification import SourceClass
from .source_classification import classify_source

_DOC_RE = re.compile(r"<document\b[^>]*>(.*?)</document>", re.DOTALL | re.IGNORECASE)
_EXA_DOC_RE = re.compile(
    r"<Document\b[^>]*href=[\"'](?P<url>https?://.*?)[\"'][^>]*>(?P<body>.*?)</Document>",
    re.DOTALL,
)
_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"<url>\s*(https?://.*?)\s*</url>", re.DOTALL | re.IGNORECASE)
_CONTENT_RE = re.compile(r"<content>\s*(.*?)\s*</content>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class ScrapeArtifact:
    """Metadata for a persisted fetched document."""

    job_id: str
    url: str
    title: str | None
    extraction_status: str
    content_hash: str
    researcher: str | None
    tool: str
    source_class: SourceClass
    artifact_path: str
    content_length: int
    created_at: str


def scrape_artifact_root() -> Path:
    """Return the artifact root directory."""
    return Path(os.environ.get("AIQ_SCRAPE_ARTIFACT_DIR", "./data/scrape_artifacts"))


def persist_scrape_artifacts(
    *,
    job_id: str | None,
    tool_name: str,
    tool_output: str,
    researcher: str | None = None,
) -> list[ScrapeArtifact]:
    """Persist every document-like search result as compressed JSON."""
    if not isinstance(job_id, str) or not job_id.strip() or not tool_output.strip():
        return []

    artifacts = []
    normalized_job_id = job_id.strip()
    target_dir = scrape_artifact_root() / _safe_segment(normalized_job_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    for document in _extract_documents(tool_output):
        url = document.get("url", "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        content = document.get("content", "").strip()
        title = document.get("title") or None
        source_class = classify_source(url)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        host = _safe_segment(urlparse(url).netloc or "document")
        filename = f"{host}-{url_hash}-{content_hash[:16]}.json.gz"
        artifact_path = target_dir / filename
        created_at = datetime.now(UTC).isoformat()
        extraction_status = "extracted" if content else "metadata_only"
        payload = {
            "job_id": normalized_job_id,
            "url": url,
            "title": title,
            "extraction_status": extraction_status,
            "content_hash": content_hash,
            "researcher": researcher,
            "tool": tool_name,
            "source_class": source_class,
            "content": content,
            "content_length": len(content),
            "created_at": created_at,
        }
        with gzip.open(artifact_path, "wt", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
        artifacts.append(
            ScrapeArtifact(
                job_id=normalized_job_id,
                url=url,
                title=title,
                extraction_status=extraction_status,
                content_hash=content_hash,
                researcher=researcher,
                tool=tool_name,
                source_class=source_class,
                artifact_path=str(artifact_path),
                content_length=len(content),
                created_at=created_at,
            )
        )

    return artifacts


def _extract_documents(tool_output: str) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for match in _DOC_RE.finditer(tool_output):
        block = match.group(1)
        url = _first_group(_URL_RE, block)
        if not url:
            continue
        title = _clean_text(_first_group(_TITLE_RE, block))
        content = _clean_text(_first_group(_CONTENT_RE, block) or _strip_tags(block))
        documents.append({"url": html.unescape(url), "title": title, "content": content})

    for match in _EXA_DOC_RE.finditer(tool_output):
        body = match.group("body")
        title = _clean_text(_first_group(_TITLE_RE, body))
        content = _clean_text(_TITLE_RE.sub("", body))
        documents.append({"url": html.unescape(match.group("url")), "title": title, "content": content})

    return documents


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return html.unescape(match.group(1).strip()) if match else None


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text)


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(_strip_tags(text))).strip()


def _safe_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:120] or "artifact"


def artifact_event_payload(artifact: ScrapeArtifact) -> dict[str, str | int | None]:
    """Return a compact event payload for a stored scrape artifact."""
    return asdict(artifact)

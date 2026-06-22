# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable storage for fetched web documents."""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .source_classification import SourceClass
from .source_classification import classify_source

logger = logging.getLogger(__name__)

_DOC_RE = re.compile(r"<document\b[^>]*>(.*?)</document>", re.DOTALL | re.IGNORECASE)
_EXA_DOC_RE = re.compile(
    r"<Document\b[^>]*href=[\"'](?P<url>https?://.*?)[\"'][^>]*>(?P<body>.*?)</Document>",
    re.DOTALL,
)
_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"<url>\s*(https?://.*?)\s*</url>", re.DOTALL | re.IGNORECASE)
_CONTENT_RE = re.compile(r"<content>\s*(.*?)\s*</content>", re.DOTALL | re.IGNORECASE)
_PUBLISHED_AT_RE = re.compile(r"<published_at>\s*(.*?)\s*</published_at>", re.DOTALL | re.IGNORECASE)
_EXTRACTION_QUALITY_RE = re.compile(r"<extraction_quality>\s*(.*?)\s*</extraction_quality>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

_INDEX_FILENAME = "artifacts_index.db"
_index_lock = threading.Lock()
_backfilled_roots: set[str] = set()


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
    published_at: str | None = None
    extraction_quality: str | None = None


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
        published_at = document.get("published_at") or None
        extraction_quality = document.get("extraction_quality") or None
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
            "published_at": published_at,
            "extraction_quality": extraction_quality,
        }
        with gzip.open(artifact_path, "wt", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False)
        _index_artifact_entry(
            url=url,
            path=str(artifact_path),
            content_hash=content_hash,
            extraction_status=extraction_status,
            published_at=published_at,
            created_at=created_at,
        )
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
                published_at=published_at,
                extraction_quality=extraction_quality,
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
        published_at = _first_group(_PUBLISHED_AT_RE, block) or ""
        extraction_quality = _first_group(_EXTRACTION_QUALITY_RE, block) or ""
        documents.append(
            {
                "url": html.unescape(url),
                "title": title,
                "content": content,
                "published_at": published_at,
                "extraction_quality": extraction_quality,
            }
        )

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


# ---------------------------------------------------------------------------
# Durable scrape cache index (read path for previously persisted artifacts).
#
# Every sqlite/file error below degrades to a cache miss; the cache must never
# raise into the search or persistence paths.
# ---------------------------------------------------------------------------


def _index_path() -> Path:
    return scrape_artifact_root() / _INDEX_FILENAME


def _normalize_scrape_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _cache_url_hash(url: str) -> str:
    return hashlib.sha256(_normalize_scrape_url(url).encode("utf-8")).hexdigest()[:32]


def _open_index(*, create: bool = True) -> sqlite3.Connection | None:
    """Open the artifact lookup index, creating it when requested. None on failure."""
    try:
        root = scrape_artifact_root()
        if create:
            root.mkdir(parents=True, exist_ok=True)
        elif not _index_path().exists():
            return None
        conn = sqlite3.connect(str(_index_path()), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                url_hash TEXT PRIMARY KEY,
                normalized_url TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT,
                extraction_status TEXT,
                published_at TEXT,
                created_at TEXT
            )
            """
        )
        return conn
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Scrape artifact index unavailable: %s", exc)
        return None


def _upsert_index_row(
    conn: sqlite3.Connection,
    *,
    url: str,
    path: str,
    content_hash: str | None,
    extraction_status: str | None,
    published_at: str | None,
    created_at: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO artifacts (url_hash, normalized_url, path, content_hash,
                               extraction_status, published_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url_hash) DO UPDATE SET
            normalized_url = excluded.normalized_url,
            path = excluded.path,
            content_hash = excluded.content_hash,
            extraction_status = excluded.extraction_status,
            published_at = excluded.published_at,
            created_at = excluded.created_at
        WHERE excluded.created_at >= artifacts.created_at
        """,
        (
            _cache_url_hash(url),
            _normalize_scrape_url(url),
            path,
            content_hash,
            extraction_status,
            published_at,
            created_at or "",
        ),
    )


def _index_artifact_entry(
    *,
    url: str,
    path: str,
    content_hash: str | None,
    extraction_status: str | None,
    published_at: str | None,
    created_at: str | None,
) -> None:
    """Record one artifact in the lookup index. Errors degrade to a no-op."""
    try:
        with _index_lock:
            conn = _open_index(create=True)
            if conn is None:
                return
            try:
                _upsert_index_row(
                    conn,
                    url=url,
                    path=path,
                    content_hash=content_hash,
                    extraction_status=extraction_status,
                    published_at=published_at,
                    created_at=created_at,
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("Failed to index scrape artifact for %s: %s", url, exc)


def scan_existing() -> int:
    """Backfill the lookup index from artifact files already on disk.

    Returns the number of artifact files indexed. Errors on individual files
    are skipped; a completely unavailable index returns 0.
    """
    indexed = 0
    try:
        with _index_lock:
            conn = _open_index(create=True)
            if conn is None:
                return 0
            try:
                for artifact_file in sorted(scrape_artifact_root().glob("*/*.json.gz")):
                    try:
                        with gzip.open(artifact_file, "rt", encoding="utf-8") as fp:
                            payload = json.load(fp)
                        if not isinstance(payload, dict):
                            continue
                        url = str(payload.get("url") or "").strip()
                        if not url:
                            continue
                        _upsert_index_row(
                            conn,
                            url=url,
                            path=str(artifact_file),
                            content_hash=payload.get("content_hash"),
                            extraction_status=payload.get("extraction_status"),
                            published_at=payload.get("published_at"),
                            created_at=payload.get("created_at"),
                        )
                        indexed += 1
                    except Exception:
                        continue
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.debug("Scrape artifact index backfill failed: %s", exc)
        return indexed
    return indexed


def _ensure_index_backfilled() -> None:
    """Lazily backfill the index once per artifact root when missing or empty."""
    root_key = str(scrape_artifact_root())
    if root_key in _backfilled_roots:
        return
    needs_scan = True
    try:
        with _index_lock:
            conn = _open_index(create=False)
            if conn is not None:
                try:
                    row = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()
                    needs_scan = not row or int(row[0] or 0) == 0
                except Exception:
                    needs_scan = True
                finally:
                    conn.close()
    except Exception:
        needs_scan = True
    if needs_scan:
        scan_existing()
    _backfilled_roots.add(root_key)


def find_cached_scrape(url: str, max_age_seconds: float) -> dict | None:
    """Return the freshest stored artifact payload for ``url`` if within TTL.

    Returns ``None`` on miss, stale entries, missing files, or any storage
    error. Never raises.
    """
    try:
        normalized = _normalize_scrape_url(url)
        if not normalized:
            return None
        _ensure_index_backfilled()
        conn = _open_index(create=False)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT path, created_at FROM artifacts WHERE url_hash = ?",
                (_cache_url_hash(url),),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        path, created_at = row
        created = datetime.fromisoformat(str(created_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_seconds = (datetime.now(UTC) - created).total_seconds()
        if age_seconds > max(float(max_age_seconds), 0.0):
            return None
        artifact_file = Path(str(path))
        if not artifact_file.is_file():
            return None
        with gzip.open(artifact_file, "rt", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.debug("Scrape cache lookup failed for %s: %s", url, exc)
        return None

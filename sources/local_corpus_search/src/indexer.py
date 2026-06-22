# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingest persisted scrape artifacts into the SQLite local corpus index.

Artifacts are the gzip JSON files written by
``src/aiq_agent/common/scrape_artifacts.py`` under ``AIQ_SCRAPE_ARTIFACT_DIR``
(default ``./data/scrape_artifacts``) at
``{job_id}/{host}-{urlhash}-{contenthash}.json.gz``.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sqlite3
import time
from array import array
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingError
from .embeddings import get_default_embedder

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./data/corpus_index.db"
DEFAULT_ARTIFACT_DIR = "./data/scrape_artifacts"

MIN_CONTENT_WORDS = 120
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

_ACCEPTED_STATUSES = {"extracted", "full"}

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Marker object: "resolve the default embedder from the environment".
_AUTO = object()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    title TEXT,
    source_class TEXT,
    published_at TEXT,
    content_hash TEXT NOT NULL UNIQUE,
    job_id TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


def corpus_db_path() -> str:
    """Return the corpus DB path from AIQ_CORPUS_DB or the default."""
    return os.environ.get("AIQ_CORPUS_DB", DEFAULT_DB_PATH)


def artifact_root() -> str:
    """Return the scrape artifact root from AIQ_SCRAPE_ARTIFACT_DIR or the default."""
    return os.environ.get("AIQ_SCRAPE_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the corpus DB in WAL mode, creating schema as needed."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def encode_embedding(vector: list[float]) -> bytes:
    """Serialize an embedding vector as little-endian float32 bytes."""
    return array("f", vector).tobytes()


def decode_embedding(blob: bytes) -> list[float]:
    """Deserialize float32 bytes back into a list of floats."""
    values = array("f")
    values.frombytes(blob)
    return list(values)


def chunk_text(text: str, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into ~chunk_size character chunks with ~overlap overlap.

    Prefers paragraph boundaries, then sentence boundaries, then word
    boundaries. Returns at least one chunk for non-empty input.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    units: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_split_long_unit(sentence, chunk_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = f"{_overlap_tail(current, overlap)} {unit}".strip()
            if len(current) > chunk_size:
                chunks.append(current)
                current = ""
        else:
            chunks.append(unit)
            current = ""
    if current:
        chunks.append(current)
    return chunks or [text[:chunk_size]]


def _split_long_unit(unit: str, chunk_size: int) -> list[str]:
    """Hard-split an oversized sentence on word boundaries."""
    words = unit.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = word[:chunk_size]
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """Return the trailing ~overlap characters of ``text`` at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut != -1 else tail


def _load_artifact(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Skipping unreadable artifact %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def ingest(
    artifact_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    *,
    embedder: Any = _AUTO,
    min_words: int = MIN_CONTENT_WORDS,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Scan artifacts and index new documents. Returns a stats dict.

    ``embedder`` may be an object with ``embed(texts, input_type=...)`` and a
    ``model`` attribute, ``None`` to force lexical mode, or left unset to
    resolve the default NIM client from the environment (lexical when
    ``NVIDIA_API_KEY`` is missing).
    """
    started = time.monotonic()
    root = Path(artifact_dir) if artifact_dir is not None else Path(artifact_root())
    resolved_db = Path(db_path) if db_path is not None else Path(corpus_db_path())
    if embedder is _AUTO:
        embedder = get_default_embedder()

    conn = connect(resolved_db)
    try:
        stored_mode = get_meta(conn, "index_mode")
        if stored_mode is None:
            mode = "embedding" if embedder is not None else "lexical"
            set_meta(conn, "index_mode", mode)
            set_meta(conn, "embed_model", getattr(embedder, "model", "") or "")
            conn.commit()
        else:
            mode = stored_mode
            if mode == "lexical" and embedder is not None:
                logger.warning(
                    "Corpus index at %s was built in lexical mode; ignoring available embedder. "
                    "Rebuild with scripts/build_corpus_index.py --rebuild to switch to embeddings.",
                    resolved_db,
                )
                embedder = None
            elif mode == "embedding" and embedder is None:
                logger.warning(
                    "Corpus index at %s is embedding-mode but no embedder is available; "
                    "new chunks will be stored without embeddings (BM25 still covers them).",
                    resolved_db,
                )
        if mode == "lexical":
            embedder = None

        existing_hashes = {row[0] for row in conn.execute("SELECT content_hash FROM docs")}
        stats: dict[str, Any] = {
            "mode": mode,
            "embed_model": get_meta(conn, "embed_model") or "",
            "artifacts_scanned": 0,
            "docs_added": 0,
            "docs_skipped_existing": 0,
            "docs_skipped_quality": 0,
            "chunks_added": 0,
            "chunks_without_embedding": 0,
        }

        if not root.exists():
            logger.warning("Artifact directory %s does not exist; nothing to ingest", root)
            stats["duration_seconds"] = round(time.monotonic() - started, 3)
            return stats

        for path in sorted(root.rglob("*.json.gz")):
            stats["artifacts_scanned"] += 1
            payload = _load_artifact(path)
            if payload is None:
                stats["docs_skipped_quality"] += 1
                continue
            status = str(payload.get("extraction_status") or "")
            content = str(payload.get("content") or "").strip()
            content_hash = str(payload.get("content_hash") or "").strip()
            url = str(payload.get("url") or "").strip()
            if status not in _ACCEPTED_STATUSES or not content or not content_hash or not url:
                stats["docs_skipped_quality"] += 1
                continue
            if len(content.split()) < min_words:
                stats["docs_skipped_quality"] += 1
                continue
            if content_hash in existing_hashes:
                stats["docs_skipped_existing"] += 1
                continue

            chunks = chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                stats["docs_skipped_quality"] += 1
                continue

            vectors: list[list[float] | None] = [None] * len(chunks)
            if embedder is not None:
                try:
                    vectors = list(embedder.embed(chunks, input_type="passage"))
                except EmbeddingError as exc:
                    logger.warning("Embedding failed for %s; storing without vectors: %s", url, exc)
                    vectors = [None] * len(chunks)

            cursor = conn.execute(
                "INSERT INTO docs(url, title, source_class, published_at, content_hash, job_id, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    payload.get("title"),
                    payload.get("source_class"),
                    payload.get("published_at"),
                    content_hash,
                    payload.get("job_id"),
                    payload.get("created_at"),
                ),
            )
            doc_id = cursor.lastrowid
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                blob = encode_embedding(vector) if vector is not None else None
                if blob is None and mode == "embedding":
                    stats["chunks_without_embedding"] += 1
                conn.execute(
                    "INSERT INTO chunks(doc_id, idx, text, embedding) VALUES(?, ?, ?, ?)",
                    (doc_id, idx, chunk, blob),
                )
            conn.commit()
            existing_hashes.add(content_hash)
            stats["docs_added"] += 1
            stats["chunks_added"] += len(chunks)

        stats["duration_seconds"] = round(time.monotonic() - started, 3)
        return stats
    finally:
        conn.close()


def backfill_embeddings(
    db_path: str | Path | None = None,
    *,
    embedder: Any = _AUTO,
    batch_size: int = 64,
    commit_every: int = 10,
) -> dict[str, Any]:
    """Embed chunks whose embedding is NULL in an embedding-mode index.

    Used after an embedding endpoint outage (or model EOL) left chunks stored
    without vectors. Updates the ``embed_model`` meta key to the active model
    so query-time embeddings stay consistent with the stored vectors.
    """
    started = time.monotonic()
    resolved_db = Path(db_path) if db_path is not None else Path(corpus_db_path())
    if embedder is _AUTO:
        embedder = get_default_embedder()
    if embedder is None:
        raise EmbeddingError("No embedder available (NVIDIA_API_KEY missing); cannot backfill")

    conn = connect(resolved_db)
    stats: dict[str, Any] = {"chunks_embedded": 0, "chunks_failed": 0, "batches": 0}
    try:
        mode = get_meta(conn, "index_mode")
        if mode != "embedding":
            raise ValueError(
                f"Corpus index at {resolved_db} is {mode!r} mode; backfill only applies to "
                "embedding-mode indexes. Use rebuild() to switch modes."
            )
        set_meta(conn, "embed_model", getattr(embedder, "model", "") or "")
        conn.commit()

        # Snapshot the pending IDs up front so a failed batch does not get
        # re-selected forever, and a single transient embedding error (e.g. a
        # NIM read timeout) only loses that batch instead of aborting the whole
        # run. Remaining failures are simply retried on the next invocation.
        pending_ids = [row[0] for row in conn.execute(
            "SELECT id FROM chunks WHERE embedding IS NULL ORDER BY id"
        )]
        total = len(pending_ids)
        stats["chunks_pending_initial"] = total
        since_commit = 0
        for start in range(0, total, batch_size):
            batch_ids = pending_ids[start : start + batch_size]
            placeholders = ",".join("?" * len(batch_ids))
            id_to_text = {
                row[0]: row[1]
                for row in conn.execute(
                    f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", batch_ids
                )
            }
            ordered = [(cid, id_to_text[cid]) for cid in batch_ids if cid in id_to_text]
            if not ordered:
                continue
            try:
                vectors = embedder.embed([text for _, text in ordered], input_type="passage")
            except EmbeddingError as exc:
                stats["chunks_failed"] += len(ordered)
                conn.commit()
                logger.warning(
                    "Backfill batch failed (%d chunks skipped, retry on next run): %s",
                    len(ordered),
                    exc,
                )
                continue
            for (chunk_id, _), vector in zip(ordered, vectors, strict=True):
                conn.execute(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    (encode_embedding(vector), chunk_id),
                )
            stats["chunks_embedded"] += len(ordered)
            stats["batches"] += 1
            since_commit += 1
            if since_commit >= commit_every:
                conn.commit()
                since_commit = 0
                logger.info(
                    "Backfill progress: %d/%d chunks embedded (%d failed)",
                    stats["chunks_embedded"],
                    total,
                    stats["chunks_failed"],
                )
        conn.commit()
        stats["chunks_pending_remaining"] = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL"
        ).fetchone()[0]
        stats["duration_seconds"] = round(time.monotonic() - started, 3)
        return stats
    finally:
        conn.close()


def rebuild(
    artifact_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    *,
    embedder: Any = _AUTO,
) -> dict[str, Any]:
    """Drop all indexed data and re-ingest from scratch."""
    resolved_db = Path(db_path) if db_path is not None else Path(corpus_db_path())
    conn = connect(resolved_db)
    try:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM docs")
        conn.execute("DELETE FROM meta")
        conn.commit()
    finally:
        conn.close()
    return ingest(artifact_dir, resolved_db, embedder=embedder)

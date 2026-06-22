# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""zvec-backed approximate-nearest-neighbour index over the corpus embeddings.

SQLite (``corpus_index.db``) remains the durable content/metadata store and the
write path. This module maintains a derived `zvec <https://github.com/alibaba/zvec>`_
HNSW index that makes dense retrieval ~80x faster than the in-memory full scan
while removing the per-query multi-hundred-MB matrix load.

Everything here is import-guarded and fail-soft: if ``zvec`` is not installed or
an index cannot be opened, callers fall back to the SQLite numpy/BM25 path in
``core.py``. The index is a rebuildable cache, never the source of truth.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from array import array
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# zvec write batches are capped at 1024 documents per Insert call.
_INSERT_BATCH = 1000
_warned: set[str] = set()
# Cache of opened read handles keyed by resolved index path (per process).
_open_collections: dict[str, Any] = {}


def _warn_once(key: str, message: str, *args: object) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def zvec_available() -> bool:
    """Return True when the zvec extension can be imported."""
    try:
        import zvec  # noqa: F401
    except Exception:  # noqa: BLE001 - any import/abi failure means "unavailable"
        return False
    return True


def index_path_for(db_path: str | Path) -> Path:
    """Derive the zvec index directory next to the SQLite corpus DB."""
    db = Path(db_path).expanduser()
    return db.with_suffix(".zvec")


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _get_collection(path: Path, *, schema: Any = None) -> Any:
    """Return a cached per-process zvec handle, opening or creating as needed.

    A single handle per path is reused for both writes (build/sync) and reads
    (search) within one process — which is the supported pattern. Returns None
    when the index does not exist and no schema was supplied to create it.
    """
    import zvec

    key = str(path)
    col = _open_collections.get(key)
    if col is not None:
        return col
    if path.exists():
        col = zvec.open(path=key)
    elif schema is not None:
        col = zvec.create_and_open(path=key, schema=schema)
    else:
        return None
    _open_collections[key] = col
    return col


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    """Track which chunk ids have been pushed into the zvec index (for incremental sync)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vector_index_state (chunk_id INTEGER PRIMARY KEY)"
    )


def _detect_dimension(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1"
    ).fetchone()
    if not row or row[0] is None:
        return None
    return len(array("f", row[0]))


def build_or_sync(
    conn: sqlite3.Connection,
    db_path: str | Path,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Upsert embedded chunks from SQLite into the zvec HNSW index.

    Incremental by default: only chunks not yet recorded in ``vector_index_state``
    are pushed. ``rebuild=True`` drops the index and the state and reindexes
    everything (use when the embedding model / dimension changed).

    Returns a stats dict. Never raises for zvec-availability reasons — it returns
    ``{"available": False}`` so callers can keep using the SQLite path.
    """
    stats: dict[str, Any] = {"available": False, "synced": 0, "skipped_existing": 0}
    if not zvec_available():
        _warn_once("unavailable", "zvec not installed; skipping vector index build (SQLite path still works)")
        return stats

    import shutil

    import zvec

    stats["available"] = True
    path = index_path_for(db_path)
    _ensure_state_table(conn)

    if rebuild:
        if path.exists():
            shutil.rmtree(path)
        conn.execute("DELETE FROM vector_index_state")
        conn.commit()
        _open_collections.pop(str(path), None)

    dim = _detect_dimension(conn)
    if not dim:
        logger.info("No embedded chunks to index yet; vector index not built")
        return stats
    stats["dimension"] = dim

    schema = zvec.CollectionSchema(
        name="corpus",
        fields=[
            zvec.FieldSchema("doc_id", zvec.DataType.INT64),
            zvec.FieldSchema("source_class", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("published_at", zvec.DataType.STRING, nullable=True),
        ],
        vectors=zvec.VectorSchema(
            "embedding", zvec.DataType.VECTOR_FP32, dim, index_param=zvec.HnswIndexParam()
        ),
    )
    col = _get_collection(path, schema=schema)

    pending = conn.execute(
        "SELECT c.id, c.doc_id, c.embedding, d.source_class, d.published_at "
        "FROM chunks c JOIN docs d ON d.id = c.doc_id "
        "WHERE c.embedding IS NOT NULL "
        "AND c.id NOT IN (SELECT chunk_id FROM vector_index_state) "
        "ORDER BY c.id"
    ).fetchall()

    batch: list[Any] = []
    synced_ids: list[int] = []

    def _flush() -> None:
        if not batch:
            return
        col.upsert(batch)
        conn.executemany(
            "INSERT OR IGNORE INTO vector_index_state(chunk_id) VALUES(?)",
            [(cid,) for cid in synced_ids],
        )
        conn.commit()
        batch.clear()
        synced_ids.clear()

    for cid, doc_id, emb, sclass, pub in pending:
        vec = _normalize(list(array("f", emb)))
        if len(vec) != dim:
            stats["skipped_existing"] += 1
            continue
        batch.append(
            zvec.Doc(
                id=str(cid),
                vectors={"embedding": vec},
                fields={"doc_id": int(doc_id), "source_class": sclass or "", "published_at": pub or ""},
            )
        )
        synced_ids.append(cid)
        stats["synced"] += 1
        if len(batch) >= _INSERT_BATCH:
            _flush()
    _flush()
    try:
        col.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("zvec flush warning: %s", exc)
    return stats


def search_ids(
    db_path: str | Path,
    query_vector: list[float],
    *,
    top_k: int,
    filter_expr: str | None = None,
) -> list[tuple[int, float]] | None:
    """Return ``(chunk_id, score)`` for the top matches, or None to signal fallback.

    None means "zvec could not serve this query" (not installed, no index, or a
    runtime error) — callers should fall back to the SQLite path.
    """
    if not zvec_available():
        return None
    path = index_path_for(db_path)
    if not path.exists():
        return None
    try:
        import zvec

        col = _get_collection(path)
        if col is None:
            return None
        results = col.query(
            zvec.Query("embedding", vector=_normalize(query_vector)),
            topk=top_k,
            filter=filter_expr,
            output_fields=["doc_id"],
        )
        return [(int(doc.id), float(doc.score if doc.score is not None else 0.0)) for doc in results]
    except Exception as exc:  # noqa: BLE001 - any zvec error => fall back to SQLite
        _warn_once("query_error", "zvec query failed (%s); falling back to SQLite scan", exc)
        _open_collections.pop(str(path), None)
        return None

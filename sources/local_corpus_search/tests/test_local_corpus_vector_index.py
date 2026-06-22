# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the zvec ANN vector index and its integration with core.search."""

from __future__ import annotations

import pytest
from local_corpus_search import core
from local_corpus_search import vector_index
from local_corpus_search.indexer import connect
from local_corpus_search.indexer import ingest

zvec_only = pytest.mark.skipif(
    not vector_index.zvec_available(), reason="zvec not installed in this environment"
)


@pytest.fixture(autouse=True)
def _clear_zvec_cache():
    vector_index._open_collections.clear()
    vector_index._warned.clear()
    yield
    vector_index._open_collections.clear()


def _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    """Ingest three topically distinct docs in embedding mode; return (db, embedder)."""
    artifacts = tmp_path / "artifacts"
    db = tmp_path / "corpus.db"
    for topic, sclass in [("nvidia", "news"), ("banana", "blog"), ("reasoning", "news")]:
        artifact_writer(
            artifacts,
            url=f"https://ex.com/{topic}",
            content=words_maker([topic], 200),
            source_class=sclass,
        )
    embedder = fake_embedder_factory(["nvidia", "banana", "reasoning"])
    stats = ingest(artifacts, db, embedder=embedder)
    assert stats["mode"] == "embedding"
    return db, embedder


def test_zvec_available_returns_bool():
    assert isinstance(vector_index.zvec_available(), bool)


def test_search_ids_returns_none_without_index(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    db, embedder = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    # No build_or_sync called → no .zvec index on disk → fallback signal.
    qv = embedder.embed(["nvidia"], input_type="query")[0]
    assert vector_index.search_ids(db, qv, top_k=5) is None


@zvec_only
def test_build_or_sync_indexes_embedded_chunks(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    db, embedder = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    conn = connect(db)
    try:
        stats = vector_index.build_or_sync(conn, db)
    finally:
        conn.close()
    assert stats["available"] is True
    assert stats["synced"] >= 3
    assert vector_index.index_path_for(db).exists()


@zvec_only
def test_build_or_sync_is_incremental(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    db, _ = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    conn = connect(db)
    try:
        first = vector_index.build_or_sync(conn, db)
        second = vector_index.build_or_sync(conn, db)
    finally:
        conn.close()
    assert first["synced"] >= 3
    assert second["synced"] == 0  # nothing new to push


@zvec_only
def test_search_ids_ranks_relevant_chunk_first(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    db, embedder = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    conn = connect(db)
    try:
        vector_index.build_or_sync(conn, db)
        nvidia_doc = conn.execute("SELECT id FROM docs WHERE url = ?", ("https://ex.com/nvidia",)).fetchone()[0]
    finally:
        conn.close()
    qv = embedder.embed(["nvidia"], input_type="query")[0]
    hits = vector_index.search_ids(db, qv, top_k=5)
    assert hits is not None and hits
    # The top hit's chunk must belong to the nvidia doc.
    conn = connect(db)
    try:
        top_doc = conn.execute("SELECT doc_id FROM chunks WHERE id = ?", (hits[0][0],)).fetchone()[0]
    finally:
        conn.close()
    assert top_doc == nvidia_doc


@zvec_only
def test_search_ids_respects_scalar_filter(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    db, embedder = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    conn = connect(db)
    try:
        vector_index.build_or_sync(conn, db)
    finally:
        conn.close()
    qv = embedder.embed(["nvidia"], input_type="query")[0]
    hits = vector_index.search_ids(db, qv, top_k=10, filter_expr="source_class = 'blog'")
    assert hits is not None
    conn = connect(db)
    try:
        for cid, _ in hits:
            row = conn.execute(
                "SELECT d.source_class FROM chunks c JOIN docs d ON d.id = c.doc_id WHERE c.id = ?", (cid,)
            ).fetchone()
            assert row[0] == "blog"
    finally:
        conn.close()


@zvec_only
def test_core_search_uses_zvec_path(tmp_path, artifact_writer, words_maker, fake_embedder_factory):
    db, embedder = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    conn = connect(db)
    try:
        vector_index.build_or_sync(conn, db)
    finally:
        conn.close()
    out = core.search("nvidia chips and research", db_path=db, top_k=2, embedder=embedder)
    assert "https://ex.com/nvidia" in out


def test_core_search_falls_back_when_zvec_unavailable(
    tmp_path, artifact_writer, words_maker, fake_embedder_factory, monkeypatch
):
    """With zvec forced off, search must still return results via the numpy scan."""
    db, embedder = _seed(tmp_path, artifact_writer, words_maker, fake_embedder_factory)
    monkeypatch.setattr(vector_index, "search_ids", lambda *a, **k: None)
    out = core.search("nvidia chips and research", db_path=db, top_k=2, embedder=embedder)
    assert "https://ex.com/nvidia" in out

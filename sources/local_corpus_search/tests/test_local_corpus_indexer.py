# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scrape-artifact ingestion, chunking, and incremental indexing."""

from __future__ import annotations

import sqlite3

from local_corpus_search.indexer import chunk_text
from local_corpus_search.indexer import decode_embedding
from local_corpus_search.indexer import encode_embedding
from local_corpus_search.indexer import ingest
from local_corpus_search.indexer import rebuild


def test_chunk_text_short_text_single_chunk() -> None:
    assert chunk_text("Short paragraph.") == ["Short paragraph."]


def test_chunk_text_empty_returns_nothing() -> None:
    assert chunk_text("   ") == []


def test_chunk_text_splits_with_overlap() -> None:
    sentences = [f"Sentence number {i} talks about a research subject in moderate detail." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=1200, overlap=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1200 + 200  # chunk plus carried overlap headroom
    # Overlap: the start of each subsequent chunk repeats text from the previous one.
    for previous, current in zip(chunks, chunks[1:], strict=False):
        overlap_probe = current[:60].split(" ")[0:4]
        assert " ".join(overlap_probe) in previous


def test_chunk_text_prefers_sentence_boundaries() -> None:
    sentences = [f"Topic sentence {i} ends cleanly here." for i in range(80)]
    chunks = chunk_text(" ".join(sentences), chunk_size=600, overlap=100)
    # Every chunk should end at a sentence boundary because sentences fit within chunks.
    assert all(chunk.endswith(".") for chunk in chunks)


def test_embedding_blob_roundtrip() -> None:
    vector = [0.25, -1.5, 3.0, 0.0]
    assert decode_embedding(encode_embedding(vector)) == vector


def test_ingest_indexes_quality_docs_and_skips_bad_ones(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"

    artifact_writer(
        artifacts,
        url="https://example.com/good",
        content=words_maker(["nvidia", "research", "agents"], 200),
    )
    # Too short.
    artifact_writer(artifacts, url="https://example.com/short", content="too short to keep")
    # Metadata-only extraction status.
    artifact_writer(
        artifacts,
        url="https://example.com/meta",
        content=words_maker(["metadata"], 200),
        extraction_status="metadata_only",
    )

    stats = ingest(artifacts, db_path, embedder=None)

    assert stats["mode"] == "lexical"
    assert stats["artifacts_scanned"] == 3
    assert stats["docs_added"] == 1
    assert stats["docs_skipped_quality"] == 2
    assert stats["chunks_added"] >= 1

    conn = sqlite3.connect(db_path)
    try:
        urls = [row[0] for row in conn.execute("SELECT url FROM docs")]
        assert urls == ["https://example.com/good"]
        mode = conn.execute("SELECT value FROM meta WHERE key='index_mode'").fetchone()[0]
        assert mode == "lexical"
        embeddings = [row[0] for row in conn.execute("SELECT embedding FROM chunks")]
        assert all(blob is None for blob in embeddings)
    finally:
        conn.close()


def test_ingest_dedupes_by_content_hash_across_jobs(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    content = words_maker(["duplicate", "content"], 200)

    artifact_writer(artifacts, job_id="job-1", url="https://example.com/a", content=content)
    artifact_writer(artifacts, job_id="job-2", url="https://example.com/a", content=content)

    stats = ingest(artifacts, db_path, embedder=None)
    assert stats["docs_added"] == 1
    assert stats["docs_skipped_existing"] == 1


def test_ingest_is_incremental(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"

    artifact_writer(artifacts, url="https://example.com/one", content=words_maker(["first", "topic"], 200))
    first = ingest(artifacts, db_path, embedder=None)
    assert first["docs_added"] == 1

    artifact_writer(artifacts, url="https://example.com/two", content=words_maker(["second", "topic"], 200))
    second = ingest(artifacts, db_path, embedder=None)
    assert second["docs_added"] == 1
    assert second["docs_skipped_existing"] == 1

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 2
    finally:
        conn.close()


def test_ingest_with_embedder_stores_vectors_and_mode(
    tmp_path, artifact_writer, words_maker, fake_embedder_factory
) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(artifacts, url="https://example.com/vec", content=words_maker(["vector", "topic"], 250))

    embedder = fake_embedder_factory(["vector", "topic", "other"])
    stats = ingest(artifacts, db_path, embedder=embedder)

    assert stats["mode"] == "embedding"
    assert stats["chunks_without_embedding"] == 0
    assert embedder.calls and all(input_type == "passage" for _, input_type in embedder.calls)

    conn = sqlite3.connect(db_path)
    try:
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        assert meta["index_mode"] == "embedding"
        assert meta["embed_model"] == "fake-test-embedder"
        blobs = [row[0] for row in conn.execute("SELECT embedding FROM chunks")]
        assert blobs and all(blob is not None for blob in blobs)
        assert len(decode_embedding(blobs[0])) == 4
    finally:
        conn.close()


def test_rebuild_drops_and_reindexes(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(artifacts, url="https://example.com/r", content=words_maker(["rebuild", "topic"], 200))

    ingest(artifacts, db_path, embedder=None)
    stats = rebuild(artifacts, db_path, embedder=None)

    assert stats["docs_added"] == 1
    assert stats["docs_skipped_existing"] == 0


def test_ingest_missing_artifact_dir_is_safe(tmp_path) -> None:
    stats = ingest(tmp_path / "does-not-exist", tmp_path / "corpus.db", embedder=None)
    assert stats["docs_added"] == 0
    assert stats["artifacts_scanned"] == 0


class _FailingEmbedder:
    """Embedder that always raises, simulating an EOL/unavailable endpoint."""

    model = "dead-model"

    def embed(self, texts, *, input_type="passage"):
        from local_corpus_search.embeddings import EmbeddingError

        raise EmbeddingError("endpoint returned HTTP 410: Gone")


def test_backfill_embeddings_fills_null_vectors(tmp_path, artifact_writer, words_maker, fake_embedder_factory) -> None:
    from local_corpus_search.indexer import backfill_embeddings

    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(artifacts, url="https://example.com/a", content=words_maker(["nvidia"], 200))
    artifact_writer(
        artifacts,
        url="https://example.com/b",
        content=words_maker(["research"], 200),
    )

    # Ingest with a dead embedder: embedding mode, but all vectors NULL.
    stats = ingest(artifacts, db_path, embedder=_FailingEmbedder())
    assert stats["mode"] == "embedding"
    assert stats["chunks_without_embedding"] == stats["chunks_added"] > 0

    working = fake_embedder_factory(["nvidia", "research"])
    backfill_stats = backfill_embeddings(db_path, embedder=working, batch_size=1)

    assert backfill_stats["chunks_embedded"] == stats["chunks_added"]
    assert backfill_stats["chunks_pending_remaining"] == 0
    conn = sqlite3.connect(db_path)
    try:
        nulls = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
        model = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()[0]
    finally:
        conn.close()
    assert nulls == 0
    assert model == "fake-test-embedder"


def test_backfill_embeddings_rejects_lexical_index(
    tmp_path, artifact_writer, words_maker, fake_embedder_factory
) -> None:
    import pytest
    from local_corpus_search.indexer import backfill_embeddings

    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(artifacts, url="https://example.com/a", content=words_maker(["nvidia"], 200))
    ingest(artifacts, db_path, embedder=None)

    with pytest.raises(ValueError, match="lexical"):
        backfill_embeddings(db_path, embedder=fake_embedder_factory(["nvidia"]))


def test_backfill_embeddings_continues_when_endpoint_dies(tmp_path, artifact_writer, words_maker) -> None:
    from local_corpus_search.indexer import backfill_embeddings

    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(artifacts, url="https://example.com/a", content=words_maker(["nvidia"], 200))
    ingest(artifacts, db_path, embedder=_FailingEmbedder())

    # Endpoint is dead: every batch fails, but the run completes without raising
    # and leaves the chunks NULL for a later retry (no hard "aborted" stop).
    stats = backfill_embeddings(db_path, embedder=_FailingEmbedder())
    assert "aborted" not in stats
    assert stats["chunks_embedded"] == 0
    assert stats["chunks_failed"] > 0
    assert stats["chunks_pending_remaining"] == stats["chunks_pending_initial"]


def test_backfill_embeddings_skips_failed_batch_and_keeps_going(
    tmp_path, artifact_writer, words_maker, fake_embedder_factory
) -> None:
    """A transient failure on one batch must not block the remaining batches."""
    from local_corpus_search.embeddings import EmbeddingError
    from local_corpus_search.indexer import backfill_embeddings

    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    for i in range(4):
        artifact_writer(artifacts, url=f"https://example.com/{i}", content=words_maker([f"topic{i}"], 200))
    ingest(artifacts, db_path, embedder=_FailingEmbedder())

    base = fake_embedder_factory([f"topic{i}" for i in range(4)])

    class _FlakyOnce:
        model = base.model

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts, *, input_type="passage"):
            self.calls += 1
            if self.calls == 1:
                raise EmbeddingError("transient timeout")
            return base.embed(texts, input_type=input_type)

    flaky = _FlakyOnce()
    stats = backfill_embeddings(db_path, embedder=flaky, batch_size=1, commit_every=1)

    assert stats["chunks_failed"] >= 1
    assert stats["chunks_embedded"] >= 1
    # The one failed chunk is still pending; everything else got embedded.
    assert stats["chunks_pending_remaining"] == stats["chunks_failed"]

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end search tests: BM25 fallback, embedding mode, and friendly errors."""

from __future__ import annotations

from local_corpus_search.core import NOT_BUILT_MESSAGE
from local_corpus_search.core import _bm25_scores
from local_corpus_search.core import _cosine_scores
from local_corpus_search.core import search
from local_corpus_search.indexer import ingest


def test_search_missing_index_returns_friendly_message(tmp_path) -> None:
    result = search("anything", db_path=tmp_path / "missing.db")
    assert result == NOT_BUILT_MESSAGE
    assert "build_corpus_index" in result


def test_search_empty_index_returns_friendly_message(tmp_path) -> None:
    ingest(tmp_path / "no-artifacts", tmp_path / "corpus.db", embedder=None)
    result = search("anything", db_path=tmp_path / "corpus.db")
    assert result == NOT_BUILT_MESSAGE


def test_bm25_search_end_to_end(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(
        artifacts,
        url="https://example.com/quantum",
        title="Quantum Computing Advances",
        content=words_maker(["quantum", "computing", "qubits"], 220),
        source_class="academic",
    )
    artifact_writer(
        artifacts,
        url="https://example.com/cooking",
        title="Pasta Recipes",
        content=words_maker(["pasta", "tomato", "basil"], 220),
    )
    ingest(artifacts, db_path, embedder=None)

    result = search("quantum computing qubits", db_path=db_path, top_k=3)

    assert "<document idx=\"0\">" in result
    assert "<url>https://example.com/quantum</url>" in result
    assert "<title>Quantum Computing Advances</title>" in result
    assert "<source_class>academic</source_class>" in result
    assert "<cached_corpus>true</cached_corpus>" in result
    # The irrelevant pasta doc must not outrank the quantum doc.
    assert result.index("quantum") < len(result)
    assert "<url>https://example.com/cooking</url>" not in result.split("---")[0]


def test_bm25_no_match_returns_no_results_message(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(artifacts, content=words_maker(["solar", "panels"], 200))
    ingest(artifacts, db_path, embedder=None)

    result = search("zzzunmatchable tokenxyz", db_path=db_path)
    assert "No local corpus documents matched" in result


def test_embedding_search_with_fake_embedder(tmp_path, artifact_writer, words_maker, fake_embedder_factory) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(
        artifacts,
        url="https://example.com/fusion",
        title="Fusion Energy Milestones",
        content=words_maker(["fusion", "plasma", "tokamak"], 250),
    )
    artifact_writer(
        artifacts,
        url="https://example.com/banking",
        title="Banking Regulation",
        content=words_maker(["banking", "regulation", "capital"], 250),
    )

    embedder = fake_embedder_factory(["fusion", "plasma", "tokamak", "banking", "regulation", "capital"])
    ingest(artifacts, db_path, embedder=embedder)

    result = search("fusion plasma tokamak progress", db_path=db_path, embedder=embedder)

    first_block = result.split("---")[0]
    assert "<url>https://example.com/fusion</url>" in first_block
    assert "<cached_corpus>true</cached_corpus>" in result
    # Query was embedded with input_type="query".
    assert any(input_type == "query" for _, input_type in embedder.calls)


def test_embedding_index_without_key_falls_back_to_bm25(
    tmp_path, artifact_writer, words_maker, fake_embedder_factory, caplog
) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(
        artifacts,
        url="https://example.com/climate",
        title="Climate Modelling",
        content=words_maker(["climate", "modelling", "emissions"], 250),
    )
    embedder = fake_embedder_factory(["climate", "modelling", "emissions"])
    ingest(artifacts, db_path, embedder=embedder)

    # No embedder injected and no NVIDIA_API_KEY in env -> BM25 over the same chunks.
    with caplog.at_level("WARNING"):
        result = search("climate modelling emissions", db_path=db_path)

    assert "<url>https://example.com/climate</url>" in result
    assert "<cached_corpus>true</cached_corpus>" in result
    assert any("falling back to BM25" in record.message for record in caplog.records)


def test_search_never_raises_on_corrupt_db(tmp_path) -> None:
    bad_db = tmp_path / "corrupt.db"
    bad_db.write_bytes(b"this is not a sqlite database at all")
    result = search("anything", db_path=bad_db)
    assert isinstance(result, str)
    assert "<document" not in result


def test_snippet_is_trimmed(tmp_path, artifact_writer, words_maker) -> None:
    artifacts = tmp_path / "artifacts"
    db_path = tmp_path / "corpus.db"
    artifact_writer(
        artifacts,
        url="https://example.com/long",
        content=words_maker(["lengthy", "report", "details"], 1200),
    )
    ingest(artifacts, db_path, embedder=None)

    result = search("lengthy report details", db_path=db_path, max_snippet_chars=500)
    content = result.split("<content>")[1].split("</content>")[0]
    assert len(content) <= 600


def test_cosine_scores_pure_python_matches_numpy_path() -> None:
    query = [1.0, 0.0, 2.0]
    vectors = [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [2.0, 0.0, 4.0]]
    scores = _cosine_scores(query, vectors)
    assert scores[0] > 0.99
    assert scores[2] > 0.99
    assert scores[1] < 0.01


def test_bm25_scores_rank_relevant_chunk_higher() -> None:
    chunks = [
        "quantum computing with qubits and entanglement",
        "cooking pasta with tomato sauce and basil",
    ]
    scores = _bm25_scores("quantum qubits", chunks)
    assert scores[0] > scores[1]

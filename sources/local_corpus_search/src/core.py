# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Search the local scrape-artifact corpus index.

Results are formatted as the same XML ``<document>`` blocks the web search
tool emits so agents can consume cached corpus hits identically, with an extra
``<cached_corpus>true</cached_corpus>`` marker.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from . import vector_index
from .embeddings import EmbeddingError
from .embeddings import get_default_embedder
from .indexer import corpus_db_path
from .indexer import decode_embedding

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 6
MAX_SNIPPET_CHARS = 1500
_CANDIDATE_MULTIPLIER = 5
_MAX_CHUNKS_PER_DOC = 2

# Marker object: "resolve the default embedder from the environment".
_AUTO = object()

NOT_BUILT_MESSAGE = (
    "Local corpus index not built yet - run scripts/build_corpus_index.py to index "
    "previously scraped pages, then retry. Falling back to web search is appropriate."
)

_STOPWORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]+")


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _cosine_scores(query_vec: list[float], chunk_vectors: list[list[float]]) -> list[float]:
    """Cosine similarity of the query against each chunk vector.

    Uses numpy when importable, otherwise a pure-python fallback.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        matrix = np.asarray(chunk_vectors, dtype=np.float32)
        query = np.asarray(query_vec, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        norms[norms == 0] = 1.0
        return (matrix @ query / norms).tolist()

    query_norm = math.sqrt(sum(value * value for value in query_vec)) or 1.0
    scores: list[float] = []
    for vector in chunk_vectors:
        dot = sum(q * v for q, v in zip(query_vec, vector, strict=False))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        scores.append(dot / (norm * query_norm))
    return scores


def _bm25_scores(query: str, chunk_texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Pure-python BM25 scores for the query against each chunk text."""
    query_terms = Counter(tokenize(query))
    if not query_terms or not chunk_texts:
        return [0.0] * len(chunk_texts)

    token_lists = [tokenize(text) for text in chunk_texts]
    doc_count = len(token_lists)
    avgdl = (sum(len(tokens) for tokens in token_lists) / doc_count) or 1.0

    df: Counter[str] = Counter()
    for tokens in token_lists:
        for token in set(tokens):
            df[token] += 1
    idf = {
        token: math.log((doc_count - count + 0.5) / (count + 0.5) + 1.0)
        for token, count in df.items()
    }

    scores: list[float] = []
    for tokens in token_lists:
        counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term, weight in query_terms.items():
            tf = counts.get(term, 0)
            if tf:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
                score += idf.get(term, 0.0) * (numerator / denominator) * weight
        scores.append(score)
    return scores


def _trim_snippet(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(" ", 0, max_chars - 3)
    if cutoff < max_chars // 2:
        cutoff = max_chars - 3
    return text[:cutoff].rstrip() + "..."


def format_results(results: list[dict[str, Any]], *, max_snippet_chars: int = MAX_SNIPPET_CHARS) -> str:
    if not results:
        return "No local corpus documents matched the query. Use web search to research this topic."
    documents: list[str] = []
    for idx, result in enumerate(results):
        snippet = _trim_snippet(str(result.get("snippet") or ""), max_snippet_chars)
        documents.append(
            f'<document idx="{idx}">\n'
            f"<title>{escape(str(result.get('title') or result.get('url') or ''))}</title>\n"
            f"<url>{escape(str(result.get('url') or ''))}</url>\n"
            f"<source_class>{escape(str(result.get('source_class') or ''))}</source_class>\n"
            f"<published_at>{escape(str(result.get('published_at') or ''))}</published_at>\n"
            f"<content>{escape(snippet)}</content>\n"
            f"<cached_corpus>true</cached_corpus>\n"
            f"</document>"
        )
    return "\n\n---\n\n".join(documents)


def _fetch_scored_chunks(
    conn: sqlite3.Connection, hits: list[tuple[int, float]]
) -> tuple[list[dict[str, Any]], list[float]]:
    """Fetch chunk rows for zvec hit ids, preserving the hit (score) ordering."""
    if not hits:
        return [], []
    id_order = [cid for cid, _ in hits]
    score_by_id = {cid: score for cid, score in hits}
    placeholders = ",".join("?" * len(id_order))
    rows = conn.execute(
        "SELECT c.id, c.doc_id, c.idx, c.text, c.embedding, "
        "d.url, d.title, d.source_class, d.published_at "
        "FROM chunks c JOIN docs d ON d.id = c.doc_id "
        f"WHERE c.id IN ({placeholders})",
        id_order,
    ).fetchall()
    by_id = {row[0]: row for row in rows}
    chunks: list[dict[str, Any]] = []
    scores: list[float] = []
    for cid in id_order:
        row = by_id.get(cid)
        if row is None:
            continue
        chunks.append(
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "idx": row[2],
                "text": row[3],
                "embedding": row[4],
                "url": row[5],
                "title": row[6],
                "source_class": row[7],
                "published_at": row[8],
            }
        )
        scores.append(score_by_id[cid])
    return chunks, scores


def _load_chunks(conn: sqlite3.Connection, *, with_embeddings: bool) -> list[dict[str, Any]]:
    query = (
        "SELECT chunks.id, chunks.doc_id, chunks.idx, chunks.text, chunks.embedding, "
        "docs.url, docs.title, docs.source_class, docs.published_at "
        "FROM chunks JOIN docs ON docs.id = chunks.doc_id"
    )
    if with_embeddings:
        query += " WHERE chunks.embedding IS NOT NULL"
    rows = conn.execute(query).fetchall()
    return [
        {
            "chunk_id": row[0],
            "doc_id": row[1],
            "idx": row[2],
            "text": row[3],
            "embedding": row[4],
            "url": row[5],
            "title": row[6],
            "source_class": row[7],
            "published_at": row[8],
        }
        for row in rows
    ]


def _group_by_doc(
    chunks: list[dict[str, Any]],
    scores: list[float],
    *,
    top_k: int,
    max_snippet_chars: int,
) -> list[dict[str, Any]]:
    """Group the best-scoring chunks by document and build snippets."""
    candidates = sorted(zip(scores, chunks, strict=True), key=lambda item: item[0], reverse=True)
    candidates = [(score, chunk) for score, chunk in candidates if score > 0]
    candidates = candidates[: max(top_k * _CANDIDATE_MULTIPLIER, top_k)]

    by_doc: dict[int, dict[str, Any]] = {}
    for score, chunk in candidates:
        doc_id = chunk["doc_id"]
        entry = by_doc.setdefault(
            doc_id,
            {
                "url": chunk["url"],
                "title": chunk["title"],
                "source_class": chunk["source_class"],
                "published_at": chunk["published_at"],
                "best_score": score,
                "chunks": [],
            },
        )
        if len(entry["chunks"]) < _MAX_CHUNKS_PER_DOC:
            entry["chunks"].append(chunk)

    results: list[dict[str, Any]] = []
    for entry in sorted(by_doc.values(), key=lambda item: item["best_score"], reverse=True)[:top_k]:
        ordered = sorted(entry["chunks"], key=lambda chunk: chunk["idx"])
        snippet = " ... ".join(chunk["text"] for chunk in ordered)
        results.append(
            {
                "url": entry["url"],
                "title": entry["title"],
                "source_class": entry["source_class"],
                "published_at": entry["published_at"],
                "score": round(entry["best_score"], 4),
                "snippet": snippet[: max_snippet_chars + 200],
            }
        )
    return results


def search(
    query: str,
    *,
    db_path: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_snippet_chars: int = MAX_SNIPPET_CHARS,
    embedder: Any = _AUTO,
) -> str:
    """Search the local corpus and return web-search-style XML document blocks.

    Never raises: missing/empty indexes and runtime failures return friendly
    message strings instead.
    """
    try:
        return _search_impl(
            query,
            db_path=db_path,
            top_k=top_k,
            max_snippet_chars=max_snippet_chars,
            embedder=embedder,
        )
    except Exception as exc:  # noqa: BLE001 - tool surface must not raise
        logger.exception("Local corpus search failed")
        return f"Local corpus search failed ({exc}). Use web search for this topic instead."


def _search_impl(
    query: str,
    *,
    db_path: str | Path | None,
    top_k: int,
    max_snippet_chars: int,
    embedder: Any,
) -> str:
    resolved_db = Path(db_path) if db_path is not None else Path(corpus_db_path())
    if not resolved_db.exists():
        return NOT_BUILT_MESSAGE

    conn = sqlite3.connect(str(resolved_db), timeout=30.0)
    try:
        try:
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            mode_row = conn.execute("SELECT value FROM meta WHERE key = 'index_mode'").fetchone()
        except sqlite3.OperationalError:
            return NOT_BUILT_MESSAGE
        if not chunk_count:
            return NOT_BUILT_MESSAGE
        mode = mode_row[0] if mode_row else "lexical"

        if embedder is _AUTO:
            embedder = get_default_embedder() if mode == "embedding" else None

        if mode == "embedding" and embedder is not None:
            try:
                query_vec: list[float] | None = embedder.embed([query], input_type="query")[0]
            except EmbeddingError as exc:
                logger.warning("Query embedding failed (%s); falling back to BM25", exc)
                query_vec = None

            if query_vec is not None:
                # Fast path: zvec ANN index (sub-linear, no full matrix load).
                hits = vector_index.search_ids(
                    resolved_db, query_vec, top_k=top_k * _CANDIDATE_MULTIPLIER
                )
                if hits is not None:
                    chunks, scores = _fetch_scored_chunks(conn, hits)
                    if chunks:
                        results = _group_by_doc(
                            chunks, scores, top_k=top_k, max_snippet_chars=max_snippet_chars
                        )
                        return format_results(results, max_snippet_chars=max_snippet_chars)
                # Fallback: in-memory full scan over SQLite embeddings.
                chunks = _load_chunks(conn, with_embeddings=True)
                if chunks:
                    vectors = [decode_embedding(chunk["embedding"]) for chunk in chunks]
                    scores = _cosine_scores(query_vec, vectors)
                    results = _group_by_doc(
                        chunks, scores, top_k=top_k, max_snippet_chars=max_snippet_chars
                    )
                    return format_results(results, max_snippet_chars=max_snippet_chars)
        elif mode == "embedding":
            logger.warning(
                "Corpus index is embedding-mode but NVIDIA_API_KEY is unavailable; "
                "falling back to BM25 over the same chunks."
            )

        chunks = _load_chunks(conn, with_embeddings=False)
        scores = _bm25_scores(query, [chunk["text"] for chunk in chunks])
        results = _group_by_doc(chunks, scores, top_k=top_k, max_snippet_chars=max_snippet_chars)
        return format_results(results, max_snippet_chars=max_snippet_chars)
    finally:
        conn.close()

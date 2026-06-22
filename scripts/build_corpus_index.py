#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build or update the local corpus index over persisted scrape artifacts.

Scans gzip JSON scrape artifacts (written by the research jobs under
AIQ_SCRAPE_ARTIFACT_DIR) and indexes them into the SQLite corpus DB consumed
by the local_corpus_search tool. Embeddings use NVIDIA NIM when
NVIDIA_API_KEY is set; otherwise the index is built in lexical (BM25) mode.

The DB uses WAL mode, so this is safe to run while the app is serving.

Usage:
    uv run python scripts/build_corpus_index.py
    uv run python scripts/build_corpus_index.py --artifact-dir ./data/scrape_artifacts --db ./data/corpus_index.db
    uv run python scripts/build_corpus_index.py --rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _import_indexer():
    """Import the indexer, falling back to a path-based load when the package is not installed."""
    try:
        from local_corpus_search import indexer
    except ImportError:
        import importlib.util

        package_root = Path(__file__).resolve().parent.parent / "sources" / "local_corpus_search" / "src"
        spec = importlib.util.spec_from_file_location(
            "local_corpus_search",
            package_root / "__init__.py",
            submodule_search_locations=[str(package_root)],
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["local_corpus_search"] = module
        spec.loader.exec_module(module)
        from local_corpus_search import indexer
    return indexer


def _sync_vector_index(indexer, db: str | None, *, rebuild: bool) -> None:
    """Build/sync the zvec ANN index from the SQLite corpus (no-op if zvec absent)."""
    from local_corpus_search import vector_index

    if not vector_index.zvec_available():
        print("  vector index:            skipped (zvec not installed; SQLite path active)")
        return
    resolved_db = db or indexer.corpus_db_path()
    conn = indexer.connect(resolved_db)
    try:
        stats = vector_index.build_or_sync(conn, resolved_db, rebuild=rebuild)
    finally:
        conn.close()
    print(
        f"  vector index:            synced {stats.get('synced', 0)} chunks "
        f"(dim={stats.get('dimension', '?')}) -> {vector_index.index_path_for(resolved_db)}"
    )


def main() -> int:
    indexer = _import_indexer()

    parser = argparse.ArgumentParser(description="Build the local corpus index over scrape artifacts.")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=f"Scrape artifact root (default: $AIQ_SCRAPE_ARTIFACT_DIR or {indexer.DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite corpus index path (default: $AIQ_CORPUS_DB or {indexer.DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing index and re-ingest every artifact from scratch.",
    )
    parser.add_argument(
        "--backfill-embeddings",
        action="store_true",
        help="Embed chunks stored without vectors (e.g. after an embedding endpoint outage).",
    )
    parser.add_argument(
        "--no-vector-index",
        action="store_true",
        help="Skip building the zvec ANN index (SQLite numpy/BM25 search still works).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.backfill_embeddings:
        stats = indexer.backfill_embeddings(args.db)
        print("Embedding backfill complete")
        print(f"  chunks embedded:         {stats['chunks_embedded']}")
        print(f"  chunks failed (batch):   {stats['chunks_failed']}")
        print(f"  chunks still pending:    {stats['chunks_pending_remaining']}")
        print(f"  duration:                {stats['duration_seconds']}s")
        if not args.no_vector_index:
            _sync_vector_index(indexer, args.db, rebuild=False)
        # Non-zero exit when work remains so a cron/wrapper can re-invoke to retry.
        return 0 if stats["chunks_pending_remaining"] == 0 else 1

    if args.rebuild:
        stats = indexer.rebuild(args.artifact_dir, args.db)
    else:
        stats = indexer.ingest(args.artifact_dir, args.db)

    print("Local corpus index build complete")
    print(f"  mode:                    {stats['mode']}")
    if stats.get("embed_model"):
        print(f"  embed model:             {stats['embed_model']}")
    print(f"  artifacts scanned:       {stats['artifacts_scanned']}")
    print(f"  docs added:              {stats['docs_added']}")
    print(f"  docs skipped (existing): {stats['docs_skipped_existing']}")
    print(f"  docs skipped (quality):  {stats['docs_skipped_quality']}")
    print(f"  chunks added:            {stats['chunks_added']}")
    if stats.get("chunks_without_embedding"):
        print(f"  chunks w/o embedding:    {stats['chunks_without_embedding']}")
    print(f"  duration:                {stats['duration_seconds']}s")
    if not args.no_vector_index:
        _sync_vector_index(indexer, args.db, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

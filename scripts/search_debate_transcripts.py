#!/usr/bin/env python3
"""Search the prepared local debate transcript corpus from the terminal."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "sources" / "debate_transcript_search" / "src" / "core.py"

spec = importlib.util.spec_from_file_location("debate_transcript_search_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load debate transcript search module: {CORE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

DEFAULT_CORPUS_JSONL = module.DEFAULT_CORPUS_JSONL
format_results = module.format_results
search_rows = module.search_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument("--corpus-jsonl", default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-content-length", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = search_rows(
        args.query,
        corpus_jsonl=args.corpus_jsonl,
        max_results=args.top_k,
        max_content_length=args.max_content_length,
    )
    print(format_results(args.query, rows))


if __name__ == "__main__":
    main()

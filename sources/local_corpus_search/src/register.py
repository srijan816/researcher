# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NAT registration for the local scrape-artifact corpus search tool."""

from __future__ import annotations

import logging
import os

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .core import DEFAULT_TOP_K
from .core import MAX_SNIPPET_CHARS
from .core import search
from .indexer import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


class LocalCorpusSearchToolConfig(FunctionBaseConfig, name="local_corpus_search"):
    """Search the local embedding-indexed corpus of previously scraped pages."""

    db_path: str = Field(
        default_factory=lambda: os.getenv("AIQ_CORPUS_DB", DEFAULT_DB_PATH),
        description="Path to the SQLite corpus index built by scripts/build_corpus_index.py.",
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="Maximum documents to return.")
    max_snippet_chars: int = Field(
        default=MAX_SNIPPET_CHARS, ge=300, le=6000, description="Maximum characters per document snippet."
    )


@register_function(config_type=LocalCorpusSearchToolConfig)
async def local_corpus_search(tool_config: LocalCorpusSearchToolConfig, builder: Builder):
    async def _search(query: str) -> str:
        """Search the local corpus of previously scraped web pages before spending web-search budget.

        Use this tool FIRST for topics likely covered by prior research runs (recurring companies,
        technologies, people, or themes). Results are real web documents persisted from earlier
        jobs and include their original source URLs, so cite the returned URLs exactly as you would
        cite web search results. If the local corpus has no relevant documents, fall back to web
        search. This tool does not access the live web.
        """
        try:
            return search(
                query,
                db_path=tool_config.db_path,
                top_k=tool_config.top_k,
                max_snippet_chars=tool_config.max_snippet_chars,
            )
        except Exception as exc:  # noqa: BLE001 - defensive: core.search should not raise
            logger.exception("Local corpus search tool failed")
            return f"Error: local corpus search failed - {exc}"

    yield FunctionInfo.from_fn(_search, description=_search.__doc__)

"""Local debate transcript search tool for AI-Q.

The tool reads the prepared corpus JSONL produced by
scripts/prepare_debate_transcript_corpus.py and returns citation-ready chunks.
It is intentionally lexical and dependency-light so it can run locally, in
Docker, or on the ARM VPS without a separate vector database.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_JSONL = (
    "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/"
    "prepared_debate_corpus/chunks/search_chunks.jsonl"
)

STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "are",
    "around",
    "because",
    "been",
    "being",
    "between",
    "debate",
    "debates",
    "does",
    "extract",
    "find",
    "for",
    "from",
    "have",
    "how",
    "into",
    "local",
    "motion",
    "not",
    "only",
    "research",
    "search",
    "should",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "transcript",
    "transcripts",
    "using",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

SYNONYMS = {
    "ai": ("artificial", "intelligence"),
    "free": ("freedom",),
    "freedom": ("free",),
    "universities": ("university", "campus"),
    "university": ("universities", "campus"),
}


class DebateTranscriptSearchToolConfig(FunctionBaseConfig, name="debate_transcript_search"):
    """Search a prepared local debate transcript corpus."""

    corpus_jsonl: str = Field(
        default_factory=lambda: os.getenv("DEBATE_TRANSCRIPT_CORPUS_JSONL", DEFAULT_CORPUS_JSONL),
        description="Path to prepared_debate_corpus/chunks/search_chunks.jsonl.",
    )
    max_results: int = Field(default=8, ge=1, le=30, description="Maximum chunks to return.")
    max_content_length: int = Field(default=2500, ge=500, le=12000, description="Maximum characters per chunk.")
    max_results_per_transcript: int = Field(default=2, ge=1, le=10, description="Diversity cap per transcript.")
    candidate_pool: int = Field(default=80, ge=10, le=500, description="Candidate chunks retained before diversity.")


def _tokenize(text: str) -> list[str]:
    return [
        token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'\-]{2,}", text) if token.lower() not in STOPWORDS
    ]


def _query_phrases(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    phrases = re.findall(r'"([^"]{4,120})"', normalized)
    words = _tokenize(normalized)
    phrases.extend(" ".join(words[i : i + 2]) for i in range(len(words) - 1))
    phrases.extend(" ".join(words[i : i + 3]) for i in range(len(words) - 2))
    return [phrase for phrase in phrases if len(phrase) >= 5]


def _expand_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        if term.endswith("ies") and len(term) > 4:
            expanded.append(term[:-3] + "y")
        elif term.endswith("s") and len(term) > 4:
            expanded.append(term[:-1])
        expanded.extend(SYNONYMS.get(term, ()))
    return expanded


def _truncate(text: str, max_length: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_length:
        return text
    cutoff = text.rfind(" ", 0, max_length - 3)
    if cutoff < max_length // 2:
        cutoff = max_length - 3
    return text[:cutoff].rstrip() + "..."


@lru_cache(maxsize=4)
def _load_rows(corpus_jsonl: str) -> tuple[dict[str, Any], ...]:
    path = Path(corpus_jsonl).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Debate transcript corpus not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_search_text"] = " ".join(
                [
                    str(row.get("motion") or ""),
                    str(row.get("source_title") or ""),
                    " ".join(row.get("themes") or []),
                    str(row.get("competition") or ""),
                    str(row.get("round") or ""),
                    str(row.get("year") or ""),
                    str(row.get("text") or ""),
                ]
            ).lower()
            row["_motion_text"] = str(row.get("motion") or "").lower()
            row["_theme_text"] = " ".join(row.get("themes") or []).lower()
            rows.append(row)

    logger.info("Loaded %d debate transcript chunks from %s", len(rows), path)
    return tuple(rows)


def search_rows(
    query: str,
    *,
    corpus_jsonl: str = DEFAULT_CORPUS_JSONL,
    max_results: int = 8,
    max_content_length: int = 2500,
    max_results_per_transcript: int = 2,
    candidate_pool: int = 80,
) -> list[dict[str, Any]]:
    rows = _load_rows(corpus_jsonl)
    query_terms = _expand_terms(_tokenize(query))
    phrases = _query_phrases(query)
    if not query_terms and not phrases:
        return []

    term_counts = Counter(query_terms)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        haystack = row["_search_text"]
        motion_text = row["_motion_text"]
        theme_text = row["_theme_text"]
        score = 0.0
        for term, weight in term_counts.items():
            count = haystack.count(term)
            if count:
                score += (1.0 + math.log1p(count)) * weight
                if term in motion_text:
                    score += 2.5
                if term in theme_text:
                    score += 1.5
        for phrase in phrases:
            if phrase in haystack:
                score += 4.0
                if phrase in motion_text:
                    score += 5.0
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    transcript_counts: Counter[str] = Counter()
    for score, row in scored[:candidate_pool]:
        transcript_id = str(row.get("transcript_id") or "")
        if transcript_counts[transcript_id] >= max_results_per_transcript:
            continue
        result = dict(row)
        result["score"] = round(score, 3)
        result["text"] = _truncate(str(row.get("text") or ""), max_content_length)
        selected.append(result)
        transcript_counts[transcript_id] += 1
        if len(selected) >= max_results:
            break
    return selected


def format_results(query: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"No local debate transcript chunks found for query: {query!r}"

    formatted: list[str] = []
    for index, row in enumerate(rows, 1):
        themes = ", ".join(row.get("themes") or [])
        word_range = f"{row.get('word_start', '')}-{row.get('word_end', '')}"
        source_ref = str(row.get("youtube_url") or row.get("canonical_path") or "")
        citation = f"{row.get('transcript_id')} {row.get('chunk_id')} {source_ref}".strip()
        attrs = {
            "rank": str(index),
            "score": str(row.get("score", "")),
            "transcript_id": str(row.get("transcript_id") or ""),
            "chunk_id": str(row.get("chunk_id") or ""),
            "corpus_source": str(row.get("corpus_source") or ""),
        }
        attr_text = " ".join(f'{key}="{escape(value)}"' for key, value in attrs.items())
        formatted.append(
            f"<local_debate_transcript {attr_text}>\n"
            f"<motion>{escape(str(row.get('motion') or ''))}</motion>\n"
            f"<themes>{escape(themes)}</themes>\n"
            f"<competition>{escape(str(row.get('competition') or ''))}</competition>\n"
            f"<year>{escape(str(row.get('year') or ''))}</year>\n"
            f"<round>{escape(str(row.get('round') or ''))}</round>\n"
            f"<word_range>{escape(word_range)}</word_range>\n"
            f"<canonical_path>{escape(str(row.get('canonical_path') or ''))}</canonical_path>\n"
            f"<source_raw_text_path>{escape(str(row.get('source_raw_text_path') or ''))}</source_raw_text_path>\n"
            f"<youtube_url>{escape(str(row.get('youtube_url') or ''))}</youtube_url>\n"
            f"<citation>{escape(citation)}</citation>\n"
            f"<excerpt>{escape(str(row.get('text') or ''))}</excerpt>\n"
            "</local_debate_transcript>"
        )
    return "\n\n".join(formatted)


@register_function(config_type=DebateTranscriptSearchToolConfig)
async def debate_transcript_search(tool_config: DebateTranscriptSearchToolConfig, builder: Builder):
    async def _search(query: str) -> str:
        """Search local debate transcripts for motions, framing, clash, rebuttal, and examples.

        Use this tool when the user asks for debate technique, argument interactions, motion
        analogies, government/opposition framing, or examples from the local transcript corpus.
        Cite transcript_id, chunk_id, and the returned citation URL/path. This tool does not search the web.
        """
        try:
            rows = search_rows(
                query,
                corpus_jsonl=tool_config.corpus_jsonl,
                max_results=tool_config.max_results,
                max_content_length=tool_config.max_content_length,
                max_results_per_transcript=tool_config.max_results_per_transcript,
                candidate_pool=tool_config.candidate_pool,
            )
        except Exception as exc:
            logger.exception("Local debate transcript search failed")
            return f"Error: local debate transcript search failed - {exc}"
        return format_results(query, rows)

    yield FunctionInfo.from_fn(_search, description=_search.__doc__)

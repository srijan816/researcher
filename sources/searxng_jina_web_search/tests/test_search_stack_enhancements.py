# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reranking, recency, extraction quality, and the scrape cache read path."""

from __future__ import annotations

import gzip
import json
from datetime import UTC
from datetime import datetime
from datetime import timedelta

import pytest
from searxng_jina_web_search import enrichment
from searxng_jina_web_search import register
from searxng_jina_web_search import reranker

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def test_resolve_backend_defaults_to_heuristic_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIQ_RERANK_BACKEND", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert reranker.resolve_rerank_backend() == "heuristic"


def test_resolve_backend_defaults_to_nvidia_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIQ_RERANK_BACKEND", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    assert reranker.resolve_rerank_backend() == "nvidia"


def test_resolve_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_RERANK_BACKEND", "local")
    assert reranker.resolve_rerank_backend() == "local"


def test_build_passage_truncates_content() -> None:
    passage = reranker.build_passage({"title": "Title", "content": "x" * 5000})
    assert passage.startswith("Title\n")
    assert len(passage) <= len("Title\n") + reranker.PASSAGE_MAX_CONTENT_CHARS


async def test_rerank_results_nvidia_uses_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_RERANK_BACKEND", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    captured: dict = {}

    def fake_post_json(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["auth"] = headers.get("Authorization")
        return {"rankings": [{"index": 1, "logit": 4.0}, {"index": 0, "logit": -1.5}]}

    monkeypatch.setattr(reranker, "_post_json", fake_post_json)
    results = [
        {"title": "weak", "content": "barely related"},
        {"title": "strong", "content": "highly relevant passage"},
    ]

    scores = await reranker.rerank_results("test query", results)

    assert scores == [-1.5, 4.0]
    assert captured["payload"]["query"] == {"text": "test query"}
    assert len(captured["payload"]["passages"]) == 2
    assert captured["auth"] == "Bearer test-key"


async def test_rerank_results_fails_open_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_RERANK_BACKEND", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    def broken_post_json(url, payload, headers, timeout):
        raise RuntimeError("HTTP 503 from NIM")

    monkeypatch.setattr(reranker, "_post_json", broken_post_json)

    scores = await reranker.rerank_results("q", [{"title": "a"}, {"title": "b"}])

    assert scores is None


async def test_rerank_results_heuristic_backend_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_RERANK_BACKEND", "heuristic")
    assert await reranker.rerank_results("q", [{"title": "a"}]) is None


def test_normalize_scores_handles_degenerate_spread() -> None:
    assert reranker.normalize_scores([2.0, 2.0]) == [0.5, 0.5]
    assert reranker.normalize_scores([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


async def test_rerank_merged_results_blends_and_reorders(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        {"title": "heuristic-first", "url": "https://example.com/a"},
        {"title": "rerank-first", "url": "https://example.com/b"},
        {"title": "third", "url": "https://example.com/c"},
    ]

    async def fake_rerank(query, candidates, timeout=None):
        return [0.0, 10.0, 5.0]

    monkeypatch.setattr(register, "rerank_results", fake_rerank)
    reordered = await register._rerank_merged_results("test query", list(results))

    assert [r["title"] for r in reordered] == ["rerank-first", "third", "heuristic-first"]


async def test_rerank_merged_results_keeps_order_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [{"title": "a", "url": "https://example.com/a"}, {"title": "b", "url": "https://example.com/b"}]

    async def fake_rerank(query, candidates, timeout=None):
        return None

    monkeypatch.setattr(register, "rerank_results", fake_rerank)
    assert await register._rerank_merged_results("q", list(results)) == results


def test_blend_prefers_authority_and_freshness_on_ties() -> None:
    today = datetime.now(UTC).date().isoformat()
    fresh_authoritative = {"url": "https://nature.com/article", "published_at": today}
    stale_weak = {"url": "https://medium.com/post", "published_at": "2015-01-01"}

    blended = register._blend_rerank_scores([fresh_authoritative, stale_weak], [1.0, 1.0])

    assert blended[0] > blended[1]


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


def test_extract_published_date_from_url_patterns() -> None:
    assert enrichment.extract_published_date_from_url("https://example.com/2026/05/article") == "2026-05-01"
    assert enrichment.extract_published_date_from_url("https://example.com/2026/05/12/post") == "2026-05-12"
    assert enrichment.extract_published_date_from_url("https://example.com/news-2026-05-12-update") == "2026-05-12"
    assert enrichment.extract_published_date_from_url("https://example.com/article/12345") is None


def test_extract_published_date_from_html_meta_tags() -> None:
    html_doc = '<html><head><meta property="article:published_time" content="2026-05-12T08:30:00Z"></head></html>'
    assert enrichment.extract_published_date_from_html(html_doc) == "2026-05-12"

    og_doc = '<meta content="2025-11-03" property="og:published_time">'
    assert enrichment.extract_published_date_from_html(og_doc) == "2025-11-03"

    json_ld = '<script type="application/ld+json">{"datePublished": "2026-01-20T00:00:00+00:00"}</script>'
    assert enrichment.extract_published_date_from_html(json_ld) == "2026-01-20"

    jina_markdown = "Title: Example\nPublished Time: 2026-03-04T10:00:00Z\nMarkdown body"
    assert enrichment.extract_published_date_from_html(jina_markdown) == "2026-03-04"

    assert enrichment.extract_published_date_from_html("<html><body>no dates</body></html>") is None


def test_extract_published_date_from_searxng_field() -> None:
    result = {"url": "https://example.com/article", "publishedDate": "2026-04-01T12:00:00"}
    assert enrichment.extract_published_date_from_result(result) == "2026-04-01"


def test_extract_published_date_from_result_falls_back_to_url() -> None:
    result = {"url": "https://example.com/2026/02/14/story"}
    assert enrichment.extract_published_date_from_result(result) == "2026-02-14"
    assert enrichment.extract_published_date_from_result({"url": "https://example.com/story"}) is None


# ---------------------------------------------------------------------------
# Recency intent
# ---------------------------------------------------------------------------


def test_detect_recency_intent_month_terms() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    assert enrichment.detect_recency_intent("latest MiniMax model release", now=now) == "month"
    assert enrichment.detect_recency_intent("AI chip export news", now=now) == "month"
    assert enrichment.detect_recency_intent("what happened today in markets", now=now) == "month"
    assert enrichment.detect_recency_intent("breaking developments this week", now=now) == "month"


def test_detect_recency_intent_year_mentions() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    assert enrichment.detect_recency_intent("MiniMax M3 benchmarks 2026", now=now) == "year"
    assert enrichment.detect_recency_intent("GPU market share 2025", now=now) == "year"
    assert enrichment.detect_recency_intent("dot com bubble 2000 crash analysis", now=now) is None


def test_detect_recency_intent_neutral_queries() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    assert enrichment.detect_recency_intent("photosynthesis mechanism chlorophyll", now=now) is None
    assert enrichment.detect_recency_intent("", now=now) is None


def test_freshness_score_decay_and_neutral() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    fresh = enrichment.freshness_score(now.date().isoformat(), now=now)
    month_old = enrichment.freshness_score((now - timedelta(days=45)).date().isoformat(), now=now)
    ancient = enrichment.freshness_score("2010-01-01", now=now)

    assert fresh == 1.0
    assert month_old == pytest.approx(0.5, abs=0.02)
    assert ancient < 0.01
    assert enrichment.freshness_score(None) == 0.5
    assert enrichment.freshness_score("not-a-date") == 0.5


def test_result_rank_prefers_fresh_results() -> None:
    query = "ai funding"
    base = {"title": "ai funding report", "url": "https://example.com/report", "content": "ai funding analysis"}
    fresh = {**base, "published_at": datetime.now(UTC).date().isoformat()}
    stale = {**base, "published_at": "2018-03-01"}

    assert register._result_rank(fresh, query) > register._result_rank(stale, query)


# ---------------------------------------------------------------------------
# Extraction quality
# ---------------------------------------------------------------------------


def test_classify_extraction_full() -> None:
    text = " ".join(f"word{i}" for i in range(450))
    content, quality = enrichment.classify_extraction(text, "snippet")
    assert quality == "full"
    assert content == text


def test_classify_extraction_partial() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    content, quality = enrichment.classify_extraction(text, "snippet")
    assert quality == "partial"
    assert content == text


def test_classify_extraction_link_heavy_page_is_not_full() -> None:
    text = " ".join(f"https://example.com/{i} link" for i in range(300))
    _, quality = enrichment.classify_extraction(text, "snippet")
    assert quality == "partial"


def test_classify_extraction_snippet_fallback() -> None:
    content, quality = enrichment.classify_extraction(None, "a useful search snippet")
    assert quality == "snippet_only"
    assert content == "a useful search snippet"

    thin = "too thin"
    content, quality = enrichment.classify_extraction(thin, "a much longer and richer search snippet text")
    assert quality == "snippet_only"


def test_classify_extraction_metadata_only() -> None:
    content, quality = enrichment.classify_extraction("", "")
    assert quality == "metadata_only"
    assert content == ""


def test_content_metrics() -> None:
    word_count, link_density = enrichment.content_metrics("plain words only here")
    assert word_count == 4
    assert link_density == 0.0
    word_count, link_density = enrichment.content_metrics("see https://a.com and https://b.com now")
    assert word_count == 5
    assert link_density == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# Durable scrape cache (index write/read/TTL/backfill)
# ---------------------------------------------------------------------------

scrape_artifacts = pytest.importorskip("aiq_agent.common.scrape_artifacts")

_DOC_BLOCK = (
    '<document idx="0">\n'
    "<title>Example Report</title>\n"
    "<url>https://example.com/report</url>\n"
    "<engines>searxng</engines>\n"
    "<extraction_quality>full</extraction_quality>\n"
    "<published_at>2026-05-12</published_at>\n"
    "<content>Example body content for the cache.</content>\n"
    "</document>"
)


@pytest.fixture
def artifact_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "scrape_artifacts"
    monkeypatch.setenv("AIQ_SCRAPE_ARTIFACT_DIR", str(root))
    yield root


def test_cache_write_then_read(artifact_root) -> None:
    artifacts = scrape_artifacts.persist_scrape_artifacts(
        job_id="job-1", tool_name="web_search_tool", tool_output=_DOC_BLOCK
    )
    assert len(artifacts) == 1
    assert artifacts[0].published_at == "2026-05-12"
    assert artifacts[0].extraction_quality == "full"

    payload = scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600)

    assert payload is not None
    assert payload["content"] == "Example body content for the cache."
    assert payload["published_at"] == "2026-05-12"
    assert payload["extraction_quality"] == "full"


def test_cache_miss_for_unknown_url(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    assert scrape_artifacts.find_cached_scrape("https://other.example.com/x", max_age_seconds=3600) is None


def test_cache_ttl_expiry(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    assert scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=0) is None
    assert scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600) is not None


def test_cache_url_normalization_ignores_trailing_slash(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    assert scrape_artifacts.find_cached_scrape("https://example.com/report/", max_age_seconds=3600) is not None


def test_cache_backfill_from_existing_artifacts(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    # Simulate a pre-index deployment: remove the sqlite index entirely.
    for suffix in ("", "-wal", "-shm"):
        index_file = artifact_root / f"artifacts_index.db{suffix}"
        if index_file.exists():
            index_file.unlink()

    indexed = scrape_artifacts.scan_existing()
    assert indexed >= 1

    payload = scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600)
    assert payload is not None
    assert payload["url"] == "https://example.com/report"


def test_cache_lazy_backfill_when_index_missing(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    for suffix in ("", "-wal", "-shm"):
        index_file = artifact_root / f"artifacts_index.db{suffix}"
        if index_file.exists():
            index_file.unlink()
    scrape_artifacts._backfilled_roots.discard(str(artifact_root))

    payload = scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600)

    assert payload is not None


def test_cache_handles_corrupt_index_as_miss(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    index_file = artifact_root / "artifacts_index.db"
    index_file.write_bytes(b"this is not a sqlite database at all" * 10)
    scrape_artifacts._backfilled_roots.add(str(artifact_root))

    assert scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600) is None


def test_cache_handles_missing_artifact_file_as_miss(artifact_root) -> None:
    artifacts = scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    import pathlib

    pathlib.Path(artifacts[0].artifact_path).unlink()

    assert scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600) is None


def test_newer_artifact_wins_in_index(artifact_root) -> None:
    scrape_artifacts.persist_scrape_artifacts(job_id="job-1", tool_name="t", tool_output=_DOC_BLOCK)
    updated_block = _DOC_BLOCK.replace("Example body content for the cache.", "Updated body content.")
    scrape_artifacts.persist_scrape_artifacts(job_id="job-2", tool_name="t", tool_output=updated_block)

    payload = scrape_artifacts.find_cached_scrape("https://example.com/report", max_age_seconds=3600)

    assert payload is not None
    assert payload["content"] == "Updated body content."
    assert payload["job_id"] == "job-2"


def test_scan_existing_indexes_raw_artifact_files(artifact_root) -> None:
    job_dir = artifact_root / "legacy-job"
    job_dir.mkdir(parents=True)
    payload = {
        "job_id": "legacy-job",
        "url": "https://legacy.example.com/page",
        "title": "Legacy",
        "extraction_status": "extracted",
        "content_hash": "abc",
        "content": "Legacy content",
        "content_length": 14,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with gzip.open(job_dir / "legacy.example.com-aaaa-bbbb.json.gz", "wt", encoding="utf-8") as fp:
        json.dump(payload, fp)

    assert scrape_artifacts.scan_existing() >= 1
    cached = scrape_artifacts.find_cached_scrape("https://legacy.example.com/page", max_age_seconds=3600)
    assert cached is not None
    assert cached["content"] == "Legacy content"


# ---------------------------------------------------------------------------
# Rate limiting knobs
# ---------------------------------------------------------------------------


def test_discovery_semaphore_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_SEARCH_DISCOVERY_CONCURRENCY", "2")
    register._DISCOVERY_SEMAPHORES.clear()
    semaphore = register._discovery_semaphore("searxng")
    assert semaphore._value == 2
    # Same backend reuses the same semaphore; other backends get their own.
    assert register._discovery_semaphore("searxng") is semaphore
    assert register._discovery_semaphore("ddgs") is not semaphore
    register._DISCOVERY_SEMAPHORES.clear()


def test_global_scrape_semaphore_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_SCRAPE_CONCURRENCY", "3")
    register._GLOBAL_SCRAPE_SEMAPHORE = None
    semaphore = register._global_scrape_semaphore()
    assert semaphore._value == 3
    register._GLOBAL_SCRAPE_SEMAPHORE = None


def test_scrape_cache_ttl_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIQ_SCRAPE_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("AIQ_SCRAPE_CACHE_RECENT_TTL_SECONDS", raising=False)
    assert register._scrape_cache_ttl_seconds(False) == 604800.0
    assert register._scrape_cache_ttl_seconds(True) == 21600.0
    monkeypatch.setenv("AIQ_SCRAPE_CACHE_RECENT_TTL_SECONDS", "60")
    assert register._scrape_cache_ttl_seconds(True) == 60.0


def test_scrape_cache_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIQ_SCRAPE_CACHE_ENABLED", raising=False)
    assert register._scrape_cache_enabled() is True
    monkeypatch.setenv("AIQ_SCRAPE_CACHE_ENABLED", "0")
    assert register._scrape_cache_enabled() is False

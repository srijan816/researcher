"""SearXNG + concurrent page extraction web search tool for AI-Q.

The tool uses a local/self-hosted SearXNG JSON endpoint for broad web
discovery, then expands top result URLs through Scrapling or Jina Reader so
research agents see page content instead of snippets only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from html import escape
from typing import Literal
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import quote
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

DEFAULT_BLOCKED_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "reddit.com",
    "youtube.com",
    "youtu.be",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "pinterest.com",
)

LOW_VALUE_URL_PARTS = (
    "/login",
    "/signin",
    "/sign_in",
    "/account",
    "/accounts/",
    "/recover",
    "/password",
    "/privacy",
    "/terms",
    "/careers",
    "/help",
    "/support",
    "/contact",
)


class SearXNGJinaWebSearchToolConfig(FunctionBaseConfig, name="searxng_jina_web_search"):
    """Search SearXNG and optionally extract full page text from top results."""

    searxng_url: str = Field(
        default="http://localhost:8080",
        description="Base URL of the SearXNG instance.",
    )
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum de-duplicated SearXNG results to return.")
    max_content_length: int | None = Field(
        default=4000,
        description="Maximum characters per result. Set null to disable truncation.",
    )
    categories: str = Field(default="general", description="SearXNG categories parameter.")
    language: str = Field(default="en", description="SearXNG language parameter.")
    engines: str | None = Field(
        default=None,
        description="Optional comma-separated SearXNG engines. Leave null for instance defaults.",
    )
    safe_search: int = Field(default=0, ge=0, le=2, description="SearXNG safesearch: 0 off, 1 moderate, 2 strict.")
    time_range: str | None = Field(default=None, description="Optional SearXNG time_range value.")
    search_pages: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of SearXNG result pages to fetch for each search tool call.",
    )
    search_page_concurrency: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Maximum SearXNG result pages to fetch concurrently.",
    )
    discovery_backend: Literal["searxng", "ddgs", "hybrid", "websurfx", "websurfx_hybrid"] = Field(
        default="searxng",
        description=(
            "URL discovery backend. Hybrid queries SearXNG and DDGS concurrently; "
            "websurfx queries a Websurfx JSON endpoint; websurfx_hybrid queries all three."
        ),
    )
    websurfx_url: str = Field(
        default="http://localhost:8081",
        description="Base URL of the Websurfx instance when discovery_backend uses Websurfx.",
    )
    websurfx_engines: str = Field(
        default="",
        description=(
            "Optional Websurfx-specific engine list. When blank, Websurfx reuses the generic engines setting."
        ),
    )
    ddgs_backend: str = Field(
        default="auto",
        description=(
            "DDGS backend for hybrid/DDGS discovery: auto, bing, brave, duckduckgo, google, mojeek, yandex, yahoo."
        ),
    )
    ddgs_max_results: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum DDGS discovery results to request when discovery_backend is ddgs or hybrid.",
    )
    extraction_backend: Literal["scrapling", "jina", "none"] = Field(
        default="scrapling",
        description="Page extraction backend for top search results.",
    )
    enable_jina: bool = Field(
        default=True,
        description="Backward-compatible extraction toggle. Set extraction_backend='none' to disable extraction.",
    )
    jina_max_results: int = Field(
        default=4,
        ge=0,
        le=100,
        description="Backward-compatible number of top results to extract.",
    )
    jina_base_url: str = Field(default="https://r.jina.ai", description="Jina Reader base URL.")
    scrape_max_results: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Number of top results to extract. Defaults to jina_max_results for compatibility.",
    )
    scrape_concurrency: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Maximum number of result URLs to scrape concurrently.",
    )
    scrapling_impersonate: str | None = Field(
        default="chrome",
        description="Scrapling browser TLS/header impersonation profile.",
    )
    scrapling_stealthy_headers: bool = Field(
        default=True,
        description="Use Scrapling stealthy headers for HTTP extraction.",
    )
    request_timeout: int = Field(default=20, ge=3, le=120, description="HTTP timeout in seconds.")
    max_retries: int = Field(default=2, ge=1, le=5, description="Maximum retry attempts per request.")
    blocked_domains: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BLOCKED_DOMAINS),
        description="Domains to exclude from results before optional Jina expansion.",
    )
    filter_low_value_urls: bool = Field(
        default=True,
        description="Exclude login, account, terms, privacy, and similar low-value URLs.",
    )


def _truncate(content: str, max_length: int | None) -> str:
    if max_length and len(content) > max_length:
        return content[: max_length - 3] + "..."
    return content


def _read_url(url: str, timeout: int) -> str:
    req = Request(url, headers={"User-Agent": "AI-Q-Research/1.0"})
    with urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _jina_reader_url(base_url: str, target_url: str) -> str:
    base = base_url.rstrip("/")
    if target_url.startswith(("https://", "http://")):
        return f"{base}/{target_url}"
    return f"{base}/http://{target_url}"


def _is_blocked_url(url: str, blocked_domains: Sequence[str], filter_low_value_urls: bool) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return True

    for domain in blocked_domains:
        normalized = domain.lower().lstrip(".")
        if hostname == normalized or hostname.endswith(f".{normalized}"):
            return True

    if filter_low_value_urls:
        lowered_url = url.lower()
        if any(part in lowered_url for part in LOW_VALUE_URL_PARTS):
            return True

    return False


def _clean_content(content: str) -> str:
    """Keep extracted pages dense enough for an LLM without nav/link clutter."""
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        if lowered.startswith(("![", "[![", "skip to ", "cookie", "privacy policy")):
            continue
        if line.count("http") > 2 or line.count(")[") > 2:
            continue
        seen.add(lowered)
        lines.append(line)
    return "\n".join(lines)


def _query_terms(query: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9.\-]{2,}", query)}


def _result_rank(result: dict, query: str) -> float:
    """Lightweight ranker for merged SearXNG/DDGS results."""
    title = str(result.get("title") or "")
    url = str(result.get("url") or "")
    content = str(result.get("content") or "")
    haystack = f"{title} {url} {content}".lower()
    terms = _query_terms(query)
    score = 0.0
    score += sum(1.0 for term in terms if term in haystack)

    hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
    official_domains = (
        "minimax.io",
        "platform.minimax.io",
        "platform.minimaxi.com",
        "api.minimax.io",
        "github.com",
        "huggingface.co",
        "developer.nvidia.com",
        "docs.aws.amazon.com",
    )
    if any(hostname == domain or hostname.endswith(f".{domain}") for domain in official_domains):
        score += 4.0
    if hostname.endswith("minimax.io") or hostname.endswith("minimaxi.com"):
        score += 4.0

    # Preserve upstream relevance where available without letting it dominate.
    try:
        score += min(float(result.get("score") or 0.0), 10.0) / 5.0
    except (TypeError, ValueError):
        pass
    if result.get("_discovery_backend") in {"searxng", "websurfx"}:
        score += 0.5
    return score


def _normalize_websurfx_result(item: dict) -> dict | None:
    """Map Websurfx camelCase JSON results into the SearXNG-shaped result dict."""
    url = str(item.get("url") or "").strip()
    if not url:
        return None
    engines = item.get("engine") or item.get("engines") or []
    if isinstance(engines, str):
        engines = [engines]
    if not isinstance(engines, list):
        engines = []
    try:
        score = float(item.get("relevanceScore") or item.get("relevance_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "title": str(item.get("title") or url),
        "url": url,
        "content": str(item.get("description") or item.get("content") or ""),
        "engines": [f"websurfx:{engine}" for engine in engines] or ["websurfx"],
        "score": score,
        "_discovery_backend": "websurfx",
    }


async def _scrapling_extract(url: str, tool_config: SearXNGJinaWebSearchToolConfig) -> str:
    """Fetch a URL with Scrapling and return dense visible page text."""
    try:
        from scrapling.fetchers import AsyncFetcher
    except ImportError as exc:  # pragma: no cover - only hit when optional dependency is missing
        raise RuntimeError("Scrapling is not installed. Install scrapling[fetchers].") from exc

    kwargs = {
        "timeout": tool_config.request_timeout,
        "stealthy_headers": tool_config.scrapling_stealthy_headers,
        "follow_redirects": "safe",
        "retries": tool_config.max_retries,
        "retry_delay": 1,
    }
    if tool_config.scrapling_impersonate:
        kwargs["impersonate"] = tool_config.scrapling_impersonate

    page = await AsyncFetcher.get(url, **kwargs)
    status = getattr(page, "status", None)
    if isinstance(status, int) and status >= 400:
        raise RuntimeError(f"Scrapling fetch returned HTTP {status}")
    return _clean_content(str(page.get_all_text(separator="\n", strip=True)))


@register_function(config_type=SearXNGJinaWebSearchToolConfig)
async def searxng_jina_web_search(tool_config: SearXNGJinaWebSearchToolConfig, builder: Builder):
    configured_url = os.environ.get("SEARXNG_URL") or tool_config.searxng_url
    configured_websurfx_url = os.environ.get("WEBSURFX_URL") or tool_config.websurfx_url

    async def _fetch_text(url: str) -> str:
        loop = asyncio.get_event_loop()
        last_error: Exception | None = None
        for attempt in range(tool_config.max_retries):
            try:
                return await loop.run_in_executor(None, _read_url, url, tool_config.request_timeout)
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < tool_config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(str(last_error) if last_error else "request failed")

    async def _search(question: str) -> str:
        """Searches the web using SearXNG and expands top results with Jina Reader.

        Args:
            question: Search query, truncated to 400 characters for compatibility.

        Returns:
            XML-like documents containing title, URL, engine metadata, and page content.
        """
        query = question[:397] + "..." if len(question) > 400 else question
        base_url = configured_url.rstrip("/")
        params = {
            "q": query,
            "format": "json",
            "categories": tool_config.categories,
            "language": tool_config.language,
            "safesearch": tool_config.safe_search,
        }
        if tool_config.engines:
            params["engines"] = tool_config.engines
        if tool_config.time_range:
            params["time_range"] = tool_config.time_range

        async def _fetch_searxng_page(page_no: int) -> list[dict]:
            page_params = {**params, "pageno": page_no}
            search_url = f"{base_url}/search?{urlencode(page_params)}"
            payload = await _fetch_text(search_url)
            data = json.loads(payload)
            raw_results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(raw_results, list):
                return []
            for result in raw_results:
                if isinstance(result, dict):
                    result["_discovery_backend"] = "searxng"
            return raw_results

        async def _fetch_searxng_results() -> list[dict]:
            page_semaphore = asyncio.Semaphore(tool_config.search_page_concurrency)

            async def _guarded_page(page_no: int) -> list[dict]:
                async with page_semaphore:
                    return await _fetch_searxng_page(page_no)

            pages = await asyncio.gather(
                *(_guarded_page(page_no) for page_no in range(1, tool_config.search_pages + 1))
            )
            return [result for page in pages for result in page]

        async def _fetch_ddgs_results() -> list[dict]:
            try:
                from ddgs import DDGS
            except ImportError:
                logger.warning("DDGS discovery requested but ddgs is not installed.")
                return []

            def _do_ddgs_search() -> list[dict]:
                client = DDGS()
                raw = client.text(
                    query,
                    max_results=tool_config.ddgs_max_results,
                    backend=tool_config.ddgs_backend,
                )
                normalized = []
                for item in raw or []:
                    if not isinstance(item, dict):
                        continue
                    normalized.append(
                        {
                            "title": item.get("title") or "",
                            "url": item.get("href") or item.get("url") or "",
                            "content": item.get("body") or item.get("content") or "",
                            "engines": [f"ddgs:{tool_config.ddgs_backend}"],
                            "_discovery_backend": "ddgs",
                        }
                    )
                return normalized

            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(None, _do_ddgs_search)
            except Exception as exc:
                logger.warning("DDGS discovery failed for %r: %s", query, exc)
                return []

        async def _fetch_websurfx_page(page_no: int) -> list[dict]:
            websurfx_params = {
                "q": query,
                "page": page_no,
                "safesearch": tool_config.safe_search,
                "json": "true",
            }
            websurfx_engines = (tool_config.websurfx_engines or tool_config.engines or "").strip()
            if websurfx_engines:
                websurfx_params["engines"] = websurfx_engines
            search_url = f"{configured_websurfx_url.rstrip('/')}/search?{urlencode(websurfx_params)}"
            payload = await _fetch_text(search_url)
            data = json.loads(payload)
            raw_results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(raw_results, list):
                return []
            normalized = []
            for item in raw_results:
                if isinstance(item, dict):
                    result = _normalize_websurfx_result(item)
                    if result:
                        normalized.append(result)
            return normalized

        async def _fetch_websurfx_results() -> list[dict]:
            page_semaphore = asyncio.Semaphore(tool_config.search_page_concurrency)

            async def _guarded_page(page_no: int) -> list[dict]:
                async with page_semaphore:
                    return await _fetch_websurfx_page(page_no)

            try:
                pages = await asyncio.gather(
                    *(_guarded_page(page_no) for page_no in range(1, tool_config.search_pages + 1))
                )
                return [result for page in pages for result in page]
            except Exception as exc:
                logger.warning("Websurfx discovery failed for %r: %s", query, exc)
                return []

        discovery_tasks = []
        if tool_config.discovery_backend in {"searxng", "hybrid", "websurfx_hybrid"}:
            discovery_tasks.append(_fetch_searxng_results())
        if tool_config.discovery_backend in {"ddgs", "hybrid", "websurfx_hybrid"}:
            discovery_tasks.append(_fetch_ddgs_results())
        if tool_config.discovery_backend in {"websurfx", "websurfx_hybrid"}:
            discovery_tasks.append(_fetch_websurfx_results())

        try:
            discovery_results = await asyncio.gather(*discovery_tasks)
        except Exception as exc:
            return f"Error: web search discovery failed - {exc}"

        raw_results = [result for group in discovery_results for result in group]
        if not raw_results:
            return "No results found for this query."

        results = []
        seen_urls = set()
        for result in raw_results:
            if not isinstance(result, dict) or not result.get("url"):
                continue
            url = str(result.get("url"))
            normalized_url = url.rstrip("/")
            if normalized_url in seen_urls:
                continue
            if _is_blocked_url(url, tool_config.blocked_domains, tool_config.filter_low_value_urls):
                continue
            seen_urls.add(normalized_url)
            results.append(result)
        results.sort(key=lambda result: _result_rank(result, query), reverse=True)
        results = results[: tool_config.max_results]
        if not results:
            return "No results found for this query."

        extracted: dict[str, str] = {}
        extract_limit = tool_config.scrape_max_results
        if extract_limit is None:
            extract_limit = tool_config.jina_max_results
        should_extract = tool_config.enable_jina and tool_config.extraction_backend != "none" and extract_limit > 0
        if should_extract:
            targets = [str(r["url"]) for r in results[:extract_limit]]

            async def _extract_with_jina(url: str) -> tuple[str, str | None]:
                try:
                    reader_url = _jina_reader_url(tool_config.jina_base_url, quote(url, safe=":/?&=%#[]@!$'()*+,;"))
                    return url, await _fetch_text(reader_url)
                except Exception as exc:
                    logger.warning("Jina Reader extraction failed for %s: %s", url, exc)
                    return url, None

            scrape_semaphore = asyncio.Semaphore(tool_config.scrape_concurrency)

            async def _extract_with_scrapling(url: str) -> tuple[str, str | None]:
                try:
                    async with scrape_semaphore:
                        return url, await _scrapling_extract(url, tool_config)
                except Exception as exc:
                    logger.warning("Scrapling extraction failed for %s: %s", url, exc)
                    return url, None

            extractor = _extract_with_jina if tool_config.extraction_backend == "jina" else _extract_with_scrapling
            for url, content in await asyncio.gather(*(extractor(url) for url in targets)):
                if content:
                    extracted[url] = _clean_content(content)

        documents: list[str] = []
        for idx, result in enumerate(results):
            url = str(result.get("url", ""))
            title = str(result.get("title") or url)
            snippet = str(result.get("content") or "")
            content = extracted.get(url) or snippet
            engines = result.get("engines") or result.get("engine") or ""
            if isinstance(engines, list):
                engines = ", ".join(str(engine) for engine in engines)
            content = _truncate(_clean_content(content), tool_config.max_content_length)
            documents.append(
                f'<document idx="{idx}">\n'
                f"<title>{escape(title)}</title>\n"
                f"<url>{escape(url)}</url>\n"
                f"<engines>{escape(str(engines))}</engines>\n"
                f"<content>{escape(content)}</content>\n"
                f"</document>"
            )

        return "\n\n---\n\n".join(documents)

    yield FunctionInfo.from_fn(_search, description=_search.__doc__)

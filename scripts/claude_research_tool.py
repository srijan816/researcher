#!/usr/bin/env python3
"""Repo-local search/scrape CLI for Claude Code research runs.

The tool is intentionally small and boring. Claude Code can call it from a shell
without needing to understand AI-Q internals. It writes ordinary run-folder
artifacts that the app can inspect later.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

try:
    from aiq_agent.common.source_classification import classify_url
except Exception:  # pragma: no cover - optional when used outside editable env
    classify_url = None  # type: ignore[assignment]


DEFAULT_WEBSURFX_ENGINES = "Searx,Brave,DuckDuckGo,LibreX,Mojeek,Qwant,Startpage,Yahoo,Bing,SepiaSearch"
RUN_FILES = {
    "README.md": "# Claude Research Run\n\nThis folder is managed by `scripts/claude_research_tool.py`.\n",
    "plan.md": "# Research Plan\n\n",
    "queries.json": '{\n  "modules": []\n}\n',
    "sources.json": '{\n  "sources": []\n}\n',
    "contradictions.md": "# Contradictions\n\n",
    "gaps.md": "# Gaps\n\n",
    "research.md": "# Research Compile\n\n",
    "final.md": "# Final Report\n\n",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str, *, fallback: str = "item") -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    return value[:120] or fallback


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_dir(path: str | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None


def init_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("notes", "source_summaries", "logs", "raw"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    for filename, content in RUN_FILES.items():
        target = run_dir / filename
        if not target.exists():
            target.write_text(content, encoding="utf-8")


def _classify(url: str) -> dict[str, Any]:
    if classify_url is None:
        return {"source_class": "unknown", "classification_reason": "classifier unavailable"}
    result = classify_url(url)
    return {
        "source_class": result.source_class.value,
        "classification_reason": result.classification_reason,
        "normalized_domain": result.normalized_domain,
        "classification_confidence": result.confidence,
    }


def _normalize_result(item: dict[str, Any], backend: str) -> dict[str, Any] | None:
    url = item.get("url") or item.get("href")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    title = item.get("title") or item.get("name") or url
    snippet = item.get("content") or item.get("body") or item.get("snippet") or item.get("description") or ""
    engines = item.get("engines") or item.get("engine") or []
    if isinstance(engines, str):
        engines = [engines]
    normalized = {
        "id": hashlib.sha1(url.rstrip("/").encode("utf-8")).hexdigest()[:12],
        "url": url,
        "title": str(title),
        "snippet": str(snippet),
        "backend": backend,
        "engines": [str(engine) for engine in engines] if isinstance(engines, list) else [],
        "discovered_at": _now(),
    }
    normalized.update(_classify(url))
    return normalized


def _request_json(url: str, *, timeout: float) -> dict[str, Any]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"accept": "application/json", "user-agent": "aiq-claude-research/0.1"},
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _search_searxng(query: str, *, page: int, timeout: float, limit: int) -> list[dict[str, Any]]:
    base_url = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8080").rstrip("/")
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
        "language": "en",
        "safesearch": os.environ.get("AIQ_CLAUDE_RESEARCH_SAFE_SEARCH", "2"),
        "pageno": page,
    }
    payload = _request_json(f"{base_url}/search?{urlencode(params)}", timeout=timeout)
    results = payload.get("results") if isinstance(payload, dict) else []
    normalized = [_normalize_result(item, "searxng") for item in results[:limit] if isinstance(item, dict)]
    return [item for item in normalized if item is not None]


def _search_websurfx(query: str, *, page: int, timeout: float, limit: int) -> list[dict[str, Any]]:
    base_url = os.environ.get("WEBSURFX_URL", "http://127.0.0.1:8081").rstrip("/")
    params = {
        "q": query,
        "json": "true",
        "page": page,
        "safesearch": os.environ.get("AIQ_CLAUDE_RESEARCH_SAFE_SEARCH", "2"),
        "engines": os.environ.get("WEBSURFX_ENGINES", DEFAULT_WEBSURFX_ENGINES),
    }
    try:
        payload = _request_json(f"{base_url}/search?{urlencode(params)}", timeout=timeout)
    except Exception:
        # Websurfx can stall when one configured upstream engine is slow. Retry
        # with its server-side defaults before giving up on the whole backend.
        params.pop("engines", None)
        payload = _request_json(f"{base_url}/search?{urlencode(params)}", timeout=timeout)
    results = payload.get("results") if isinstance(payload, dict) else []
    normalized = [_normalize_result(item, "websurfx") for item in results[:limit] if isinstance(item, dict)]
    return [item for item in normalized if item is not None]


def _search_ddgs(query: str, *, limit: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    results = DDGS().text(query, max_results=limit)
    normalized = [_normalize_result(item, "ddgs") for item in results if isinstance(item, dict)]
    return [item for item in normalized if item is not None]


def search(query: str, *, backend: str, pages: int, limit: int, timeout: float) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for page in range(1, pages + 1):
        if backend in {"searxng", "hybrid", "websurfx_hybrid"}:
            try:
                all_results.extend(_search_searxng(query, page=page, timeout=timeout, limit=limit))
            except Exception as exc:
                failures.append(f"searxng page {page}: {exc}")
        if backend in {"websurfx", "hybrid", "websurfx_hybrid"}:
            try:
                all_results.extend(_search_websurfx(query, page=page, timeout=timeout, limit=limit))
            except Exception as exc:
                failures.append(f"websurfx page {page}: {exc}")
    if backend in {"hybrid", "websurfx_hybrid"} and not all_results:
        try:
            all_results.extend(_search_ddgs(query, limit=limit))
        except Exception as exc:
            failures.append(f"ddgs: {exc}")
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in all_results:
        key = str(item.get("url", "")).rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    if failures and not deduped:
        raise RuntimeError("; ".join(failures))
    return deduped


def _append_sources(run_dir: Path, results: list[dict[str, Any]]) -> None:
    init_run(run_dir)
    sources_path = run_dir / "sources.json"
    data = _read_json(sources_path, {"sources": []})
    existing = data.get("sources") if isinstance(data.get("sources"), list) else []
    by_url = {str(item.get("url", "")).rstrip("/"): item for item in existing if isinstance(item, dict)}
    for result in results:
        key = str(result.get("url", "")).rstrip("/")
        by_url[key] = {**by_url.get(key, {}), **result}
    data["sources"] = list(by_url.values())
    _write_json(sources_path, data)


def _extract_text_with_scrapling(url: str, *, timeout: float) -> tuple[str, str]:
    from scrapling.fetchers import Fetcher

    page = Fetcher.get(url, timeout=timeout)
    text = ""
    title = ""
    for attr in ("text", "body"):
        value = getattr(page, attr, None)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if not text:
        html_value = getattr(page, "html", None) or str(page)
        soup = BeautifulSoup(str(html_value), "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = soup.get_text("\n", strip=True)
    else:
        title_value = getattr(page, "title", None)
        title = str(title_value).strip() if title_value else ""
    return title, text


def _extract_text_with_requests(url: str, *, timeout: float) -> tuple[str, str]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"user-agent": "aiq-claude-research/0.1"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text" not in content_type and "html" not in content_type and "json" not in content_type:
        return url, response.text
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    return title, soup.get_text("\n", strip=True)


def scrape(url: str, *, timeout: float, max_chars: int) -> dict[str, Any]:
    errors: list[str] = []
    title = url
    text = ""
    try:
        title, text = _extract_text_with_scrapling(url, timeout=timeout)
        if _looks_like_fetcher_placeholder(text):
            raise RuntimeError("scrapling returned fetch metadata instead of extracted page text")
    except Exception as exc:
        errors.append(f"scrapling: {exc}")
        try:
            title, text = _extract_text_with_requests(url, timeout=timeout)
        except Exception as fallback_exc:
            errors.append(f"requests: {fallback_exc}")
            raise RuntimeError("; ".join(errors)) from fallback_exc
    text = re.sub(r"\n{3,}", "\n\n", html.unescape(text)).strip()
    result = {
        "id": hashlib.sha1(url.rstrip("/").encode("utf-8")).hexdigest()[:12],
        "url": url,
        "title": title or url,
        "text": text[:max_chars],
        "text_length": len(text),
        "scraped_at": _now(),
        "errors": errors,
    }
    result.update(_classify(url))
    return result


def _looks_like_fetcher_placeholder(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"<\d{3}\s+https?://[^>]+>", stripped))


def _write_source_summary(run_dir: Path, result: dict[str, Any]) -> Path:
    init_run(run_dir)
    source_id = str(result.get("id") or hashlib.sha1(str(result.get("url", "")).encode()).hexdigest()[:12])
    target = run_dir / "source_summaries" / f"{_safe_name(source_id)}.md"
    content = (
        f"# {result.get('title') or result.get('url')}\n\n"
        f"- URL: {result.get('url')}\n"
        f"- Source class: {result.get('source_class', 'unknown')}\n"
        f"- Classification reason: {result.get('classification_reason', 'n/a')}\n"
        f"- Scraped at: {result.get('scraped_at')}\n"
        f"- Text length: {result.get('text_length')}\n\n"
        "## Extracted Text\n\n"
        f"{result.get('text', '')}\n"
    )
    target.write_text(content, encoding="utf-8")
    return target


def cmd_init_run(args: argparse.Namespace) -> int:
    init_run(Path(args.run_dir).expanduser().resolve())
    print(json.dumps({"ok": True, "run_dir": str(Path(args.run_dir).expanduser().resolve())}, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    results = search(args.query, backend=args.backend, pages=args.pages, limit=args.limit, timeout=args.timeout)
    run_dir = _run_dir(args.run_dir)
    if run_dir:
        _append_sources(run_dir, results)
    print(
        json.dumps(
            {"query": args.query, "backend": args.backend, "count": len(results), "results": results},
            indent=2,
        )
    )
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    result = scrape(args.url, timeout=args.timeout, max_chars=args.max_chars)
    run_dir = _run_dir(args.run_dir)
    if run_dir:
        summary_path = _write_source_summary(run_dir, result)
        source_record = {k: v for k, v in result.items() if k != "text"}
        source_record["summary_path"] = str(summary_path)
        _append_sources(run_dir, [source_record])
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and scrape helper for Claude Code research runs.")
    subparsers = parser.add_subparsers(required=True)

    init_parser = subparsers.add_parser("init-run", help="Create the run folder and required artifact files.")
    init_parser.add_argument("--run-dir", required=True)
    init_parser.set_defaults(func=cmd_init_run)

    search_parser = subparsers.add_parser("search", help="Search SearXNG/Websurfx and optionally append sources.json.")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--backend",
        choices=["searxng", "websurfx", "hybrid", "websurfx_hybrid"],
        default="websurfx_hybrid",
    )
    search_parser.add_argument("--pages", type=int, default=1)
    search_parser.add_argument("--limit", type=int, default=8)
    search_parser.add_argument("--timeout", type=float, default=12.0)
    search_parser.add_argument("--run-dir")
    search_parser.set_defaults(func=cmd_search)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape a URL and optionally write source_summaries/*.md.")
    scrape_parser.add_argument("url")
    scrape_parser.add_argument("--timeout", type=float, default=20.0)
    scrape_parser.add_argument("--max-chars", type=int, default=12000)
    scrape_parser.add_argument("--run-dir")
    scrape_parser.set_defaults(func=cmd_scrape)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

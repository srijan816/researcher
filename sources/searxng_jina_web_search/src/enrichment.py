# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort recency and extraction-quality enrichment for web search results.

Everything in this module is fail-open: helpers return ``None`` or neutral
defaults instead of raising, so search never breaks because enrichment failed.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import UTC
from datetime import datetime

logger = logging.getLogger(__name__)

# Extraction-quality thresholds.
FULL_MIN_WORDS = 400
PARTIAL_MIN_WORDS = 120
MAX_SANE_LINK_DENSITY = 0.10

# Freshness decay: score halves roughly every FRESHNESS_HALF_LIFE_DAYS.
FRESHNESS_HALF_LIFE_DAYS = 45.0
UNKNOWN_DATE_FRESHNESS = 0.5

_URL_DATE_PATTERNS = (
    re.compile(r"(?P<year>19\d{2}|20\d{2})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])"),
    re.compile(r"/(?P<year>19\d{2}|20\d{2})/(?P<month>0[1-9]|1[0-2])(?:/(?P<day>0[1-9]|[12]\d|3[01]))?(?:/|$)"),
)

_META_DATE_PATTERNS = (
    # <meta property="article:published_time" content="2026-05-12T08:30:00Z">
    re.compile(
        r"(?is)<meta[^>]+(?:property|name|itemprop)\s*=\s*[\"']"
        r"(?:article:published_time|og:published_time|datePublished|date|dc\.date(?:\.issued)?|publish-date|"
        r"publication_date|sailthru\.date|parsely-pub-date)[\"'][^>]*?content\s*=\s*[\"']([^\"']+)[\"']"
    ),
    # content attribute appearing before property/name.
    re.compile(
        r"(?is)<meta[^>]+content\s*=\s*[\"']([^\"']+)[\"'][^>]*?(?:property|name|itemprop)\s*=\s*[\"']"
        r"(?:article:published_time|og:published_time|datePublished|date|dc\.date(?:\.issued)?|publish-date|"
        r"publication_date|sailthru\.date|parsely-pub-date)[\"']"
    ),
    # JSON-LD: "datePublished": "2026-05-12"
    re.compile(r"(?i)\"datePublished\"\s*:\s*\"([^\"]+)\""),
    # Jina Reader markdown header: "Published Time: 2026-05-12T08:30:00Z"
    re.compile(r"(?im)^published\s+time:\s*(\S+)\s*$"),
    # <time datetime="2026-05-12">
    re.compile(r"(?is)<time[^>]+datetime\s*=\s*[\"']([^\"']+)[\"']"),
)

_RESULT_DATE_FIELDS = ("publishedDate", "published_date", "publishedAt", "published_at", "pubDate", "date")

_MONTH_RANGE_TERMS = (
    "latest",
    "recent",
    "recently",
    "today",
    "yesterday",
    "this week",
    "this month",
    "news",
    "newest",
    "current",
    "currently",
    "breaking",
    "right now",
    "as of now",
    "up to date",
    "up-to-date",
)

_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_LINK_RE = re.compile(r"https?://")

_VALID_TIME_RANGES = {"day", "week", "month", "year"}


def _coerce_iso_date(value: object) -> str | None:
    """Normalize assorted date strings/objects to an ISO ``YYYY-MM-DD`` string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        pass
    # Common non-ISO formats seen in meta tags and search engines.
    head = text[:25]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%b %d, %Y", "%a, %d %b %Y %H:%M:%S"):
        try:
            return datetime.strptime(head, fmt).date().isoformat()
        except ValueError:
            continue
    match = _URL_DATE_PATTERNS[0].search(text)
    if match:
        return _date_from_match(match)
    return None


def _date_from_match(match: re.Match[str]) -> str | None:
    try:
        year = int(match.group("year"))
        month = int(match.group("month"))
        day_raw = match.groupdict().get("day")
        day = int(day_raw) if day_raw else 1
        return datetime(year, month, day).date().isoformat()
    except (ValueError, IndexError):
        return None


def extract_published_date_from_url(url: str) -> str | None:
    """Best-effort published date from URL path patterns like ``/2026/05/`` or ``2026-05-12``."""
    if not url:
        return None
    for pattern in _URL_DATE_PATTERNS:
        match = pattern.search(url)
        if match:
            date = _date_from_match(match)
            if date:
                return date
    return None


def extract_published_date_from_html(html_text: str | None) -> str | None:
    """Best-effort published date from raw HTML meta tags, JSON-LD, or reader output."""
    if not html_text:
        return None
    head = html_text[:60000]
    for pattern in _META_DATE_PATTERNS:
        match = pattern.search(head)
        if match:
            date = _coerce_iso_date(match.group(1))
            if date:
                return date
    return None


def extract_published_date_from_result(result: dict) -> str | None:
    """Best-effort published date from a discovery result dict (SearXNG fields, then URL)."""
    if not isinstance(result, dict):
        return None
    for field in _RESULT_DATE_FIELDS:
        date = _coerce_iso_date(result.get(field))
        if date:
            return date
    return extract_published_date_from_url(str(result.get("url") or ""))


def detect_recency_intent(query: str, *, now: datetime | None = None) -> str | None:
    """Return a time range ("month"/"year") when the query asks for recent information."""
    text = (query or "").lower()
    if not text:
        return None
    padded = f" {text} "
    for term in _MONTH_RANGE_TERMS:
        if f" {term} " in padded or f" {term}," in padded or f" {term}." in padded or f" {term}?" in padded:
            return "month"
    current_year = (now or datetime.now(UTC)).year
    for year_text in _YEAR_RE.findall(text):
        if int(year_text) >= current_year - 1:
            return "year"
    return None


def normalize_time_range(value: str | None) -> str | None:
    """Validate a SearXNG-style time range value."""
    normalized = (value or "").strip().lower()
    return normalized if normalized in _VALID_TIME_RANGES else None


def freshness_score(published_at: str | None, *, now: datetime | None = None) -> float:
    """Exponential-decay freshness in [0, 1]. Unknown dates are neutral (0.5)."""
    date = _coerce_iso_date(published_at)
    if not date:
        return UNKNOWN_DATE_FRESHNESS
    try:
        published = datetime.fromisoformat(date).replace(tzinfo=UTC)
    except ValueError:
        return UNKNOWN_DATE_FRESHNESS
    reference = now or datetime.now(UTC)
    age_days = (reference - published).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return math.exp(-math.log(2.0) * age_days / FRESHNESS_HALF_LIFE_DAYS)


def content_metrics(text: str) -> tuple[int, float]:
    """Return (word_count, link_density) for extracted page text."""
    words = (text or "").split()
    word_count = len(words)
    if not word_count:
        return 0, 0.0
    link_count = len(_LINK_RE.findall(text))
    return word_count, link_count / word_count


def classify_extraction(extracted: str | None, snippet: str | None) -> tuple[str, str]:
    """Pick the best available content and label its quality.

    Explicit fallback chain: full extraction -> partial extraction -> search
    snippet -> metadata only. Returns ``(content, extraction_quality)`` where
    quality is one of ``full``, ``partial``, ``snippet_only``, ``metadata_only``.
    """
    extracted_text = (extracted or "").strip()
    snippet_text = (snippet or "").strip()

    if extracted_text:
        word_count, link_density = content_metrics(extracted_text)
        if word_count >= FULL_MIN_WORDS and link_density <= MAX_SANE_LINK_DENSITY:
            return extracted_text, "full"
        if word_count >= PARTIAL_MIN_WORDS:
            return extracted_text, "partial"
        # Extraction produced something too thin to trust; prefer the snippet
        # when it carries at least as much signal.
        if snippet_text and len(snippet_text) >= len(extracted_text):
            return snippet_text, "snippet_only"
        return extracted_text, "partial"

    if snippet_text:
        return snippet_text, "snippet_only"

    return "", "metadata_only"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Universal source authority classification for research citations.

The classifier is intentionally deterministic and conservative. It labels URLs
before the model sees them so downstream research prompts and quality gates can
distinguish official documentation, primary data issuers, academic evidence,
trade press, content marketing, forums, and unknown sources.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field


class SourceClass(StrEnum):
    """Source classes in roughly descending authority for factual claims."""

    FIRST_PARTY = "first_party"
    PRIMARY_ISSUER = "primary_issuer"
    ACADEMIC = "academic"
    AUTHORITATIVE_THIRD_PARTY = "authoritative_third_party"
    TRADE_PRESS = "trade_press"
    VENDOR_MARKETING = "vendor_marketing"
    CONTENT_MARKETING = "content_marketing"
    FORUM = "forum"
    UNKNOWN = "unknown"


LEGACY_SOURCE_CLASS_ALIASES: dict[str, SourceClass] = {
    "third_party_authoritative": SourceClass.AUTHORITATIVE_THIRD_PARTY,
    "blog": SourceClass.CONTENT_MARKETING,
    "government": SourceClass.PRIMARY_ISSUER,
    "practitioner": SourceClass.FORUM,
}

AUTHORITATIVE_CLASSES = {
    SourceClass.FIRST_PARTY.value,
    SourceClass.PRIMARY_ISSUER.value,
    SourceClass.ACADEMIC.value,
    SourceClass.AUTHORITATIVE_THIRD_PARTY.value,
    # Backward-compatible class name still present in older saved registries.
    "third_party_authoritative",
}

WEAK_DERIVATIVE_CLASSES = {
    SourceClass.CONTENT_MARKETING.value,
    SourceClass.VENDOR_MARKETING.value,
    # Backward-compatible class name still present in older saved registries.
    "blog",
}

_REGISTRY_PATH = Path(__file__).with_name("data") / "source_domains.json"
_DOCS_SUBDOMAIN_RE = re.compile(r"^(docs?|developers?|api|help|support)\.", re.IGNORECASE)
_FORUM_HOST_RE = re.compile(
    r"(^|\.)(reddit\.com|stackoverflow\.com|stackexchange\.com|news\.ycombinator\.com)$",
    re.IGNORECASE,
)
_BLOG_HOST_RE = re.compile(r"(^|\.)((medium|substack|hashnode|dev)\.com|blogspot\.com)$", re.IGNORECASE)


class SourceClassification(BaseModel):
    """Deterministic classification metadata for one URL."""

    url: str
    normalized_domain: str
    source_class: SourceClass
    confidence: float = Field(ge=0.0, le=1.0)
    owner: str | None = None
    classification_reason: str


def classify_url(url: str, entity_hint: str | None = None) -> SourceClassification:
    """Return a detailed source classification for ``url``.

    ``entity_hint`` is reserved for claim-aware reclassification. Phase 1 keeps
    the decision URL-centric so the classification can run everywhere sources
    are captured without needing model context.
    """
    normalized_domain, path = _normalize_url_parts(url)
    if not normalized_domain:
        return SourceClassification(
            url=url,
            normalized_domain="",
            source_class=SourceClass.UNKNOWN,
            confidence=0.0,
            classification_reason="URL has no parseable host",
        )

    registry = _load_registry()
    domains = registry.get("domains", {})

    exact = domains.get(normalized_domain)
    if isinstance(exact, dict):
        return _classification_from_registry(url, normalized_domain, exact, "exact domain registry match")

    suffix = _suffix_registry_match(normalized_domain, domains)
    if suffix is not None:
        matched_domain, entry = suffix
        source_class = _coerce_source_class(entry.get("class"))
        if matched_domain != normalized_domain and source_class == SourceClass.FIRST_PARTY:
            return SourceClassification(
                url=url,
                normalized_domain=normalized_domain,
                source_class=SourceClass.FIRST_PARTY,
                confidence=float(entry.get("confidence", 0.85)),
                owner=entry.get("owner"),
                classification_reason=f"subdomain inherited first-party classification from {matched_domain}",
            )
        return _classification_from_registry(url, normalized_domain, entry, f"suffix registry match: {matched_domain}")

    if _FORUM_HOST_RE.search(normalized_domain) or _path_matches(path, registry.get("forum_patterns", [])):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.FORUM,
            confidence=0.9,
            classification_reason="forum/community host or path pattern",
        )

    if _BLOG_HOST_RE.search(normalized_domain) or _path_matches(path, registry.get("content_marketing_patterns", [])):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.CONTENT_MARKETING,
            confidence=0.7,
            classification_reason="content-marketing/blog host or path pattern",
        )

    if _path_matches(path, registry.get("first_party_url_patterns", [])) or _DOCS_SUBDOMAIN_RE.search(
        normalized_domain
    ):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.FIRST_PARTY,
            confidence=0.6,
            owner=None,
            classification_reason="documentation/help/API-shaped URL",
        )

    if _path_matches(path, registry.get("vendor_marketing_patterns", [])):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.VENDOR_MARKETING,
            confidence=0.65,
            owner=None,
            classification_reason="vendor-marketing URL path pattern",
        )

    if _is_government_domain(normalized_domain):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.PRIMARY_ISSUER,
            confidence=0.85,
            classification_reason="government domain suffix",
        )

    if normalized_domain.endswith(".edu"):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.ACADEMIC,
            confidence=0.7,
            classification_reason="academic institution domain suffix",
        )

    if entity_hint and _host_matches_entity(normalized_domain, entity_hint):
        return SourceClassification(
            url=url,
            normalized_domain=normalized_domain,
            source_class=SourceClass.FIRST_PARTY,
            confidence=0.55,
            owner=entity_hint,
            classification_reason="host loosely matches entity hint",
        )

    return SourceClassification(
        url=url,
        normalized_domain=normalized_domain,
        source_class=SourceClass.UNKNOWN,
        confidence=0.3,
        classification_reason="no registry, suffix, or path rule matched",
    )


def classify_source(url: str | None, *, entity_name: str | None = None) -> str:
    """Backward-compatible source-class string helper used by existing code."""
    if not url:
        return SourceClass.UNKNOWN.value
    return classify_url(url, entity_hint=entity_name).source_class.value


def normalize_source_class(source_class: str | SourceClass | None) -> str:
    """Normalize legacy and enum source-class values to the current taxonomy."""
    if source_class is None:
        return SourceClass.UNKNOWN.value
    raw = str(source_class.value if isinstance(source_class, SourceClass) else source_class)
    return LEGACY_SOURCE_CLASS_ALIASES.get(raw, _coerce_source_class(raw)).value


def source_class_rank(source_class: str | SourceClass | None) -> int:
    """Return a descending quality rank for value-aware pruning and sorting."""
    normalized = normalize_source_class(source_class)
    return {
        SourceClass.FIRST_PARTY.value: 8,
        SourceClass.PRIMARY_ISSUER.value: 8,
        SourceClass.ACADEMIC.value: 7,
        SourceClass.AUTHORITATIVE_THIRD_PARTY.value: 6,
        SourceClass.TRADE_PRESS.value: 5,
        SourceClass.VENDOR_MARKETING.value: 3,
        SourceClass.CONTENT_MARKETING.value: 2,
        SourceClass.FORUM.value: 2,
        SourceClass.UNKNOWN.value: 0,
    }.get(normalized, 0)


def source_class_counts(entries: list[Any]) -> dict[str, int]:
    """Count source classes on arbitrary dict/dataclass entries."""
    counts = {source_class.value: 0 for source_class in SourceClass}
    for entry in entries:
        value = getattr(entry, "source_class", None)
        if value is None and isinstance(entry, dict):
            value = entry.get("source_class")
        counts[normalize_source_class(value)] += 1
    return counts


def reload_registry() -> dict[str, Any]:
    """Clear and reload the source domain registry from disk."""
    _load_registry.cache_clear()
    return _load_registry()


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return {
            "domains": {},
            "first_party_url_patterns": [],
            "vendor_marketing_patterns": [],
            "content_marketing_patterns": [],
            "forum_patterns": [],
        }
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _classification_from_registry(
    url: str,
    normalized_domain: str,
    entry: dict[str, Any],
    reason: str,
) -> SourceClassification:
    return SourceClassification(
        url=url,
        normalized_domain=normalized_domain,
        source_class=_coerce_source_class(entry.get("class")),
        confidence=float(entry.get("confidence", 1.0)),
        owner=entry.get("owner"),
        classification_reason=reason,
    )


def _coerce_source_class(value: Any) -> SourceClass:
    raw = str(value or SourceClass.UNKNOWN.value)
    if raw in LEGACY_SOURCE_CLASS_ALIASES:
        return LEGACY_SOURCE_CLASS_ALIASES[raw]
    try:
        return SourceClass(raw)
    except ValueError:
        return SourceClass.UNKNOWN


def _normalize_url_parts(url: str) -> tuple[str, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return "", "/"
    host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0].removeprefix("www.")
    return host, parsed.path.rstrip("/") or "/"


def _suffix_registry_match(host: str, domains: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    parts = host.split(".")
    for idx in range(1, len(parts)):
        candidate = ".".join(parts[idx:])
        entry = domains.get(candidate)
        if isinstance(entry, dict):
            return candidate, entry
    return None


def _path_matches(path: str, patterns: list[str]) -> bool:
    lowered = path.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _is_government_domain(host: str) -> bool:
    return host.endswith(".gov") or ".gov." in host or host.endswith(".gov.uk")


def _host_matches_entity(host: str, entity_name: str) -> bool:
    entity_tokens = [token for token in re.split(r"[^a-z0-9]+", entity_name.lower()) if len(token) >= 3]
    if not entity_tokens:
        return False
    host_root = host.split(".")[-2] if "." in host else host
    return any(token in host_root or host_root in token for token in entity_tokens)

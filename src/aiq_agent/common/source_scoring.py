# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic source scoring for research audit artifacts."""

from __future__ import annotations

import re
from datetime import UTC
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field

from .source_classification import SourceClass
from .source_classification import classify_url
from .source_classification import normalize_source_class
from .source_classification import source_class_rank

_WORD_RE = re.compile(r"[a-z][a-z0-9-]+", re.IGNORECASE)
_RELEVANCE_STOPWORDS = {
    "about",
    "academic",
    "analysis",
    "answer",
    "application",
    "brief",
    "build",
    "case",
    "cases",
    "complete",
    "comprehensive",
    "content",
    "context",
    "current",
    "data",
    "deep",
    "deeper",
    "deliverable",
    "evidence",
    "example",
    "examples",
    "explain",
    "final",
    "for",
    "from",
    "guide",
    "include",
    "latest",
    "lesson",
    "notes",
    "overview",
    "plan",
    "practical",
    "report",
    "research",
    "section",
    "sections",
    "social",
    "source",
    "sources",
    "strategy",
    "summary",
    "the",
    "this",
    "training",
    "use",
    "with",
}


class SourceScore(BaseModel):
    """Auditable source score used by durable research run artifacts."""

    url: str
    title: str | None = None
    source_class: str = SourceClass.UNKNOWN.value
    authority: int = Field(ge=1, le=5)
    recency: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    bias_risk: int = Field(ge=1, le=5)
    used_for: list[str] = Field(default_factory=list)
    scoring_reason: str = ""


def score_source(
    *,
    url: str,
    title: str | None = None,
    source_class: str | None = None,
    used_for: list[str] | None = None,
    extract_count: int = 0,
    relevance_text: str | None = None,
    evidence_text: str | None = None,
    generated_at: datetime | None = None,
) -> SourceScore:
    """Return a conservative 1-5 quality score for one source.

    The score is deterministic and intentionally simple. It is not a substitute
    for claim-aware evaluation; it makes the run folder auditable and helps the
    final writer distinguish official/primary evidence from weak sources.
    """

    classification = classify_url(url)
    normalized_class = normalize_source_class(source_class or classification.source_class)
    used = [str(item) for item in (used_for or []) if str(item).strip()]
    authority = _authority_score(normalized_class)
    recency = _recency_score(" ".join([url, title or ""]), generated_at=generated_at)
    relevance = _relevance_score(
        used_for=used,
        extract_count=extract_count,
        relevance_text=relevance_text,
        source_text=" ".join(
            part
            for part in (
                title or "",
                urlparse(url).netloc.replace("www.", ""),
                urlparse(url).path.replace("/", " "),
                evidence_text or "",
            )
            if part
        ),
    )
    bias_risk = _bias_risk_score(normalized_class)
    reason = (
        f"class={normalized_class}; authority={authority}; recency={recency}; "
        f"relevance={relevance}; bias_risk={bias_risk}"
    )
    return SourceScore(
        url=url,
        title=title,
        source_class=normalized_class,
        authority=authority,
        recency=recency,
        relevance=relevance,
        bias_risk=bias_risk,
        used_for=used,
        scoring_reason=reason,
    )


def source_score_from_entry(entry: Any, *, generated_at: datetime | None = None) -> SourceScore | None:
    """Score a SourceEntry-like object, returning None for empty entries."""

    url = str(getattr(entry, "url", "") or "")
    if not url:
        return None
    claim_ids = list(getattr(entry, "claim_ids", []) or [])
    extracts = list(getattr(entry, "extracts", []) or [])
    return score_source(
        url=url,
        title=getattr(entry, "title", None),
        source_class=getattr(entry, "source_class", None),
        used_for=[str(item) for item in claim_ids],
        extract_count=len(extracts),
        generated_at=generated_at,
    )


def _authority_score(source_class: str) -> int:
    rank = source_class_rank(source_class)
    if rank >= 8:
        return 5
    if rank >= 6:
        return 4
    if rank >= 5:
        return 3
    if rank >= 2:
        return 2
    return 1


def _recency_score(text: str, *, generated_at: datetime | None = None) -> int:
    now = generated_at or datetime.now(UTC)
    years = [int(match.group(1)) for match in re.finditer(r"\b(20[1-3]\d)\b", text)]
    if not years:
        return 3
    newest = max(years)
    age = max(0, now.year - newest)
    if age <= 1:
        return 5
    if age == 2:
        return 4
    if age <= 4:
        return 3
    if age <= 7:
        return 2
    return 1


def _relevance_score(
    *,
    used_for: list[str],
    extract_count: int,
    relevance_text: str | None = None,
    source_text: str | None = None,
) -> int:
    semantic = _semantic_relevance_score(relevance_text, source_text)
    structural = 2
    if len(used_for) >= 3 or extract_count >= 4:
        structural = 5
    elif len(used_for) >= 2 or extract_count >= 2:
        structural = 4
    elif used_for or extract_count:
        structural = 3

    if semantic is None:
        return structural
    if semantic <= 1 and not used_for and extract_count == 0:
        return 1
    return max(structural, semantic)


def _semantic_relevance_score(relevance_text: str | None, source_text: str | None) -> int | None:
    """Return a conservative topic-match score from query terms to source text."""

    scope_terms = _terms(relevance_text)
    if not scope_terms:
        return None
    source_terms = _terms(source_text)
    if not source_terms:
        return 1
    overlap = scope_terms & source_terms
    if len(overlap) >= 6:
        return 5
    if len(overlap) >= 3:
        return 4
    if len(overlap) >= 1:
        return 3
    return 1


def _terms(text: str | None) -> set[str]:
    if not text:
        return set()
    text = re.sub(r"[-_/]+", " ", text)
    terms: set[str] = set()
    for token in _WORD_RE.findall(text.lower()):
        if len(token) < 4 or token in _RELEVANCE_STOPWORDS:
            continue
        terms.add(token)
        if len(token) > 5 and token.endswith("s"):
            terms.add(token[:-1])
    return terms


def _bias_risk_score(source_class: str) -> int:
    normalized = normalize_source_class(source_class)
    if normalized in {
        SourceClass.FIRST_PARTY.value,
        SourceClass.PRIMARY_ISSUER.value,
        SourceClass.ACADEMIC.value,
    }:
        return 1
    if normalized == SourceClass.AUTHORITATIVE_THIRD_PARTY.value:
        return 2
    if normalized == SourceClass.TRADE_PRESS.value:
        return 3
    if normalized in {SourceClass.VENDOR_MARKETING.value, SourceClass.CONTENT_MARKETING.value}:
        return 4
    return 5 if normalized == SourceClass.UNKNOWN.value else 4

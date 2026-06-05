# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build ranked long-context evidence packets for MiniMax M3 synthesis.

The evidence packet is deliberately deterministic. Researchers can write noisy
notes and small claim/extract fragments, but Python decides what becomes the
canonical dossier that the final long-context writer sees.
"""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field

from .claim_table import ClaimTable
from .claim_table import validate_claim_table_json
from .source_classification import classify_url
from .source_classification import normalize_source_class
from .source_classification import source_class_rank
from .source_scoring import score_source


class EvidenceExtract(BaseModel):
    """A compact source-backed extract used by one or more claims."""

    text: str
    claim_ids: list[str] = Field(default_factory=list)
    extraction_status: str = "extracted"


class EvidencePacketSource(BaseModel):
    """One ranked source entry in the synthesis dossier."""

    url: str
    title: str | None = None
    source_class: str = "unknown"
    extraction_status: str = "unknown"
    claim_ids: list[str] = Field(default_factory=list)
    extracts: list[EvidenceExtract] = Field(default_factory=list)
    full_text: str | None = None
    rank_score: int = 0
    authority: int = 1
    recency: int = 3
    relevance: int = 1
    bias_risk: int = 5
    used_for: list[str] = Field(default_factory=list)


class EvidencePacket(BaseModel):
    """Canonical evidence dossier read by the final M3 synthesis pass."""

    schema_version: str = "1.0"
    job_id: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_count: int = 0
    claim_count: int = 0
    sources: list[EvidencePacketSource] = Field(default_factory=list)
    source_class_distribution: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def build_evidence_packet(
    *,
    job_id: str | None = None,
    request_text: str = "",
    claim_table_content: str | None = None,
    extract_contents: list[str] | None = None,
    registry_sources: list[Any] | None = None,
    max_extract_chars: int = 1600,
    max_full_text_chars: int = 16000,
) -> EvidencePacket:
    """Build a ranked evidence packet from claim table, extracts, and registry.

    Args:
        job_id: Optional async job id.
        request_text: Original user request or approved scope text, used for
            deterministic source relevance scoring.
        claim_table_content: JSON content from `/shared/claim_table.json`.
        extract_contents: JSON fragments from `/shared/extracts/*.json`.
        registry_sources: SourceEntry-like objects captured from search tools.
        max_extract_chars: Per-extract character cap.
        max_full_text_chars: Per-source full-text cap for high-authority source
            payloads. Full text is never required; extracts are preferred.
    """
    sources: dict[str, EvidencePacketSource] = {}
    notes: list[str] = []
    claim_count = 0

    if claim_table_content:
        table, errors = validate_claim_table_json(claim_table_content)
        if table is None:
            notes.append("claim_table_invalid: " + "; ".join(errors))
        else:
            claim_count = len(table.all_claims())
            _add_claim_table_evidence(sources, table, max_extract_chars=max_extract_chars)

    for raw in extract_contents or []:
        _add_extract_fragment(
            sources, raw, notes, max_extract_chars=max_extract_chars, max_full_text_chars=max_full_text_chars
        )

    for source in registry_sources or []:
        url = str(getattr(source, "url", "") or "")
        if not url:
            continue
        entry = _get_or_create_source(sources, url)
        title = getattr(source, "title", None)
        source_class = getattr(source, "source_class", None)
        if title and not entry.title:
            entry.title = str(title)
        if source_class:
            entry.source_class = normalize_source_class(source_class)
        if entry.extraction_status == "unknown":
            entry.extraction_status = "registered"

    _score_sources(sources.values(), request_text=request_text)
    ranked = sorted(
        (source for source in sources.values() if _keep_source(source)),
        key=_source_sort_key,
        reverse=True,
    )
    distribution: dict[str, int] = {}
    for source in ranked:
        distribution[source.source_class] = distribution.get(source.source_class, 0) + 1

    return EvidencePacket(
        job_id=job_id,
        source_count=len(ranked),
        claim_count=claim_count,
        sources=ranked,
        source_class_distribution=dict(sorted(distribution.items())),
        notes=notes,
    )


def _add_claim_table_evidence(
    sources: dict[str, EvidencePacketSource],
    table: ClaimTable,
    *,
    max_extract_chars: int,
) -> None:
    for claim in table.all_claims():
        claim_id = getattr(claim, "claim_id", "")
        evidence_items = getattr(claim, "evidence", []) or []
        for evidence in evidence_items:
            url = str(getattr(evidence, "source_url", "") or "")
            extract = str(getattr(evidence, "extract", "") or "")
            if not url or not extract:
                continue
            entry = _get_or_create_source(sources, url)
            source_class = getattr(evidence, "source_class", None)
            if source_class:
                entry.source_class = normalize_source_class(source_class)
            if claim_id and claim_id not in entry.claim_ids:
                entry.claim_ids.append(claim_id)
            _append_extract(entry, extract, [claim_id] if claim_id else [], max_extract_chars=max_extract_chars)

        # Backward-compatible ClaimResolution shape.
        url = str(getattr(claim, "source_url", "") or "")
        extract = str(getattr(claim, "source_extract", "") or "")
        if url and extract:
            entry = _get_or_create_source(sources, url)
            source_class = getattr(claim, "source_class", None)
            if source_class:
                entry.source_class = normalize_source_class(source_class)
            if claim_id and claim_id not in entry.claim_ids:
                entry.claim_ids.append(claim_id)
            _append_extract(entry, extract, [claim_id] if claim_id else [], max_extract_chars=max_extract_chars)


def _add_extract_fragment(
    sources: dict[str, EvidencePacketSource],
    raw: str,
    notes: list[str],
    *,
    max_extract_chars: int,
    max_full_text_chars: int,
) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        notes.append(f"extract_fragment_invalid_json: {exc.msg}")
        return

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("extracts") or payload.get("sources") or payload.get("entries") or [payload]
    else:
        notes.append("extract_fragment_unsupported_shape")
        return

    for item in entries:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("source_url") or "")
        if not url:
            continue
        entry = _get_or_create_source(sources, url)
        if item.get("title"):
            entry.title = str(item["title"])
        source_class = item.get("source_class")
        if source_class:
            entry.source_class = normalize_source_class(source_class)
        entry.extraction_status = str(item.get("extraction_status") or item.get("status") or entry.extraction_status)
        claim_ids = _coerce_claim_ids(item.get("claim_ids") or item.get("claim_id"))
        for claim_id in claim_ids:
            if claim_id not in entry.claim_ids:
                entry.claim_ids.append(claim_id)
        extract = str(item.get("extract") or item.get("source_extract") or item.get("text") or "")
        if extract:
            _append_extract(entry, extract, claim_ids, max_extract_chars=max_extract_chars)
        full_text = str(item.get("full_text") or item.get("content") or "")
        if full_text and source_class_rank(entry.source_class) >= source_class_rank("authoritative_third_party"):
            entry.full_text = _truncate(full_text, max_full_text_chars)


def _get_or_create_source(sources: dict[str, EvidencePacketSource], url: str) -> EvidencePacketSource:
    normalized = _normalize_url(url)
    existing = sources.get(normalized)
    if existing is not None:
        return existing
    classification = classify_url(url)
    title = urlparse(url).netloc.replace("www.", "") or url
    entry = EvidencePacketSource(
        url=url,
        title=title,
        source_class=classification.source_class.value,
    )
    sources[normalized] = entry
    return entry


def _append_extract(
    entry: EvidencePacketSource,
    text: str,
    claim_ids: list[str],
    *,
    max_extract_chars: int,
) -> None:
    cleaned = _truncate(" ".join(str(text).split()), max_extract_chars)
    if not cleaned:
        return
    for existing in entry.extracts:
        if existing.text == cleaned:
            for claim_id in claim_ids:
                if claim_id and claim_id not in existing.claim_ids:
                    existing.claim_ids.append(claim_id)
            return
    entry.extracts.append(EvidenceExtract(text=cleaned, claim_ids=[claim_id for claim_id in claim_ids if claim_id]))


def _coerce_claim_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _normalize_url(url: str) -> str:
    return str(url).strip().rstrip("/")


def _truncate(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + f" [... truncated from {len(value)} chars]"


def _score_sources(sources: Any, *, request_text: str) -> None:
    for source in sources:
        evidence_text = " ".join(
            [extract.text for extract in source.extracts[:5]] + ([source.full_text] if source.full_text else [])
        )
        score = score_source(
            url=source.url,
            title=source.title,
            source_class=source.source_class,
            used_for=source.claim_ids,
            extract_count=len(source.extracts),
            relevance_text=request_text,
            evidence_text=evidence_text,
        )
        source.authority = score.authority
        source.recency = score.recency
        source.relevance = score.relevance
        source.bias_risk = score.bias_risk
        source.used_for = score.used_for
        source.rank_score = _source_rank_score(source)


def _keep_source(source: EvidencePacketSource) -> bool:
    """Drop only registry-only sources that are unsupported and off-topic."""

    if source.claim_ids or source.extracts or source.full_text:
        return True
    return source.relevance > 1


def _source_rank_score(source: EvidencePacketSource) -> int:
    return (
        source_class_rank(source.source_class) * 100
        + min(len(source.claim_ids), 20) * 5
        + min(len(source.extracts), 10)
        + source.relevance * 3
        - source.bias_risk
    )


def _source_sort_key(source: EvidencePacketSource) -> tuple[int, int, int, int, int, str]:
    return (
        source_class_rank(source.source_class),
        len(source.claim_ids),
        len(source.extracts),
        source.relevance,
        -source.bias_risk,
        source.url,
    )

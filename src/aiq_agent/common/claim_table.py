# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Universal claim-resolution schema for deep research."""

from __future__ import annotations

import json
from datetime import UTC
from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

from .source_classification import SourceClass
from .source_classification import normalize_source_class
from .source_classification import source_class_rank

ClaimType = Literal[
    "specification",
    "quantitative",
    "causal",
    "comparative",
    "existence",
    "recommendation",
    "trend",
    "definitional",
    "definition",
    "discovery",
    "other",
]
ClaimStatus = Literal["verified", "partially_verified", "unverified"]
SourceRequirement = Literal[
    "first_party",
    "primary_issuer",
    "academic",
    "government",
    "third_party_authoritative",
    "practitioner",
    "mixed",
    "any_credible",
    "unknown",
]
ExpectedAnswerShape = Literal["yes_no", "number", "short_string", "list", "free_text"]
ClaimDensity = Literal["low", "medium", "high"]
Verifiability = Literal["mostly_objective", "mixed", "mostly_subjective"]


class ClaimEvidence(BaseModel):
    """One source extract that supports or partially supports an atomic claim."""

    model_config = ConfigDict(extra="allow")

    source_url: str
    source_class: str = "unknown"
    extract: str = Field(min_length=1)
    extract_confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str) -> str:
        if value.startswith(("http://", "https://")):
            return value
        raise ValueError("source_url must be http(s)")

    @field_validator("source_class", mode="before")
    @classmethod
    def _normalize_source_class(cls, value: Any) -> str:
        return normalize_source_class(value)


class AtomicClaim(BaseModel):
    """One small, verifiable claim anticipated or resolved during research."""

    model_config = ConfigDict(extra="allow")

    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    claim_type: ClaimType = "other"
    expected_answer_shape: ExpectedAnswerShape = "free_text"
    preferred_source_classes: list[str] = Field(default_factory=lambda: ["any_credible"])
    status: ClaimStatus = "unverified"
    resolved_value: str | None = None
    evidence: list[ClaimEvidence] = Field(default_factory=list)
    hedge_required: bool = False
    hedge_phrase: str | None = None
    researcher_notes: str | None = None
    attempted_sources: list[str] = Field(default_factory=list)
    resolved_at: datetime | None = None
    resolved_by_task: str | None = None

    @field_validator("preferred_source_classes", mode="before")
    @classmethod
    def _normalize_preferred_classes(cls, value: Any) -> list[str]:
        if value is None:
            return ["any_credible"]
        values = value if isinstance(value, list) else [value]
        return [normalize_source_class(item) if item != "any_credible" else "any_credible" for item in values]

    def model_post_init(self, __context: Any) -> None:
        if self.status == "verified" and not self.evidence:
            raise ValueError("verified atomic claim requires evidence")
        if self.status == "partially_verified":
            if not self.evidence:
                raise ValueError("partially verified atomic claim requires evidence")
            if not (self.hedge_required and self.hedge_phrase):
                raise ValueError("partially verified atomic claim requires hedge_required and hedge_phrase")
        # Planner-created claim profiles start as unverified targets without
        # attempted sources. Researcher-created unresolved claims should add
        # notes/attempts, but that is a quality signal rather than a schema
        # hard failure.


class ClaimProfile(BaseModel):
    """Planner output: what kinds of atomic claims an ideal answer needs."""

    model_config = ConfigDict(extra="allow")

    claim_density: ClaimDensity = "medium"
    verifiability: Verifiability = "mixed"
    claims: list[AtomicClaim] = Field(default_factory=list)


class ClaimResolution(BaseModel):
    """One anticipated claim resolved against gathered evidence."""

    model_config = ConfigDict(extra="allow")

    claim_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    claim_type: ClaimType = "other"
    status: ClaimStatus
    value: str | None = None
    source_url: str | None = None
    source_class: SourceClass | None = None
    required_source_class: SourceRequirement = "unknown"
    source_extract: str | None = None
    confidence: Literal["high", "medium", "low", "unverified"] = "medium"
    downgrade_reason: str | None = None
    attempted_sources: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str | None) -> str | None:
        if value is None or value.startswith(("http://", "https://")):
            return value
        raise ValueError("source_url must be http(s) when provided")

    @field_validator("source_extract")
    @classmethod
    def _normalize_extract(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("source_class", mode="before")
    @classmethod
    def _normalize_source_class(cls, value: Any) -> str | None:
        if value is None:
            return None
        return normalize_source_class(value)

    def model_post_init(self, __context: Any) -> None:
        if self.status == "verified":
            missing = []
            if not self.source_url:
                missing.append("source_url")
            if not self.source_extract:
                missing.append("source_extract")
            if not self.source_class:
                missing.append("source_class")
            if missing:
                raise ValueError(f"verified claim is missing required fields: {', '.join(missing)}")
        elif self.status == "partially_verified":
            missing = []
            if not self.source_url:
                missing.append("source_url")
            if not self.source_extract:
                missing.append("source_extract")
            if not self.downgrade_reason:
                missing.append("downgrade_reason")
            if missing:
                raise ValueError(f"partially verified claim is missing required fields: {', '.join(missing)}")
        elif not (self.downgrade_reason or self.notes or self.attempted_sources):
            raise ValueError("unverified claim must include downgrade_reason, notes, or attempted_sources")


class ClaimTable(BaseModel):
    """Container accepted for /shared/claim_table.json and claim fragments."""

    model_config = ConfigDict(extra="allow")

    job_id: str | None = None
    profile: ClaimProfile | None = None
    entries: list[ClaimResolution] = Field(default_factory=list)
    atomic_claims: list[AtomicClaim] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    schema_version: Literal["1.0"] = "1.0"

    @classmethod
    def from_any(cls, payload: Any) -> ClaimTable:
        if isinstance(payload, list):
            return cls(entries=payload)
        if isinstance(payload, dict):
            if "atomic_claims" in payload or "profile" in payload:
                return cls.model_validate(payload)
            if "entries" in payload:
                return cls.model_validate(payload)
            if "claims" in payload:
                claims = payload["claims"]
                if claims and isinstance(claims[0], dict) and "claim_text" in claims[0]:
                    return cls(atomic_claims=claims)
                return cls(entries=claims)
        raise ValueError("claim table must be a list or an object with entries/claims")

    def verified_count(self) -> int:
        return sum(1 for claim in self.all_claims() if claim.status == "verified")

    def partially_verified_count(self) -> int:
        return sum(1 for claim in self.all_claims() if claim.status == "partially_verified")

    def unverified_count(self) -> int:
        return sum(1 for claim in self.all_claims() if claim.status == "unverified")

    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in self.all_claims():
            counts[claim.claim_type] = counts.get(claim.claim_type, 0) + 1
        return counts

    def all_claims(self) -> list[ClaimResolution | AtomicClaim]:
        return [*self.entries, *self.atomic_claims]


def validate_claim_table_json(content: str) -> tuple[ClaimTable | None, list[str]]:
    """Validate claim-table JSON content and return human-readable errors."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc.msg}"]
    try:
        return ClaimTable.from_any(payload), []
    except (ValidationError, ValueError) as exc:
        return None, [str(exc)]


def merge_claim_tables_json(contents: list[str]) -> tuple[ClaimTable | None, list[str]]:
    """Merge claim-table fragments, deduping by claim_id when available."""
    entries: dict[str, ClaimResolution] = {}
    atomic_claims: dict[str, AtomicClaim] = {}
    errors: list[str] = []
    synthetic_index = 0
    for content in contents:
        table, table_errors = validate_claim_table_json(content)
        if table is None:
            errors.extend(table_errors)
            continue
        for entry in table.entries:
            key = entry.claim_id or f"claim-{synthetic_index}"
            synthetic_index += 1
            existing = entries.get(key)
            if existing is None or _claim_resolution_sort_key(entry) > _claim_resolution_sort_key(existing):
                entries[key] = entry
        for claim in table.atomic_claims:
            key = claim.claim_id or f"atomic-{synthetic_index}"
            synthetic_index += 1
            existing_claim = atomic_claims.get(key)
            if existing_claim is None or _atomic_claim_sort_key(claim) > _atomic_claim_sort_key(existing_claim):
                atomic_claims[key] = claim
    if errors and not entries and not atomic_claims:
        return None, errors
    return ClaimTable(entries=list(entries.values()), atomic_claims=list(atomic_claims.values())), errors


def summarize_claim_table(content: str) -> dict[str, Any]:
    """Return compact metrics for quality gates and logs."""
    table, errors = validate_claim_table_json(content)
    if table is None:
        return {"valid": False, "errors": errors, "total": 0}

    by_status = {"verified": 0, "partially_verified": 0, "unverified": 0}
    by_type: dict[str, int] = {}
    source_match_count = 0
    sourced_count = 0
    for entry in table.entries:
        by_status[entry.status] += 1
        by_type[entry.claim_type] = by_type.get(entry.claim_type, 0) + 1
        if entry.source_url:
            sourced_count += 1
        if entry.status == "verified" and _source_matches_requirement(entry.source_class, entry.required_source_class):
            source_match_count += 1
    for claim in table.atomic_claims:
        by_status[claim.status] += 1
        by_type[claim.claim_type] = by_type.get(claim.claim_type, 0) + 1
        if claim.evidence:
            sourced_count += 1
        if claim.status == "verified" and claim.evidence:
            preferred = claim.preferred_source_classes or ["any_credible"]
            best_evidence = max(claim.evidence, key=lambda evidence: source_class_rank(evidence.source_class))
            if any(
                requirement == "any_credible"
                or normalize_source_class(best_evidence.source_class) == normalize_source_class(requirement)
                for requirement in preferred
            ):
                source_match_count += 1

    return {
        "valid": True,
        "errors": [],
        "total": len(table.entries) + len(table.atomic_claims),
        "by_status": by_status,
        "by_type": by_type,
        "sourced_count": sourced_count,
        "source_match_count": source_match_count,
        "atomic_claim_count": len(table.atomic_claims),
    }


def _status_rank(status: ClaimStatus) -> int:
    return {"unverified": 0, "partially_verified": 1, "verified": 2}[status]


def _claim_resolution_sort_key(entry: ClaimResolution) -> tuple[int, int, int]:
    return (
        _status_rank(entry.status),
        source_class_rank(entry.source_class),
        1 if entry.source_extract else 0,
    )


def _atomic_claim_sort_key(claim: AtomicClaim) -> tuple[int, int, int]:
    best_rank = max((source_class_rank(evidence.source_class) for evidence in claim.evidence), default=0)
    return (_status_rank(claim.status), best_rank, len(claim.evidence))


def _source_matches_requirement(source_class: SourceClass | None, requirement: SourceRequirement) -> bool:
    if requirement in {"unknown", "any_credible", "mixed"}:
        return source_class is not None
    if requirement in {"primary_issuer", "government", "academic"}:
        return normalize_source_class(source_class) in {"first_party", "primary_issuer", "academic"}
    if requirement == "practitioner":
        return normalize_source_class(source_class) in {
            "authoritative_third_party",
            "trade_press",
            "forum",
            "content_marketing",
        }
    return normalize_source_class(source_class) == normalize_source_class(requirement)

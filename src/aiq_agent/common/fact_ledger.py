# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema and validation helpers for deep-research fact ledgers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
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


class FactLedgerEntry(BaseModel):
    """One verified or explicitly unverified fact about a named entity."""

    model_config = ConfigDict(extra="allow")

    entity: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    fact_type: str = "general"
    value: str | None = None
    unit: str | None = None
    event_date: str | None = None
    source_published_at: str | None = None
    as_of_date: str | None = None
    supersedes: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_extract: str | None = None
    source_class: SourceClass | None = None
    confidence: Literal["high", "medium", "low", "unverified"] = "medium"
    status: Literal["verified", "unverified"] = "verified"
    reason: str | None = None
    attempted_sources: list[str] = Field(default_factory=list)

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str | None) -> str | None:
        if value is None or value.startswith(("http://", "https://")):
            return value
        raise ValueError("source_url must be http(s) when provided")

    @field_validator("source_extract")
    @classmethod
    def _validate_extract(cls, value: str | None) -> str | None:
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
                raise ValueError(f"verified fact is missing required fields: {', '.join(missing)}")
        elif not self.reason:
            raise ValueError("unverified fact must include reason")


class FactLedger(BaseModel):
    """Container accepted for /shared/fact_ledger.json."""

    model_config = ConfigDict(extra="allow")

    entries: list[FactLedgerEntry] = Field(default_factory=list)

    @classmethod
    def from_any(cls, payload: Any) -> FactLedger:
        if isinstance(payload, list):
            return cls(entries=payload)
        if isinstance(payload, dict):
            if "entries" in payload:
                return cls.model_validate(payload)
            if all(isinstance(value, list) for value in payload.values()):
                entries = []
                for entity, values in payload.items():
                    for value in values:
                        if isinstance(value, dict):
                            entries.append({"entity": entity, **value})
                return cls(entries=entries)
        raise ValueError("fact ledger must be a list, an object with entries, or an entity-to-entries map")


@dataclass(frozen=True)
class LiveFactConflict:
    """A stale or conflicting current/live fact in the ledger."""

    entity: str
    fact_type: str
    stale_fact: str
    latest_fact: str
    stale_date: str
    latest_date: str
    message: str


LIVE_FACT_TYPES = frozenset(
    {
        "current_funding",
        "funding",
        "valuation",
        "pricing",
        "release_status",
        "model_limit",
        "benchmark",
        "current_metric",
    }
)


def validate_fact_ledger_json(content: str) -> tuple[FactLedger | None, list[str]]:
    """Validate fact-ledger JSON content and return human-readable errors."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc.msg}"]
    try:
        return FactLedger.from_any(payload), []
    except (ValidationError, ValueError) as exc:
        return None, [str(exc)]


def merge_fact_ledgers_json(contents: list[str]) -> tuple[FactLedger | None, list[str]]:
    """Merge fact-ledger fragments, deduping by entity/fact text."""
    entries: dict[str, FactLedgerEntry] = {}
    errors: list[str] = []
    synthetic_index = 0
    for content in contents:
        ledger, ledger_errors = validate_fact_ledger_json(content)
        if ledger is None:
            errors.extend(ledger_errors)
            continue
        for entry in ledger.entries:
            key = f"{entry.entity.strip().lower()}::{entry.fact.strip().lower()}" or f"fact-{synthetic_index}"
            synthetic_index += 1
            existing = entries.get(key)
            if existing is None or _fact_entry_sort_key(entry) > _fact_entry_sort_key(existing):
                entries[key] = entry
    if errors and not entries:
        return None, errors
    return FactLedger(entries=list(entries.values())), errors


def summarize_fact_ledger(content: str) -> dict[str, Any]:
    """Return compact metrics for quality gates and logs."""
    ledger, errors = validate_fact_ledger_json(content)
    if ledger is None:
        return {"valid": False, "errors": errors, "entities": {}, "verified_count": 0, "unverified_count": 0}

    entities: dict[str, dict[str, int]] = {}
    verified_count = 0
    unverified_count = 0
    for entry in ledger.entries:
        entity = entities.setdefault(entry.entity, {"verified": 0, "unverified": 0})
        if entry.status == "verified":
            entity["verified"] += 1
            verified_count += 1
        else:
            entity["unverified"] += 1
            unverified_count += 1
    return {
        "valid": True,
        "errors": [],
        "entities": entities,
        "verified_count": verified_count,
        "unverified_count": unverified_count,
        "live_fact_conflicts": [conflict.__dict__ for conflict in live_fact_conflicts(ledger)],
    }


def live_fact_conflicts(ledger: FactLedger) -> list[LiveFactConflict]:
    """Return stale current/live facts that have a newer conflicting value.

    This deliberately focuses on facts where recency changes the answer:
    funding, valuation, pricing, release status, model limits, and benchmarks.
    Historical facts can coexist; current/live facts need a newest-value rule.
    """

    groups: dict[tuple[str, str], list[FactLedgerEntry]] = {}
    for entry in ledger.entries:
        fact_type = _normalize_fact_type(entry.fact_type)
        if entry.status != "verified" or fact_type not in LIVE_FACT_TYPES:
            continue
        groups.setdefault((_normalize_entity(entry.entity), fact_type), []).append(entry)

    conflicts: list[LiveFactConflict] = []
    for (_entity_key, fact_type), entries in groups.items():
        dated = [(entry, _entry_date(entry)) for entry in entries]
        dated = [(entry, parsed_date) for entry, parsed_date in dated if parsed_date is not None]
        if len(dated) < 2:
            continue
        latest_entry, latest_date = max(dated, key=lambda item: item[1])
        latest_value = _entry_value(latest_entry)
        for entry, parsed_date in dated:
            if entry is latest_entry:
                continue
            if parsed_date >= latest_date:
                continue
            if _entry_value(entry) == latest_value:
                continue
            conflicts.append(
                LiveFactConflict(
                    entity=latest_entry.entity,
                    fact_type=fact_type,
                    stale_fact=entry.fact,
                    latest_fact=latest_entry.fact,
                    stale_date=entry.event_date or entry.source_published_at or entry.as_of_date or "",
                    latest_date=latest_entry.event_date
                    or latest_entry.source_published_at
                    or latest_entry.as_of_date
                    or "",
                    message=(
                        f"{entry.entity} {fact_type} has newer verified evidence dated "
                        f"{latest_date.isoformat()}; older value dated {parsed_date.isoformat()} "
                        "must be framed as historical or omitted."
                    ),
                )
            )
    return conflicts


def _fact_entry_sort_key(entry: FactLedgerEntry) -> tuple[int, int, int, int]:
    source_rank = 0
    if entry.source_class:
        source_rank = {
            "unknown": 0,
            "forum": 1,
            "content_marketing": 2,
            "vendor_marketing": 2,
            "trade_press": 3,
            "authoritative_third_party": 4,
            "academic": 5,
            "primary_issuer": 6,
            "first_party": 7,
        }.get(normalize_source_class(entry.source_class), 0)
    return (
        1 if entry.status == "verified" else 0,
        source_rank,
        _date_sort_value(_entry_date(entry)),
        1 if entry.source_extract else 0,
    )


def _normalize_entity(entity: str) -> str:
    return " ".join(entity.lower().split())


def _normalize_fact_type(fact_type: str | None) -> str:
    value = " ".join(str(fact_type or "general").lower().replace("-", "_").split())
    value = value.replace(" ", "_")
    aliases = {
        "funding_round": "funding",
        "fundraise": "funding",
        "valuation_current": "valuation",
        "current_valuation": "valuation",
        "price": "pricing",
        "api_pricing": "pricing",
    }
    return aliases.get(value, value)


def _entry_value(entry: FactLedgerEntry) -> str:
    return " ".join(str(entry.value or entry.fact).lower().split())


def _entry_date(entry: FactLedgerEntry) -> date | None:
    for value in (entry.event_date, entry.source_published_at, entry.as_of_date):
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.date()
        except ValueError:
            continue
    return None


def _date_sort_value(value: date | None) -> int:
    if value is None:
        return 0
    return value.toordinal()

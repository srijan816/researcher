# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent cross-run fact ledger backed by SQLite.

On research completion the merged ``/shared/fact_ledger.json`` is persisted
here (verified entries only, upsert keyed on entity + fact_type). On job start
the planner context can be primed with previously verified facts whose entity
appears in the user query, clearly labeled so the model re-verifies anything
staleness-sensitive.

Every database error is a no-op: a persistence/lookup failure must never crash
or block a research job. The whole store can be disabled with
``AIQ_FACT_LEDGER_ENABLED=0``.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from .fact_ledger import validate_fact_ledger_json

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./data/fact_ledger.db"
DEFAULT_INJECTION_LIMIT = 12

_FALSEY = {"0", "false", "no", "off"}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,63}")

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "what", "when", "where",
        "which", "who", "how", "why", "are", "was", "were", "will", "can", "could",
        "should", "would", "about", "into", "over", "under", "between", "their",
        "research", "report", "deep", "analysis", "latest", "current", "best", "top",
        "new", "news", "all", "any", "its", "his", "her", "our", "your", "you",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    entity TEXT NOT NULL COLLATE NOCASE,
    fact_type TEXT NOT NULL COLLATE NOCASE,
    fact TEXT NOT NULL DEFAULT '',
    value TEXT,
    as_of TEXT,
    source_url TEXT,
    job_id TEXT,
    status TEXT NOT NULL DEFAULT 'verified',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (entity, fact_type)
)
"""


def fact_ledger_enabled() -> bool:
    """Kill switch: AIQ_FACT_LEDGER_ENABLED (default on)."""
    return os.getenv("AIQ_FACT_LEDGER_ENABLED", "1").strip().lower() not in _FALSEY


def fact_ledger_db_path() -> str:
    """Database location: AIQ_FACT_LEDGER_DB (default ./data/fact_ledger.db)."""
    return os.getenv("AIQ_FACT_LEDGER_DB", "").strip() or DEFAULT_DB_PATH


def _connect(db_path: str | None = None) -> sqlite3.Connection | None:
    """Open (and initialize) the store. Returns None on any failure."""
    path = db_path or fact_ledger_db_path()
    try:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=4000")
        connection.execute(_SCHEMA)
        connection.row_factory = sqlite3.Row
        return connection
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.debug("Fact ledger store unavailable at %s", path, exc_info=True)
        return None


def _normalize_key(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def upsert_fact(
    *,
    entity: str,
    fact_type: str,
    value: str | None,
    fact: str = "",
    as_of: str | None = None,
    source_url: str | None = None,
    job_id: str | None = None,
    status: str = "verified",
    db_path: str | None = None,
) -> bool:
    """Insert or replace one fact keyed on (entity, fact_type). No-op on error."""
    if not fact_ledger_enabled():
        return False
    entity_key = _normalize_key(entity)
    fact_type_key = _normalize_key(fact_type) or "general"
    if not entity_key:
        return False
    connection = _connect(db_path)
    if connection is None:
        return False
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO facts (entity, fact_type, fact, value, as_of, source_url, job_id, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity, fact_type) DO UPDATE SET
                    fact = excluded.fact,
                    value = excluded.value,
                    as_of = excluded.as_of,
                    source_url = excluded.source_url,
                    job_id = excluded.job_id,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    entity_key,
                    fact_type_key,
                    str(fact or "")[:600],
                    None if value is None else str(value)[:400],
                    as_of,
                    source_url,
                    job_id,
                    status or "verified",
                    datetime.now(UTC).isoformat(),
                ),
            )
        return True
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.debug("Fact ledger upsert failed for %s/%s", entity_key, fact_type_key, exc_info=True)
        return False
    finally:
        connection.close()


def persist_fact_ledger_json(content: str, *, job_id: str | None = None, db_path: str | None = None) -> int:
    """Persist verified entries from a merged /shared/fact_ledger.json payload.

    Returns the number of facts persisted. Every failure is a no-op (0).
    """
    if not fact_ledger_enabled() or not str(content or "").strip():
        return 0
    try:
        ledger, errors = validate_fact_ledger_json(content)
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.debug("Fact ledger parse crashed during persistence", exc_info=True)
        return 0
    if ledger is None:
        if errors:
            logger.debug("Fact ledger not persisted: %s", "; ".join(errors))
        return 0

    persisted = 0
    for entry in ledger.entries:
        if entry.status != "verified":
            continue
        as_of = entry.event_date or entry.source_published_at or entry.as_of_date
        ok = upsert_fact(
            entity=entry.entity,
            fact_type=entry.fact_type or "general",
            value=entry.value or entry.fact,
            fact=entry.fact,
            as_of=as_of,
            source_url=entry.source_url,
            job_id=job_id,
            status=entry.status,
            db_path=db_path,
        )
        if ok:
            persisted += 1
    if persisted:
        logger.info("Persisted %d verified fact(s) to the cross-run fact ledger", persisted)
    return persisted


def _query_tokens(query: str) -> set[str]:
    tokens = set(_TOKEN_RE.findall(str(query or "").lower()))
    return {token for token in tokens if len(token) >= 3 and token not in _STOPWORDS}


def lookup_facts_for_query(
    query: str,
    *,
    limit: int = DEFAULT_INJECTION_LIMIT,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return stored facts whose entity tokens appear in the user query.

    Matching is simple and case-insensitive: every token of the stored entity
    must appear in the query token set, or the whole entity string must be a
    substring of the query. Returns [] on any error.
    """
    if not fact_ledger_enabled():
        return []
    tokens = _query_tokens(query)
    query_lower = " ".join(str(query or "").lower().split())
    if not tokens and not query_lower:
        return []
    connection = _connect(db_path)
    if connection is None:
        return []
    try:
        rows = connection.execute(
            "SELECT entity, fact_type, fact, value, as_of, source_url, job_id, status, updated_at "
            "FROM facts ORDER BY updated_at DESC LIMIT 2000"
        ).fetchall()
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.debug("Fact ledger lookup failed", exc_info=True)
        return []
    finally:
        connection.close()

    matches: list[dict[str, Any]] = []
    for row in rows:
        entity = str(row["entity"] or "")
        entity_tokens = _query_tokens(entity) or ({entity.lower()} if entity else set())
        if not entity_tokens:
            continue
        if entity_tokens.issubset(tokens) or (entity.lower() in query_lower):
            matches.append({key: row[key] for key in row.keys()})
            if len(matches) >= max(1, limit):
                break
    return matches


def build_prior_facts_block(facts: list[dict[str, Any]], *, today: str | None = None) -> str:
    """Format previously verified facts as a clearly-labeled planner context block."""
    if not facts:
        return ""
    as_of_label = today or datetime.now(UTC).date().isoformat()
    lines = [
        f"Previously verified facts (as of {as_of_label}; re-verify when staleness matters):",
    ]
    for fact in facts[:DEFAULT_INJECTION_LIMIT]:
        entity = str(fact.get("entity") or "").strip()
        fact_type = str(fact.get("fact_type") or "general").strip()
        value = str(fact.get("value") or fact.get("fact") or "").strip()
        if not entity or not value:
            continue
        detail = f"- {entity} — {fact_type}: {value}"
        as_of = str(fact.get("as_of") or "").strip()
        source_url = str(fact.get("source_url") or "").strip()
        suffixes = []
        if as_of:
            suffixes.append(f"as of {as_of}")
        if source_url:
            suffixes.append(f"source: {source_url}")
        if suffixes:
            detail += f" ({'; '.join(suffixes)})"
        lines.append(detail)
    if len(lines) == 1:
        return ""
    lines.append(
        "These come from earlier verified research runs. Treat them as leads, not as current truth: "
        "re-verify funding, pricing, valuations, release status, limits, and benchmarks before citing."
    )
    return "\n".join(lines)


def lookup_prior_facts_block(
    query: str,
    *,
    limit: int = DEFAULT_INJECTION_LIMIT,
    db_path: str | None = None,
) -> str:
    """Convenience wrapper: lookup + format. Returns "" when nothing matches."""
    try:
        return build_prior_facts_block(lookup_facts_for_query(query, limit=limit, db_path=db_path))
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.debug("Prior-fact block construction failed", exc_info=True)
        return ""

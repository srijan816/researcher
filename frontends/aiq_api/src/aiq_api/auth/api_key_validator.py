# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API-key validators for service-to-service AI-Q API access."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import uuid
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from aiq_agent.auth import Principal

from .base import TokenValidator

_api_key_schema_initialized: set[str] = set()


def generate_api_key() -> str:
    """Return a new plaintext API key. The caller must show it only once."""
    return f"aiq_{secrets.token_urlsafe(32)}"


def hash_api_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_postgres(db_url: str) -> bool:
    return db_url.startswith("postgres")


def ensure_api_key_table(db_url: str) -> None:
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        conn.commit()


def create_api_key(db_url: str, name: str, owner: Principal) -> dict[str, Any]:
    plaintext = generate_api_key()
    key_hash = hash_api_key(plaintext)
    key_id = str(uuid.uuid4())
    created_at = _now()

    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        conn.execute(
            text(
                "INSERT INTO api_keys "
                "(key_id, key_hash, key_prefix, name, owner_auth_type, owner_subject, owner_email, created_at) "
                "VALUES (:key_id, :key_hash, :key_prefix, :name, :owner_auth_type, "
                ":owner_subject, :owner_email, :created_at)"
            ),
            {
                "key_id": key_id,
                "key_hash": key_hash,
                "key_prefix": plaintext[:12],
                "name": name.strip() or "API Key",
                "owner_auth_type": owner.type,
                "owner_subject": owner.sub,
                "owner_email": owner.email,
                "created_at": created_at,
            },
        )
        conn.commit()

    return {
        "id": key_id,
        "name": name.strip() or "API Key",
        "key": plaintext,
        "prefix": plaintext[:12],
        "created_at": created_at,
        "last_used_at": None,
    }


def list_api_keys(db_url: str, owner: Principal) -> list[dict[str, Any]]:
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        rows = conn.execute(
            text(
                "SELECT key_id, key_prefix, name, created_at, last_used_at "
                "FROM api_keys "
                "WHERE owner_auth_type = :owner_auth_type "
                "AND owner_subject = :owner_subject "
                "AND revoked_at IS NULL "
                "ORDER BY created_at DESC"
            ),
            {"owner_auth_type": owner.type, "owner_subject": owner.sub},
        ).mappings()
        return [
            {
                "id": row["key_id"],
                "name": row["name"],
                "prefix": row["key_prefix"],
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
            }
            for row in rows
        ]


def revoke_api_key(db_url: str, key_id: str, owner: Principal) -> bool:
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        result = conn.execute(
            text(
                "UPDATE api_keys SET revoked_at = :revoked_at "
                "WHERE key_id = :key_id "
                "AND owner_auth_type = :owner_auth_type "
                "AND owner_subject = :owner_subject "
                "AND revoked_at IS NULL"
            ),
            {
                "key_id": key_id,
                "owner_auth_type": owner.type,
                "owner_subject": owner.sub,
                "revoked_at": _now(),
            },
        )
        conn.commit()
        return bool(result.rowcount)


def lookup_api_key(db_url: str, token: str) -> dict[str, Any] | None:
    token_hash = hash_api_key(token)
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        row = (
            conn.execute(
                text(
                    "SELECT key_id, name, owner_auth_type, owner_subject, owner_email "
                    "FROM api_keys WHERE key_hash = :key_hash AND revoked_at IS NULL"
                ),
                {"key_hash": token_hash},
            )
            .mappings()
            .first()
        )
        if row is None:
            return None

        data = dict(row)
        owner_role = _lookup_owner_role(conn, str(data.get("owner_subject") or ""))
        if owner_role:
            data["owner_role"] = owner_role

        conn.execute(
            text("UPDATE api_keys SET last_used_at = :last_used_at WHERE key_id = :key_id"),
            {"key_id": row["key_id"], "last_used_at": _now()},
        )
        conn.commit()
        return data


def _connection(db_url: str):
    from ..jobs.event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    return engine.connect()


def _ensure_schema(conn: Connection, db_url: str) -> None:
    if db_url in _api_key_schema_initialized:
        return
    conn.execute(text(_table_sql(db_url)))
    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_api_keys_owner ON api_keys(owner_auth_type, owner_subject)"))
    _api_key_schema_initialized.add(db_url)


def _table_sql(db_url: str) -> str:
    timestamp_type = "TIMESTAMP WITH TIME ZONE" if _is_postgres(db_url) else "DATETIME"
    return (
        "CREATE TABLE IF NOT EXISTS api_keys ("
        "  key_id VARCHAR PRIMARY KEY,"
        "  key_hash VARCHAR NOT NULL,"
        "  key_prefix VARCHAR NOT NULL,"
        "  name VARCHAR NOT NULL,"
        "  owner_auth_type VARCHAR NOT NULL,"
        "  owner_subject VARCHAR NOT NULL,"
        "  owner_email VARCHAR,"
        f"  created_at {timestamp_type} NOT NULL,"
        f"  last_used_at {timestamp_type},"
        f"  revoked_at {timestamp_type}"
        ")"
    )


def _api_key_user(row: Mapping[str, Any], token: str) -> dict[str, Any]:
    key_id = str(row["key_id"])
    name = str(row.get("name") or "API Key Client")
    owner_type = str(row.get("owner_auth_type") or "api_key")
    owner_subject = str(row.get("owner_subject") or f"api_key:{key_id}")
    return {
        # Generated API keys act as the owning user, so jobs submitted by
        # external apps land in that user's durable history.
        "type": owner_type,
        "sub": owner_subject,
        "email": row.get("owner_email"),
        "name": name,
        "role": row.get("owner_role"),
        "token": token,
        "skip_clarifier": False,
        "credential_type": "api_key",
        "api_key_id": key_id,
        "api_key_owner_type": owner_type,
        "api_key_owner_subject": owner_subject,
    }


def _lookup_owner_role(conn: Connection, owner_subject: str) -> str | None:
    if not owner_subject:
        return None
    try:
        row = (
            conn.execute(
                text("SELECT role FROM local_auth_users WHERE username = :username"),
                {"username": owner_subject},
            )
            .mappings()
            .first()
        )
    except Exception:
        return None
    if row is None:
        return None
    role = row.get("role")
    return str(role) if role else None


class DatabaseAPIKeyValidator(TokenValidator):
    """Validate generated ``aiq_`` bearer API keys stored in the job database."""

    def __init__(self, db_url: str):
        self._db_url = db_url

    def can_handle(self, token: str) -> bool:
        return token.startswith("aiq_")

    async def validate(self, token: str) -> dict[str, Any] | None:
        row = await asyncio.to_thread(lookup_api_key, self._db_url, token)
        if row is None:
            return None
        return _api_key_user(row, token)


class StaticAPIKeyValidator(TokenValidator):
    """Validate bearer tokens against configured static API keys."""

    def __init__(self, keys: list[str]):
        self._keys = [key.strip() for key in keys if key.strip()]

    def can_handle(self, token: str) -> bool:
        return bool(token and self._keys)

    async def validate(self, token: str) -> dict[str, Any] | None:
        for key in self._keys:
            if hmac.compare_digest(token, key):
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
                return {
                    "type": "api_key",
                    "sub": f"api_key:{digest}",
                    "email": None,
                    "name": "API Key Client",
                    "token": token,
                    "skip_clarifier": False,
                }
        return None

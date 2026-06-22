# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local username/password users for small private AI-Q deployments."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from collections.abc import Mapping
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

DEFAULT_LOCAL_USERS = ("srijan", "mai", "jami", "naveen", "saurav", "sreyan")
DEFAULT_ADMIN_USERS = ("srijan",)
PASSWORD_ALGORITHM = "pbkdf2_sha256"  # pragma: allowlist secret
PASSWORD_ITERATIONS = 260_000
TOKEN_PREFIX = "aiq_local."
DEFAULT_TOKEN_TTL_SECONDS = 24 * 60 * 60
DEFAULT_REMEMBER_ME_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60

_local_user_schema_initialized: set[str] = set()


class LocalAuthError(RuntimeError):
    """Raised when local auth cannot be completed safely."""


def local_auth_enabled() -> bool:
    return os.environ.get("AIQ_LOCAL_AUTH_ENABLED", "false").lower() == "true"


def get_local_auth_secret() -> str:
    secret = os.environ.get("AIQ_AUTH_TOKEN_SECRET") or os.environ.get("NEXTAUTH_SECRET")
    if not secret:
        raise LocalAuthError("AIQ_AUTH_TOKEN_SECRET must be set when local auth is enabled")
    return secret


def get_local_auth_ttl_seconds(remember_me: bool = False) -> int:
    raw = os.environ.get("AIQ_AUTH_TOKEN_REMEMBER_ME_TTL_SECONDS" if remember_me else "AIQ_AUTH_TOKEN_TTL_SECONDS")
    if not raw:
        return DEFAULT_REMEMBER_ME_TOKEN_TTL_SECONDS if remember_me else DEFAULT_TOKEN_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_REMEMBER_ME_TOKEN_TTL_SECONDS if remember_me else DEFAULT_TOKEN_TTL_SECONDS
    return max(900, min(value, 7 * 24 * 60 * 60))


def ensure_local_user_table(db_url: str) -> None:
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        conn.commit()


def seed_default_local_users(db_url: str) -> None:
    """Create the configured local users if they do not already exist."""

    users = _configured_users()
    admins = _configured_admin_users()
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        for username in users:
            role = "admin" if username in admins else "user"
            display_name = _display_name(username)
            conn.execute(
                _local_user_insert_sql(db_url),
                {
                    "username": username,
                    "display_name": display_name,
                    "email": f"{username}@local",
                    "role": role,
                    "password_hash": hash_password(username),
                    "must_change_password": True,
                    "is_active": True,
                },
            )
        conn.commit()


def authenticate_local_user(
    db_url: str,
    username: str,
    password: str,
    secret: str,
    remember_me: bool = False,
) -> dict[str, Any] | None:
    """Validate username/password and return a signed local auth token."""

    normalized = _normalize_username(username)
    if not normalized or not password:
        return None

    user = get_local_user(db_url, normalized)
    if user is None or not user.get("is_active"):
        return None
    if not verify_password(password, str(user["password_hash"])):
        return None

    return _login_response(user, issue_local_user_token(user, secret, remember_me=remember_me))


def refresh_local_user_token(db_url: str, token: str, secret: str) -> dict[str, Any] | None:
    payload = verify_local_user_token(token, secret)
    if payload is None:
        return None
    user = get_local_user(db_url, str(payload.get("sub") or ""))
    if user is None or not user.get("is_active"):
        return None
    return _login_response(
        user,
        issue_local_user_token(user, secret, remember_me=bool(payload.get("remember_me", False))),
    )


def change_local_user_password(db_url: str, username: str, current_password: str, new_password: str) -> bool:
    normalized = _normalize_username(username)
    if len(new_password) < 8:
        raise ValueError("New password must be at least 8 characters")

    user = get_local_user(db_url, normalized)
    if user is None or not user.get("is_active"):
        return False
    if not verify_password(current_password, str(user["password_hash"])):
        return False

    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        conn.execute(
            text(
                "UPDATE local_auth_users SET password_hash = :password_hash, "
                "must_change_password = :must_change_password, updated_at = "
                + ("NOW()" if _is_postgres(db_url) else "CURRENT_TIMESTAMP")
                + " WHERE username = :username"
            ),
            {
                "username": normalized,
                "password_hash": hash_password(new_password),
                "must_change_password": False,
            },
        )
        conn.commit()
    return True


def get_local_user(db_url: str, username: str) -> dict[str, Any] | None:
    normalized = _normalize_username(username)
    if not normalized:
        return None
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        row = (
            conn.execute(
                text(
                    "SELECT username, display_name, email, role, password_hash, "
                    "must_change_password, is_active, created_at, updated_at "
                    "FROM local_auth_users WHERE username = :username"
                ),
                {"username": normalized},
            )
            .mappings()
            .first()
        )
        return _coerce_user_row(row) if row is not None else None


def list_local_users(db_url: str) -> list[dict[str, Any]]:
    with _connection(db_url) as conn:
        _ensure_schema(conn, db_url)
        rows = (
            conn.execute(
                text(
                    "SELECT username, display_name, email, role, must_change_password, "
                    "is_active, created_at, updated_at "
                    "FROM local_auth_users ORDER BY username"
                )
            )
            .mappings()
            .all()
        )
        return [_public_user(row) for row in rows]


def issue_local_user_token(user: Mapping[str, Any], secret: str, *, remember_me: bool = False) -> dict[str, Any]:
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=get_local_auth_ttl_seconds(remember_me))
    payload = {
        "typ": "aiq_local_user",
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "type": "local_user",
        "sub": str(user["username"]),
        "username": str(user["username"]),
        "name": user.get("display_name") or _display_name(str(user["username"])),
        "email": user.get("email") or f"{user['username']}@local",
        "role": user.get("role") or "user",
        "remember_me": remember_me,
    }
    payload_b64 = _b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    sig_b64 = _sign_payload(payload_b64, secret)
    return {
        "token": f"{TOKEN_PREFIX}{payload_b64}.{sig_b64}",
        "expires_at": expires_at.isoformat(),
        "expires_in": get_local_auth_ttl_seconds(remember_me),
        "payload": payload,
    }


def verify_local_user_token(token: str, secret: str) -> dict[str, Any] | None:
    if not token.startswith(TOKEN_PREFIX):
        return None
    try:
        body = token.removeprefix(TOKEN_PREFIX)
        payload_b64, signature_b64 = body.split(".", 1)
    except ValueError:
        return None

    expected = _sign_payload(payload_b64, secret)
    if not hmac.compare_digest(signature_b64, expected):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict) or payload.get("typ") != "aiq_local_user":
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or datetime.now(UTC).timestamp() >= exp:
        return None
    return payload


def local_token_to_user(db_url: str, token: str, secret: str) -> dict[str, Any] | None:
    payload = verify_local_user_token(token, secret)
    if payload is None:
        return None
    user = get_local_user(db_url, str(payload.get("sub") or ""))
    if user is None or not user.get("is_active"):
        return None
    return {
        "type": "local_user",
        "sub": str(user["username"]),
        "username": str(user["username"]),
        "email": user.get("email"),
        "name": user.get("display_name"),
        "role": user.get("role") or "user",
        "is_admin": (user.get("role") == "admin"),
        "skip_clarifier": False,
    }


def hash_password(password: str, *, salt: str | None = None, iterations: int = PASSWORD_ITERATIONS) -> str:
    salt = salt or secrets.token_urlsafe(18)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{PASSWORD_ALGORITHM}${iterations}${salt}${digest_b64}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, raw_iterations, salt, expected = encoded.split("$", 3)
        iterations = int(raw_iterations)
    except ValueError:
        return False
    if algorithm != PASSWORD_ALGORITHM:
        return False
    actual = hash_password(password, salt=salt, iterations=iterations).split("$", 3)[3]
    return hmac.compare_digest(actual, expected)


def _login_response(user: Mapping[str, Any], token_data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "token": token_data["token"],
        "expires_at": token_data["expires_at"],
        "expires_in": token_data["expires_in"],
        "user": _public_user(user),
    }


def _connection(db_url: str):
    from ..jobs.event_store import EventStore

    engine = EventStore._get_or_create_sync_engine(db_url)
    return engine.connect()


def _ensure_schema(conn: Connection, db_url: str) -> None:
    if db_url in _local_user_schema_initialized:
        return
    conn.execute(text(_table_sql(db_url)))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_local_auth_users_role ON local_auth_users(role)"))
    _local_user_schema_initialized.add(db_url)


def _table_sql(db_url: str) -> str:
    timestamp_type = "TIMESTAMP WITH TIME ZONE" if _is_postgres(db_url) else "DATETIME"
    bool_type = "BOOLEAN"
    return (
        "CREATE TABLE IF NOT EXISTS local_auth_users ("
        "  username VARCHAR PRIMARY KEY,"
        "  display_name VARCHAR NOT NULL,"
        "  email VARCHAR,"
        "  role VARCHAR NOT NULL,"
        "  password_hash VARCHAR NOT NULL,"
        f"  must_change_password {bool_type} NOT NULL DEFAULT TRUE,"
        f"  is_active {bool_type} NOT NULL DEFAULT TRUE,"
        f"  created_at {timestamp_type} DEFAULT " + ("NOW()" if _is_postgres(db_url) else "CURRENT_TIMESTAMP") + ","
        f"  updated_at {timestamp_type} DEFAULT " + ("NOW()" if _is_postgres(db_url) else "CURRENT_TIMESTAMP") + ")"
    )


def _local_user_insert_sql(db_url: str):
    if _is_postgres(db_url):
        return text(
            "INSERT INTO local_auth_users "
            "(username, display_name, email, role, password_hash, must_change_password, is_active) "
            "VALUES (:username, :display_name, :email, :role, :password_hash, :must_change_password, :is_active) "
            "ON CONFLICT(username) DO NOTHING"
        )
    return text(
        "INSERT OR IGNORE INTO local_auth_users "
        "(username, display_name, email, role, password_hash, must_change_password, is_active) "
        "VALUES (:username, :display_name, :email, :role, :password_hash, :must_change_password, :is_active)"
    )


def _configured_users() -> tuple[str, ...]:
    raw = os.environ.get("AIQ_LOCAL_USERS")
    values = raw.split(",") if raw else DEFAULT_LOCAL_USERS
    users = tuple(dict.fromkeys(_normalize_username(value) for value in values if _normalize_username(value)))
    return users or DEFAULT_LOCAL_USERS


def _configured_admin_users() -> set[str]:
    raw = os.environ.get("AIQ_LOCAL_ADMIN_USERS")
    values = raw.split(",") if raw else DEFAULT_ADMIN_USERS
    return {_normalize_username(value) for value in values if _normalize_username(value)}


def _public_user(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "username": str(row["username"]),
        "display_name": row.get("display_name") or _display_name(str(row["username"])),
        "email": row.get("email"),
        "role": row.get("role") or "user",
        "must_change_password": _boolish(row.get("must_change_password")),
        "is_active": _boolish(row.get("is_active", True)),
        "created_at": _iso_or_none(row.get("created_at")),
        "updated_at": _iso_or_none(row.get("updated_at")),
    }


def _coerce_user_row(row: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    data["must_change_password"] = _boolish(data.get("must_change_password"))
    data["is_active"] = _boolish(data.get("is_active", True))
    return data


def _normalize_username(username: str | None) -> str:
    return str(username or "").strip().lower()


def _display_name(username: str) -> str:
    return username[:1].upper() + username[1:]


def _is_postgres(db_url: str) -> bool:
    return db_url.startswith("postgres")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "t", "true", "yes"}
    return bool(value)


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii")).decode("utf-8")


def _sign_payload(payload_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64encode(digest)

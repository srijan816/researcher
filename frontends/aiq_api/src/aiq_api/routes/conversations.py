# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Database-backed UI conversation/session snapshots.

The frontend still keeps a local cache for fast rendering, but this route is
the durable cross-device source of truth for the session list and chat history.
Deep research artifacts remain in the async job/event tables.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from sqlalchemy import text

from aiq_agent.auth import Principal

from ..jobs.access import require_verified_principal
from ..jobs.event_store import EventStore

_conversation_schema_initialized: set[str] = set()


class ConversationSyncRequest(BaseModel):
    """Bulk upsert request for browser UI conversations."""

    conversations: list[dict[str, Any]] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    """Conversation snapshots visible to the current caller."""

    conversations: list[dict[str, Any]]


def register_conversation_routes(app_or_router: APIRouter, db_url: str) -> None:
    """Register conversation snapshot routes under /v1/conversations."""

    router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

    @router.get("", response_model=ConversationListResponse)
    async def list_conversations() -> ConversationListResponse:
        principal = _conversation_principal(require_verified_principal())
        rows = await asyncio.get_running_loop().run_in_executor(None, _list_conversations, db_url, principal)
        return ConversationListResponse(conversations=rows)

    @router.post("/sync", response_model=ConversationListResponse)
    async def sync_conversations(req: ConversationSyncRequest) -> ConversationListResponse:
        principal = _conversation_principal(require_verified_principal())
        conversations = [_clean_conversation_payload(item) for item in req.conversations]
        conversations = [item for item in conversations if item is not None]
        await asyncio.get_running_loop().run_in_executor(None, _upsert_conversations, db_url, principal, conversations)
        rows = await asyncio.get_running_loop().run_in_executor(None, _list_conversations, db_url, principal)
        return ConversationListResponse(conversations=rows)

    @router.delete("/{conversation_id}")
    async def delete_conversation(conversation_id: str) -> dict[str, Any]:
        principal = _conversation_principal(require_verified_principal())
        deleted = await asyncio.get_running_loop().run_in_executor(
            None,
            _delete_conversation,
            db_url,
            principal,
            conversation_id,
        )
        return {"conversation_id": conversation_id, "deleted": deleted}

    @router.delete("")
    async def delete_all_conversations() -> dict[str, Any]:
        principal = _conversation_principal(require_verified_principal())
        deleted = await asyncio.get_running_loop().run_in_executor(None, _delete_all_conversations, db_url, principal)
        return {"deleted": deleted}

    app_or_router.include_router(router)


def _conversation_principal(principal: Principal) -> Principal:
    """Use a shared no-auth owner so LAN devices see the same local history."""

    if os.environ.get("REQUIRE_AUTH", "false").lower() == "true":
        return principal
    return Principal(type="no_auth", sub="default-user", email=None)


def _clean_conversation_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    conversation_id = payload.get("id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        return None
    if not isinstance(payload.get("messages", []), list):
        raise HTTPException(400, f"Conversation {conversation_id} has invalid messages")
    return payload


def _list_conversations(db_url: str, principal: Principal) -> list[dict[str, Any]]:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        _ensure_conversation_schema(conn, db_url)
        if principal.role == "admin":
            rows = (
                conn.execute(
                    text("SELECT payload FROM ui_conversations ORDER BY updated_at DESC"),
                )
                .mappings()
                .all()
            )
        else:
            rows = (
                conn.execute(
                    text(
                        "SELECT payload FROM ui_conversations "
                        "WHERE owner_auth_type = :owner_auth_type AND owner_subject = :owner_subject "
                        "ORDER BY updated_at DESC"
                    ),
                    _principal_params(principal),
                )
                .mappings()
                .all()
            )
        return [_parse_payload(row["payload"]) for row in rows]


def _upsert_conversations(db_url: str, principal: Principal, conversations: list[dict[str, Any]]) -> None:
    if not conversations:
        return
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        _ensure_conversation_schema(conn, db_url)
        for conversation in conversations:
            conn.execute(
                _conversation_upsert_sql(db_url),
                {
                    **_principal_params(principal),
                    "conversation_id": conversation["id"],
                    "title": str(conversation.get("title") or "New Session")[:512],
                    "payload": json.dumps(conversation, default=str),
                },
            )
        conn.commit()


def _delete_conversation(db_url: str, principal: Principal, conversation_id: str) -> int:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        _ensure_conversation_schema(conn, db_url)
        if principal.role == "admin":
            result = conn.execute(
                text("DELETE FROM ui_conversations WHERE conversation_id = :conversation_id"),
                {"conversation_id": conversation_id},
            )
        else:
            result = conn.execute(
                text(
                    "DELETE FROM ui_conversations "
                    "WHERE owner_auth_type = :owner_auth_type AND owner_subject = :owner_subject "
                    "AND conversation_id = :conversation_id"
                ),
                {**_principal_params(principal), "conversation_id": conversation_id},
            )
        conn.commit()
        return result.rowcount or 0


def _delete_all_conversations(db_url: str, principal: Principal) -> int:
    engine = EventStore._get_or_create_sync_engine(db_url)
    with engine.connect() as conn:
        _ensure_conversation_schema(conn, db_url)
        result = conn.execute(
            text(
                "DELETE FROM ui_conversations "
                "WHERE owner_auth_type = :owner_auth_type AND owner_subject = :owner_subject"
            ),
            _principal_params(principal),
        )
        conn.commit()
        return result.rowcount or 0


def _parse_payload(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _principal_params(principal: Principal) -> dict[str, str]:
    return {
        "owner_auth_type": principal.type,
        "owner_subject": principal.sub,
    }


def _ensure_conversation_schema(conn, db_url: str) -> None:
    if db_url in _conversation_schema_initialized:
        return
    conn.execute(text(_conversation_table_sql(db_url)))
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_ui_conversations_owner_updated "
            "ON ui_conversations(owner_auth_type, owner_subject, updated_at)"
        )
    )
    conn.commit()
    _conversation_schema_initialized.add(db_url)


def _conversation_table_sql(db_url: str) -> str:
    timestamp_type = (
        "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
        if db_url.startswith("postgres")
        else "DATETIME DEFAULT CURRENT_TIMESTAMP"
    )
    text_type = "TEXT"
    return (
        "CREATE TABLE IF NOT EXISTS ui_conversations ("
        "  owner_auth_type VARCHAR NOT NULL,"
        "  owner_subject VARCHAR NOT NULL,"
        "  conversation_id VARCHAR NOT NULL,"
        "  title VARCHAR,"
        f"  payload {text_type} NOT NULL,"
        f"  created_at {timestamp_type},"
        f"  updated_at {timestamp_type},"
        "  PRIMARY KEY (owner_auth_type, owner_subject, conversation_id)"
        ")"
    )


def _conversation_upsert_sql(db_url: str):
    if db_url.startswith("postgres"):
        return text(
            "INSERT INTO ui_conversations "
            "(owner_auth_type, owner_subject, conversation_id, title, payload) "
            "VALUES (:owner_auth_type, :owner_subject, :conversation_id, :title, :payload) "
            "ON CONFLICT(owner_auth_type, owner_subject, conversation_id) DO UPDATE SET "
            "title = excluded.title, payload = excluded.payload, updated_at = NOW()"
        )
    return text(
        "INSERT INTO ui_conversations "
        "(owner_auth_type, owner_subject, conversation_id, title, payload, updated_at) "
        "VALUES (:owner_auth_type, :owner_subject, :conversation_id, :title, :payload, CURRENT_TIMESTAMP) "
        "ON CONFLICT(owner_auth_type, owner_subject, conversation_id) DO UPDATE SET "
        "title = excluded.title, payload = excluded.payload, updated_at = CURRENT_TIMESTAMP"
    )

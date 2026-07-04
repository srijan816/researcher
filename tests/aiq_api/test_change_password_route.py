# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/auth/change-password (local auth users)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from aiq_agent.auth import Principal
from aiq_api.auth.local_users import authenticate_local_user
from aiq_api.routes.auth import register_local_auth_routes

_SECRET = "unit-test-secret-not-a-real-one"  # pragma: allowlist secret


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    """FastAPI app with local auth routes over a throwaway sqlite DB, seeded with user 'alice'."""
    monkeypatch.setenv("AIQ_AUTH_TOKEN_SECRET", _SECRET)
    monkeypatch.setenv("AIQ_LOCAL_USERS", "alice")
    monkeypatch.setenv("AIQ_LOCAL_ADMIN_USERS", "alice")
    db_url = f"sqlite:///{tmp_path}/auth.db"

    app = FastAPI()
    asyncio.run(register_local_auth_routes(app, db_url))

    with TestClient(app) as client:
        yield client, db_url


def _as_principal(username: str = "alice"):
    return patch(
        "aiq_api.routes.auth.require_verified_principal",
        return_value=Principal(type="local", sub=username, email=f"{username}@local"),
    )


def test_change_password_success_and_login_with_new_password(app_client):
    client, db_url = app_client
    # Seeded initial password equals the username.
    with _as_principal():
        resp = client.post(
            "/v1/auth/change-password",
            json={"current_password": "alice", "new_password": "new-password-123"},
        )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Old password no longer works; new one does — same hash scheme as login.
    assert authenticate_local_user(db_url, "alice", "alice", _SECRET) is None
    assert authenticate_local_user(db_url, "alice", "new-password-123", _SECRET) is not None


def test_change_password_wrong_current_password_returns_400(app_client):
    client, _ = app_client
    with _as_principal():
        resp = client.post(
            "/v1/auth/change-password",
            json={"current_password": "wrong-password", "new_password": "new-password-123"},
        )
    assert resp.status_code == 400
    assert "password" not in str(resp.json()).lower() or "incorrect" in str(resp.json()).lower()


def test_change_password_short_new_password_rejected(app_client):
    client, _ = app_client
    with _as_principal():
        resp = client.post(
            "/v1/auth/change-password",
            json={"current_password": "alice", "new_password": "short"},
        )
    # Pydantic min_length=8 rejects before the handler runs.
    assert resp.status_code == 422


def test_change_password_unauthenticated_returns_auth_error(app_client):
    client, _ = app_client
    with patch(
        "aiq_api.routes.auth.require_verified_principal",
        side_effect=HTTPException(403, "Verified principal required"),
    ):
        resp = client.post(
            "/v1/auth/change-password",
            json={"current_password": "alice", "new_password": "new-password-123"},
        )
    assert resp.status_code in (401, 403)


def test_change_password_unknown_user_returns_400(app_client):
    client, _ = app_client
    with _as_principal("ghost"):
        resp = client.post(
            "/v1/auth/change-password",
            json={"current_password": "whatever1", "new_password": "new-password-123"},
        )
    assert resp.status_code == 400

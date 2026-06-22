# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiq_api.auth import local_users
from aiq_api.auth.local_users import authenticate_local_user
from aiq_api.auth.local_users import change_local_user_password
from aiq_api.auth.local_users import get_local_user
from aiq_api.auth.local_users import local_token_to_user
from aiq_api.auth.local_users import seed_default_local_users
from aiq_api.auth.local_users import verify_local_user_token
from aiq_api.jobs.event_store import EventStore


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite+aiosqlite:///{tmp_path / 'local_auth.db'}"


@pytest.fixture(autouse=True)
def clear_caches():
    EventStore._sync_engine_cache.clear()
    local_users._local_user_schema_initialized.clear()
    yield
    EventStore._sync_engine_cache.clear()
    local_users._local_user_schema_initialized.clear()


def test_seeded_default_users_can_login_and_receive_role(db_url):
    seed_default_local_users(db_url)

    login = authenticate_local_user(db_url, "srijan", "srijan", "test-secret")

    assert login is not None
    assert login["user"]["username"] == "srijan"
    assert login["user"]["role"] == "admin"
    assert login["user"]["must_change_password"] is True

    principal = local_token_to_user(db_url, login["token"], "test-secret")
    assert principal is not None
    assert principal["sub"] == "srijan"
    assert principal["role"] == "admin"


def test_remember_me_extends_token_and_survives_refresh(db_url):
    seed_default_local_users(db_url)

    login = authenticate_local_user(db_url, "mai", "mai", "test-secret", remember_me=True)

    assert login is not None
    assert login["expires_in"] > 24 * 60 * 60

    payload = verify_local_user_token(login["token"], "test-secret")
    assert payload is not None
    assert payload["remember_me"] is True

    principal = local_token_to_user(db_url, login["token"], "test-secret")
    assert principal is not None
    assert principal["sub"] == "mai"


def test_password_change_replaces_default_password(db_url):
    seed_default_local_users(db_url)

    assert change_local_user_password(db_url, "mai", "mai", "better-password") is True
    assert authenticate_local_user(db_url, "mai", "mai", "test-secret") is None

    login = authenticate_local_user(db_url, "mai", "better-password", "test-secret")
    assert login is not None
    assert login["user"]["must_change_password"] is False
    assert get_local_user(db_url, "mai")["must_change_password"] is False

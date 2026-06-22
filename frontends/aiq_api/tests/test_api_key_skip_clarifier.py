# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave 2 W2.3 — API-key auth must skip the clarifier (and thus plan_preview).

API-key callers are programmatic: they cannot answer clarifying questions or
approve a plan preview, so the clarifier node must be bypassed entirely.
These tests pin that behavior in both validator paths (static env-var keys
and DB-backed generated keys).
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest


class TestStaticAPIKeyValidator:
    @pytest.mark.asyncio
    async def test_validate_sets_skip_clarifier_true(self) -> None:
        from aiq_api.auth.api_key_validator import StaticAPIKeyValidator

        key = "test-static-key-1234567890"
        validator = StaticAPIKeyValidator({key})
        with patch.dict(os.environ, {}, clear=False):
            out = await validator.validate(key)
        assert out is not None
        assert out["type"] == "api_key"
        assert out["skip_clarifier"] is True, (
            "API-key callers must skip the clarifier — they can't answer "
            "questions or approve a plan preview."
        )


class TestApiKeyUserDict:
    def test_api_key_user_helper_sets_skip_clarifier_true(self) -> None:
        from aiq_api.auth.api_key_validator import _api_key_user

        row = {
            "key_id": "k-1",
            "name": "Test Key",
            "owner_auth_type": "jwt",
            "owner_subject": "user-1",
            "owner_email": "user@example.com",
            "owner_role": "user",
        }
        out = _api_key_user(row, token="tok-xyz")
        assert out["type"] == "jwt"  # inherited from owner
        assert out["skip_clarifier"] is True, (
            "DB-backed generated API keys (acting as their owner) must skip "
            "the clarifier."
        )
        assert out["credential_type"] == "api_key"


def test_module_imports_cleanly() -> None:
    """Sanity: importing the validator module must not raise."""
    import aiq_api.auth.api_key_validator  # noqa: F401
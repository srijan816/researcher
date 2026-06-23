# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the internal-caller auth bypass fix.

The chat UI proxy hits the backend over internal Docker DNS (``aiq-agent``)
and forwards the user's idToken. The backend's ``is_external_request`` check
classifies such requests as internal because the ``Host`` header doesn't
match ``AIQ_EXTERNAL_HOSTNAMES``. Pre-fix, an internal request with a
*rejected* token fell through to ``detect_internal_caller``, which
returned a user dict with no ``sub`` claim — and the route's
``require_verified_principal()`` then 403'd.

Post-fix, ``resolve_request_user`` rejects the request with 401 whenever
``REQUIRE_AUTH=true`` and the presented token fails validation, regardless
of external/internal classification. This is the regression test.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from aiq_api.auth.middleware import resolve_request_user


class _StubValidator:
    """Returns a fixed user dict (or None to reject)."""

    def __init__(self, result: dict[str, Any] | None):
        self._result = result

    def can_handle(self, token: str) -> bool:
        return True

    async def validate(self, token: str) -> dict[str, Any] | None:
        return self._result


def _internal_headers(token: str | None = None) -> dict[bytes, bytes]:
    """Simulate the chat-UI proxy over Docker DNS — Host=aiq-agent (internal)."""
    headers: dict[bytes, bytes] = {
        b"host": b"aiq-agent:8000",
        b"user-agent": b"next-test",
    }
    if token is not None:
        headers[b"authorization"] = f"Bearer {token}".encode()
    return headers


def _external_headers(token: str | None = None) -> dict[bytes, bytes]:
    """Simulate an external browser request — Host=app2.sniperip.com."""
    headers: dict[bytes, bytes] = {
        b"host": b"app2.sniperip.com",
        b"user-agent": b"browser-test",
    }
    if token is not None:
        headers[b"authorization"] = f"Bearer {token}".encode()
    return headers


class TestInternalCallerWithInvalidToken:
    """The core regression: internal caller + invalid token + REQUIRE_AUTH."""

    @pytest.mark.asyncio
    async def test_internal_caller_invalid_token_require_auth_returns_401(self) -> None:
        """Pre-fix: returned (unverified_jwt dict, None) → route 403.
        Post-fix: returns (None, 401) — same as the external path."""
        validators = [_StubValidator(result=None)]  # token is always rejected
        headers = _internal_headers(token="bogus-token")

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=True,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user is None
        assert status == 401
        assert is_external is False
        assert error_code == "token_invalid"

    @pytest.mark.asyncio
    async def test_internal_caller_no_token_require_auth_returns_401(self) -> None:
        """Internal caller with no token and REQUIRE_AUTH=true must 401."""
        validators = [_StubValidator(result=None)]
        headers = _internal_headers(token=None)

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=True,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user is None
        assert status == 401
        assert is_external is False
        assert error_code == "token_missing"

    @pytest.mark.asyncio
    async def test_internal_caller_no_token_require_auth_false_returns_internal_dict(self) -> None:
        """With REQUIRE_AUTH=false, internal callers still get a synthetic
        internal identity (no token, no validation) — no 401."""
        validators: list = []
        headers = _internal_headers(token=None)

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=False,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user is not None
        assert user.get("type") == "internal"
        # No 'sub' — this is fine because REQUIRE_AUTH=false means routes
        # synthesize a no-auth principal instead of calling
        # require_verified_principal()'s 403 path.
        assert "sub" not in user
        assert status is None
        assert is_external is False


class TestInternalCallerWithValidToken:
    """The happy path: internal caller with a valid token must validate."""

    @pytest.mark.asyncio
    async def test_internal_caller_valid_token_require_auth_returns_user(self) -> None:
        valid_user = {"type": "local_user", "sub": "alice", "role": "user"}
        validators = [_StubValidator(result=valid_user)]
        headers = _internal_headers(token="good-token")

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=True,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user == valid_user
        assert status is None
        assert is_external is False
        assert error_code is None

    @pytest.mark.asyncio
    async def test_external_caller_invalid_token_require_auth_returns_401(self) -> None:
        """The pre-existing external path: still 401. The fix must not break this."""
        validators = [_StubValidator(result=None)]
        headers = _external_headers(token="bogus-token")

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=True,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user is None
        assert status == 401
        assert is_external is True
        assert error_code == "token_invalid"


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_internal_caller_with_token_require_auth_false_validates(self) -> None:
        """With REQUIRE_AUTH=false but a token present, the token is still
        validated. The validated user (with sub) is preferred over the
        synthetic internal identity — useful for audit logging."""
        valid_user = {"type": "local_user", "sub": "alice"}
        validators = [_StubValidator(result=valid_user)]
        headers = _internal_headers(token="good-token")

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=False,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user == valid_user
        assert "sub" in user
        assert status is None

    @pytest.mark.asyncio
    async def test_internal_caller_with_token_require_auth_false_token_rejected(self) -> None:
        """With REQUIRE_AUTH=false and a rejected token, fall back to the
        internal identity (no 401, because REQUIRE_AUTH is off)."""
        validators = [_StubValidator(result=None)]
        headers = _internal_headers(token="bogus-token")

        user, status, is_external, error_code = await resolve_request_user(
            headers,
            validators=validators,
            require_auth=False,
            external_hostnames={"app2.sniperip.com"},
        )

        assert user is not None
        assert user.get("type") == "unverified_jwt"
        assert status is None
        assert is_external is False

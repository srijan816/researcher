# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: AUTH_EXEMPT_PATHS reachable from internal callers.

The Next.js chat-UI server in the same docker network as ``aiq-agent``
calls ``/v1/auth/login`` and ``/v1/auth/refresh`` server-side, without
forwarding a user JWT (there is no JWT to forward yet — the user is
signing in).  These requests have ``Host: aiq-agent:8000``, which is NOT
in ``AIQ_EXTERNAL_HOSTNAMES``, so they are classified as internal.

Pre-fix, the ``is_external and path in AUTH_EXEMPT_PATHS`` branch in
``AuthMiddleware.__call__`` only fired for external requests, so the
login call fell through to ``resolve_request_user`` and got 401.
Post-fix, exempt paths are reachable from both external and internal
callers, so the chat-UI's local-users sign-in works.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from aiq_api.auth.middleware import AUTH_EXEMPT_PATHS
from aiq_api.auth.middleware import AuthMiddleware


class _StubValidator:
    def can_handle(self, token: str) -> bool:
        return True

    async def validate(self, token: str) -> dict[str, Any] | None:
        # Should never run for AUTH_EXEMPT_PATHS — no token is presented.
        raise AssertionError("validator should not be called for exempt paths")


def _internal_headers() -> dict[bytes, bytes]:
    """Simulate Next.js server in the chat-UI container — Host=aiq-agent (internal)."""
    return {b"host": b"aiq-agent:8000", b"user-agent": b"next-test"}


def _external_headers() -> dict[bytes, bytes]:
    """Simulate a real browser — Host matches AIQ_EXTERNAL_HOSTNAMES."""
    return {b"host": b"app2.sniperip.com", b"user-agent": b"browser-test"}


async def _call_middleware(middleware: AuthMiddleware, headers: dict[bytes, bytes]) -> tuple[int, dict]:
    """Drive the middleware once; return ``(status_code, json_body)``."""
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/auth/login",
        "headers": [(k, v) for k, v in headers.items()],
        "query_string": b"",
    }

    await middleware(scope, receive, send)

    # The middleware is expected to send one ``http.response.start`` and
    # one ``http.response.body`` message (in that order) before delegating
    # to the inner app.  For exempt paths it delegates directly, but the
    # inner stub captures calls instead of sending messages.
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = next((m for m in sent if m["type"] == "http.response.body"), None)
    if start is not None and body is not None:
        import json
        return start["status"], json.loads(body["body"])
    return 0, {}


class _CapturingApp:
    """Minimal ASGI app that records that it was called."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(AUTH_EXEMPT_PATHS))
async def test_exempt_path_reachable_from_internal_caller(path: str) -> None:
    """Internal call to an exempt path must reach the inner app (no 401).

    This is the regression that broke local-users sign-in: the Next.js
    server hit ``/v1/auth/login`` from inside the docker network and
    got 401 because the exempt branch was gated on ``is_external``.
    """
    app = _CapturingApp()
    middleware = AuthMiddleware(
        app=app,
        validators=[_StubValidator()],
        require_auth=True,
        external_hostnames={"app2.sniperip.com"},
    )

    headers = _internal_headers()
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(k, v) for k, v in headers.items()],
        "query_string": b"",
    }

    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await middleware(scope, receive, send)

    # The inner app must have been invoked — no 401, no 404.
    assert app.called, f"exempt path {path!r} was rejected for internal caller"
    # And no error response was generated.
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    if start is not None:
        assert start["status"] != 401, f"exempt path {path!r} returned 401 for internal caller"
        assert start["status"] != 404, f"exempt path {path!r} returned 404 for internal caller"


@pytest.mark.asyncio
async def test_login_endpoint_specifically_reachable_from_internal() -> None:
    """The exact failure mode the user reported: sign-in via the local-users
    provider calls POST /v1/auth/login from the chat-UI container, which
    is an internal caller.  It must not 401."""
    app = _CapturingApp()
    middleware = AuthMiddleware(
        app=app,
        validators=[_StubValidator()],
        require_auth=True,
        external_hostnames={"app2.sniperip.com"},
    )

    status, _ = await _call_middleware(middleware, _internal_headers())
    assert status != 401
    assert app.called


@pytest.mark.asyncio
async def test_exempt_path_still_reachable_from_external_caller() -> None:
    """Pre-existing external path must still work — the fix must not regress it."""
    app = _CapturingApp()
    middleware = AuthMiddleware(
        app=app,
        validators=[_StubValidator()],
        require_auth=True,
        external_hostnames={"app2.sniperip.com"},
    )

    status, _ = await _call_middleware(middleware, _external_headers())
    assert status != 401
    assert app.called


@pytest.mark.asyncio
async def test_non_exempt_internal_path_still_401s_when_require_auth() -> None:
    """A non-exempt internal path with no token must still 401.

    The fix only relaxes the exempt-path check; non-exempt internal
    requests still go through ``resolve_request_user`` and are rejected.
    """
    app = _CapturingApp()
    middleware = AuthMiddleware(
        app=app,
        validators=[_StubValidator()],
        require_auth=True,
        external_hostnames={"app2.sniperip.com"},
    )

    headers = _internal_headers()
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/conversations",
        "headers": [(k, v) for k, v in headers.items()],
        "query_string": b"",
    }

    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await middleware(scope, receive, send)

    assert not app.called
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = next(m for m in sent if m["type"] == "http.response.body")
    assert start["status"] == 401
    import json
    assert json.loads(body["body"]).get("error") == "token_missing"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local user authentication routes."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from ..auth.local_users import authenticate_local_user
from ..auth.local_users import change_local_user_password
from ..auth.local_users import ensure_local_user_table
from ..auth.local_users import get_local_auth_secret
from ..auth.local_users import get_local_user
from ..auth.local_users import list_local_users
from ..auth.local_users import refresh_local_user_token
from ..auth.local_users import seed_default_local_users
from ..jobs.access import require_verified_principal


class LocalUser(BaseModel):
    username: str
    display_name: str
    email: str | None = None
    role: str
    must_change_password: bool
    is_active: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=256)
    remember_me: bool = False


class LoginResponse(BaseModel):
    token: str
    expires_at: str
    expires_in: int
    user: LocalUser


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=256)


class UserListResponse(BaseModel):
    users: list[LocalUser]


async def register_local_auth_routes(app: FastAPI, db_url: str) -> None:
    """Register username/password auth endpoints and seed configured users."""

    secret = get_local_auth_secret()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ensure_local_user_table, db_url)
    await loop.run_in_executor(None, seed_default_local_users, db_url)

    @app.post("/v1/auth/login", response_model=LoginResponse, tags=["auth"])
    async def login(req: LoginRequest) -> LoginResponse:
        row = await asyncio.get_running_loop().run_in_executor(
            None,
            authenticate_local_user,
            db_url,
            req.username,
            req.password,
            secret,
            req.remember_me,
        )
        if row is None:
            raise HTTPException(401, "Invalid username or password")
        return LoginResponse(**row)

    @app.post("/v1/auth/refresh", response_model=LoginResponse, tags=["auth"])
    async def refresh(authorization: str | None = Header(default=None)) -> LoginResponse:
        token = _bearer_token(authorization)
        if not token:
            raise HTTPException(401, "Missing auth token")
        row = await asyncio.get_running_loop().run_in_executor(
            None,
            refresh_local_user_token,
            db_url,
            token,
            secret,
        )
        if row is None:
            raise HTTPException(401, "Invalid auth token")
        return LoginResponse(**row)

    @app.get("/v1/auth/me", response_model=LocalUser, tags=["auth"])
    async def me() -> LocalUser:
        principal = require_verified_principal()
        row = await asyncio.get_running_loop().run_in_executor(None, get_local_user, db_url, principal.sub)
        if row is None:
            raise HTTPException(404, "User not found")
        return LocalUser(**_public_row(row))

    @app.post("/v1/auth/change-password", tags=["auth"])
    async def change_password(req: ChangePasswordRequest) -> dict[str, bool]:
        principal = require_verified_principal()
        try:
            changed = await asyncio.get_running_loop().run_in_executor(
                None,
                change_local_user_password,
                db_url,
                principal.sub,
                req.current_password,
                req.new_password,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not changed:
            raise HTTPException(403, "Current password is incorrect")
        return {"changed": True}

    @app.get("/v1/auth/users", response_model=UserListResponse, tags=["auth"])
    async def users() -> UserListResponse:
        principal = require_verified_principal()
        if principal.role != "admin":
            raise HTTPException(404, "Not found")
        rows = await asyncio.get_running_loop().run_in_executor(None, list_local_users, db_url)
        return UserListResponse(users=[LocalUser(**row) for row in rows])


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        return None
    return authorization[7:].strip() or None


def _public_row(row: dict) -> dict:
    created_at = row.get("created_at")
    updated_at = row.get("updated_at")
    return {
        "username": row["username"],
        "display_name": row.get("display_name") or row["username"],
        "email": row.get("email"),
        "role": row.get("role") or "user",
        "must_change_password": bool(row.get("must_change_password")),
        "is_active": bool(row.get("is_active", True)),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
    }

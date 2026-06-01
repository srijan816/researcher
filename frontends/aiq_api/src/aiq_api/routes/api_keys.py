# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routes for generating and revoking AI-Q service API keys."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from ..auth.api_key_validator import create_api_key
from ..auth.api_key_validator import ensure_api_key_table
from ..auth.api_key_validator import list_api_keys
from ..auth.api_key_validator import revoke_api_key
from ..jobs.access import require_verified_principal


class APIKeyCreateRequest(BaseModel):
    name: str = Field(default="API Key", min_length=1, max_length=120)


class APIKeyMetadata(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: str
    last_used_at: str | None = None


class APIKeyCreateResponse(APIKeyMetadata):
    key: str


class APIKeyListResponse(BaseModel):
    api_keys: list[APIKeyMetadata]


async def register_api_key_routes(app: FastAPI, db_url: str) -> None:
    """Register generated API-key management routes."""
    await asyncio.get_running_loop().run_in_executor(None, ensure_api_key_table, db_url)

    @app.get("/v1/api-keys", response_model=APIKeyListResponse)
    async def list_keys() -> APIKeyListResponse:
        principal = require_verified_principal()
        rows = await asyncio.get_running_loop().run_in_executor(None, list_api_keys, db_url, principal)
        return APIKeyListResponse(api_keys=[APIKeyMetadata(**row) for row in rows])

    @app.post("/v1/api-keys", response_model=APIKeyCreateResponse)
    async def create_key(req: APIKeyCreateRequest) -> APIKeyCreateResponse:
        principal = require_verified_principal()
        row = await asyncio.get_running_loop().run_in_executor(None, create_api_key, db_url, req.name, principal)
        return APIKeyCreateResponse(**row)

    @app.delete("/v1/api-keys/{key_id}", status_code=204)
    async def revoke_key(key_id: str) -> None:
        principal = require_verified_principal()
        revoked = await asyncio.get_running_loop().run_in_executor(None, revoke_api_key, db_url, key_id, principal)
        if not revoked:
            raise HTTPException(404, "API key not found")

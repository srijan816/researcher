# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for backend-signed local user tokens."""

from __future__ import annotations

import asyncio

from .base import TokenValidator
from .local_users import TOKEN_PREFIX
from .local_users import local_token_to_user


class LocalUserTokenValidator(TokenValidator):
    """Validate ``aiq_local.*`` tokens issued by the local auth routes."""

    def __init__(self, db_url: str, secret: str):
        self._db_url = db_url
        self._secret = secret

    def can_handle(self, token: str) -> bool:
        return token.startswith(TOKEN_PREFIX)

    async def validate(self, token: str) -> dict | None:
        return await asyncio.to_thread(local_token_to_user, self._db_url, token, self._secret)

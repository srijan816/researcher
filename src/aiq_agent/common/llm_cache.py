# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redis-backed cache for LLM responses.

Wave 2 W2.2: caches adversarial-verifier verdicts keyed by (model, claim, evidence).
Fail-open by design — if Redis is unreachable, every operation becomes a no-op
and the caller falls through to the LLM. No reply is ever lost to a cache fault.

Key format: ``llmcache:<sha256>`` so cache entries are namespaced away from
other Redis tenants (e.g. websurfx's content-addressed keys).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 86_400  # 24h — verifier verdicts are stable per claim+evidence
KEY_PREFIX = "llmcache:"
SOCKET_TIMEOUT_SECONDS = 2.0


class LLMResponseCache:
    """Tiny Redis cache with fail-open semantics.

    Use ``LLMResponseCache.instance()`` — the class keeps a single client per
    process. Operations are best-effort: errors are logged at WARNING and the
    caller receives ``None`` from ``get`` / silent success from ``set``.
    """

    _instance: "LLMResponseCache | None" = None

    @classmethod
    def instance(cls) -> "LLMResponseCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test helper: drop the cached singleton so a fresh one is built."""
        cls._instance = None

    def __init__(self) -> None:
        self._url = os.environ.get("REDIS_URL", "").strip() or None
        self._client: Any = None
        self._enabled: bool = False
        self._init_client()

    def _init_client(self) -> None:
        if not self._url:
            logger.info("aiq.metrics llm_cache disabled (REDIS_URL not set)")
            return
        try:
            import redis  # local import so module loads even when redis is absent

            self._client = redis.Redis.from_url(
                self._url,
                socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
                socket_timeout=SOCKET_TIMEOUT_SECONDS,
            )
            self._client.ping()
            self._enabled = True
            logger.info("aiq.metrics llm_cache enabled at %s", self._url)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            self._client = None
            self._enabled = False
            logger.warning("aiq.metrics llm_cache disabled (redis init failed: %s)", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def make_key(parts: dict[str, Any]) -> str:
        """Stable SHA-256 key. ``parts`` must be a JSON-serializable dict.

        Using a single dict (not ``*args``) keeps the call site readable and
        guarantees stable ordering via ``sort_keys=True`` below.
        """
        canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{KEY_PREFIX}{digest}"

    def get(self, key: str) -> dict | None:
        if not self._enabled or self._client is None:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - cache miss on any error
            logger.warning("aiq.metrics llm_cache get_failed key=%s: %s", key[:24], exc)
            return None

    def set(self, key: str, value: dict, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
        """Returns True on a successful SET, False otherwise. Never raises."""
        if not self._enabled or self._client is None:
            return False
        try:
            payload = json.dumps(value, ensure_ascii=False, default=str)
            self._client.set(key, payload, ex=ttl)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("aiq.metrics llm_cache set_failed key=%s: %s", key[:24], exc)
            return False

    def stats(self) -> dict[str, int]:
        """Inspect cache state (for tests / health checks)."""
        if not self._enabled or self._client is None:
            return {"enabled": 0, "keys": 0}
        try:
            keys = sum(1 for _ in self._client.scan_iter(match=f"{KEY_PREFIX}*", count=500))
            return {"enabled": 1, "keys": keys}
        except Exception:  # noqa: BLE001
            return {"enabled": 1, "keys": -1}

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Redis-backed LLM response cache (Wave 2 W2.2).

Tests focus on the cache contract:
- Key determinism: same inputs → same key.
- Key isolation: different inputs → different keys.
- Fail-open: when REDIS_URL is unset, get returns None and set returns False.
- Module importable without a live Redis.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq_agent.common.llm_cache import DEFAULT_TTL_SECONDS
from aiq_agent.common.llm_cache import KEY_PREFIX
from aiq_agent.common.llm_cache import LLMResponseCache


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees a fresh cache singleton."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    LLMResponseCache.reset_instance()
    yield
    LLMResponseCache.reset_instance()


class TestKeyDerivation:
    """Cache keys must be stable and collision-resistant."""

    def test_same_inputs_produce_same_key(self) -> None:
        k1 = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1", "claim_text": "hello"})
        k2 = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1", "claim_text": "hello"})
        assert k1 == k2

    def test_different_text_produces_different_key(self) -> None:
        k1 = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1", "claim_text": "hello"})
        k2 = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1", "claim_text": "world"})
        assert k1 != k2

    def test_different_role_produces_different_key(self) -> None:
        k1 = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1", "claim_text": "x"})
        k2 = LLMResponseCache.make_key({"role": "planner", "claim_id": "c1", "claim_text": "x"})
        assert k1 != k2

    def test_dict_order_does_not_matter(self) -> None:
        k1 = LLMResponseCache.make_key({"a": 1, "b": 2})
        k2 = LLMResponseCache.make_key({"b": 2, "a": 1})
        assert k1 == k2

    def test_key_uses_llmcache_prefix(self) -> None:
        k = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1"})
        assert k.startswith(KEY_PREFIX)
        assert len(k) == len(KEY_PREFIX) + 64  # sha256 hex digest


class TestFailOpenWhenDisabled:
    """Without REDIS_URL, the cache must be a no-op (never raise, never cache)."""

    def test_disabled_without_redis_url(self) -> None:
        cache = LLMResponseCache.instance()
        assert cache.enabled is False

    def test_get_returns_none_when_disabled(self) -> None:
        cache = LLMResponseCache.instance()
        assert cache.get("llmcache:anything") is None

    def test_set_returns_false_when_disabled(self) -> None:
        cache = LLMResponseCache.instance()
        assert cache.set("llmcache:anything", {"v": 1}) is False

    def test_stats_reports_disabled(self) -> None:
        cache = LLMResponseCache.instance()
        assert cache.stats() == {"enabled": 0, "keys": 0}


class TestFailOpenOnRedisError:
    """Even when REDIS_URL is set, transport errors must degrade silently."""

    def test_redis_init_failure_disables_cache(self) -> None:
        with patch.dict(os.environ, {"REDIS_URL": "redis://nowhere:6379/0"}):
            # Simulate redis import succeeding but ping failing.
            fake_redis = MagicMock()
            fake_redis.Redis.from_url.return_value.ping.side_effect = ConnectionError("nope")
            with patch.dict("sys.modules", {"redis": fake_redis}):
                cache = LLMResponseCache.instance()
                assert cache.enabled is False

    def test_get_returns_none_on_runtime_error(self) -> None:
        with patch.dict(os.environ, {"REDIS_URL": "redis://nowhere:6379/0"}):
            fake_redis = MagicMock()
            client = fake_redis.Redis.from_url.return_value
            client.ping.return_value = True
            client.get.side_effect = ConnectionError("blip")
            with patch.dict("sys.modules", {"redis": fake_redis}):
                cache = LLMResponseCache.instance()
                assert cache.enabled is True
                assert cache.get("llmcache:abc") is None  # never raises


class TestHappyPath:
    """End-to-end behavior when Redis is reachable."""

    def test_set_then_get_round_trip(self) -> None:
        with patch.dict(os.environ, {"REDIS_URL": "redis://mock:6379/0"}):
            fake_redis = MagicMock()
            store: dict[str, str] = {}

            def _set(key: str, value: str, ex: int | None = None) -> bool:
                store[key] = value
                return True

            def _get(key: str) -> bytes | None:
                value = store.get(key)
                return value.encode("utf-8") if value is not None else None

            client = fake_redis.Redis.from_url.return_value
            client.ping.return_value = True
            client.get.side_effect = _get
            client.set.side_effect = _set

            with patch.dict("sys.modules", {"redis": fake_redis}):
                cache = LLMResponseCache.instance()
                key = LLMResponseCache.make_key({"role": "verifier", "claim_id": "c1"})
                assert cache.set(key, {"verdict": "supported", "confidence": 0.9}) is True
                assert cache.get(key) == {"verdict": "supported", "confidence": 0.9}

    def test_default_ttl_is_24_hours(self) -> None:
        assert DEFAULT_TTL_SECONDS == 24 * 60 * 60
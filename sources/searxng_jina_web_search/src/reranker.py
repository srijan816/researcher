# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable search-result reranking: NVIDIA NIM, FlashRank ONNX, local cross-encoder, or heuristic passthrough.

All entry points are fail-open. ``rerank_results`` returns ``None`` whenever a
backend is unavailable or errors, in which case callers keep the heuristic
ranking order they already computed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from urllib.request import Request
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_NVIDIA_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
DEFAULT_NVIDIA_RERANK_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
DEFAULT_LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_RERANK_TIMEOUT_SECONDS = 8.0
PASSAGE_MAX_CONTENT_CHARS = 1800

VALID_BACKENDS = {"nvidia", "flashrank", "local", "heuristic"}

_warned_keys: set[str] = set()
_local_cross_encoder = None
_local_import_failed = False
_flashrank_ranker = None
_flashrank_import_failed = False


def _warn_once(key: str, message: str, *args: object) -> None:
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    logger.warning(message, *args)


def resolve_rerank_backend() -> str:
    """Resolve the active rerank backend from environment configuration."""
    backend = (os.environ.get("AIQ_RERANK_BACKEND") or "").strip().lower()
    if backend in VALID_BACKENDS:
        return backend
    if backend:
        _warn_once(f"backend:{backend}", "Unknown AIQ_RERANK_BACKEND %r; using default resolution.", backend)
    if os.environ.get("NVIDIA_API_KEY"):
        return "nvidia"
    import importlib.util

    if importlib.util.find_spec("flashrank") is not None:
        return "flashrank"
    return "heuristic"


def build_passage(result: dict) -> str:
    """Build the rerank passage text: title plus the first ~1800 chars of content/snippet."""
    title = str(result.get("title") or "").strip()
    content = str(result.get("content") or result.get("snippet") or "").strip()
    passage = f"{title}\n{content[:PASSAGE_MAX_CONTENT_CHARS]}".strip()
    return passage or str(result.get("url") or "")


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]; degenerate spreads collapse to 0.5."""
    values = [float(score) for score in scores]
    if not values:
        return []
    lowest = min(values)
    highest = max(values)
    spread = highest - lowest
    if spread <= 1e-9:
        return [0.5 for _ in values]
    return [(value - lowest) / spread for value in values]


async def rerank_results(query: str, results: Sequence[dict], timeout: float | None = None) -> list[float] | None:
    """Score ``results`` against ``query``; return one raw score per result or None.

    Returning ``None`` means "no rerank available" and the caller must keep its
    existing (heuristic) ordering. This function never raises.
    """
    if not query or not results:
        return None
    backend = resolve_rerank_backend()
    if backend == "heuristic":
        return None
    if timeout is None:
        try:
            timeout = float(os.environ.get("AIQ_RERANK_TIMEOUT_SECONDS", DEFAULT_RERANK_TIMEOUT_SECONDS))
        except ValueError:
            timeout = DEFAULT_RERANK_TIMEOUT_SECONDS
    passages = [build_passage(result) for result in results]
    try:
        if backend == "nvidia":
            try:
                scores = await asyncio.wait_for(_rerank_nvidia(query, passages, timeout), timeout=timeout + 2.0)
            except Exception as exc:
                # Hosted rerank is rate-limited (free tier); degrade to the
                # local ONNX reranker rather than losing reranking entirely.
                _warn_once(
                    "rerank-nvidia-fallback",
                    "NVIDIA rerank failed (%s); falling back to local flashrank.",
                    exc,
                )
                scores = await asyncio.wait_for(_rerank_flashrank(query, passages), timeout=max(timeout, 120.0))
        elif backend == "flashrank":
            # First call downloads the ONNX model (~35MB) and loads it; allow
            # extra headroom for that, warm calls finish in well under 8s.
            scores = await asyncio.wait_for(_rerank_flashrank(query, passages), timeout=max(timeout, 120.0))
        else:
            scores = await asyncio.wait_for(_rerank_local(query, passages), timeout=max(timeout, 60.0))
    except Exception as exc:
        _warn_once(
            f"rerank-failed:{backend}",
            "Reranking via %s backend failed; keeping heuristic order: %s",
            backend,
            exc,
        )
        return None
    if scores is None or len(scores) != len(passages):
        return None
    return scores


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="replace"))


async def _rerank_nvidia(query: str, passages: list[str], timeout: float) -> list[float]:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")
    model = os.environ.get("AIQ_RERANK_MODEL", "").strip() or DEFAULT_NVIDIA_RERANK_MODEL
    endpoint = os.environ.get("AIQ_RERANK_ENDPOINT", "").strip() or DEFAULT_NVIDIA_RERANK_ENDPOINT
    payload = {
        "model": model,
        "query": {"text": query},
        "passages": [{"text": passage} for passage in passages],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _post_json, endpoint, payload, headers, timeout)
    rankings = data.get("rankings") if isinstance(data, dict) else None
    if not isinstance(rankings, list):
        raise RuntimeError("rerank response did not contain a rankings list")
    scores: list[float | None] = [None] * len(passages)
    seen_logits: list[float] = []
    for entry in rankings:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry["index"])
            logit = float(entry["logit"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < len(passages):
            scores[index] = logit
            seen_logits.append(logit)
    if not seen_logits:
        raise RuntimeError("rerank response contained no usable rankings")
    floor = min(seen_logits) - 1.0
    return [score if score is not None else floor for score in scores]


async def _rerank_flashrank(query: str, passages: list[str]) -> list[float] | None:
    """Score passages with a small ONNX cross-encoder via FlashRank (CPU, no API)."""
    global _flashrank_ranker, _flashrank_import_failed
    if _flashrank_import_failed:
        return None
    loop = asyncio.get_event_loop()
    if _flashrank_ranker is None:
        try:
            from flashrank import Ranker
        except ImportError:
            _flashrank_import_failed = True
            _warn_once(
                "flashrank-import",
                "flashrank is not installed; AIQ_RERANK_BACKEND=flashrank falls back to heuristic ranking.",
            )
            return None
        model_name = os.environ.get("AIQ_RERANK_FLASHRANK_MODEL", "").strip() or DEFAULT_FLASHRANK_MODEL
        cache_dir = os.environ.get("AIQ_RERANK_CACHE_DIR", "").strip() or "/tmp/flashrank"
        _flashrank_ranker = await loop.run_in_executor(
            None, lambda: Ranker(model_name=model_name, cache_dir=cache_dir)
        )

    def _score() -> list[float]:
        from flashrank import RerankRequest

        request = RerankRequest(
            query=query,
            passages=[{"id": index, "text": passage} for index, passage in enumerate(passages)],
        )
        ranked = _flashrank_ranker.rerank(request)
        scores: list[float] = [0.0] * len(passages)
        for entry in ranked:
            index = int(entry.get("id", -1))
            if 0 <= index < len(passages):
                scores[index] = float(entry.get("score", 0.0))
        return scores

    return await loop.run_in_executor(None, _score)


async def _rerank_local(query: str, passages: list[str]) -> list[float] | None:
    global _local_cross_encoder, _local_import_failed
    if _local_import_failed:
        return None
    loop = asyncio.get_event_loop()
    if _local_cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            _local_import_failed = True
            _warn_once(
                "local-import",
                "sentence_transformers is not installed; AIQ_RERANK_BACKEND=local falls back to heuristic ranking.",
            )
            return None
        model_name = os.environ.get("AIQ_RERANK_LOCAL_MODEL", "").strip() or DEFAULT_LOCAL_RERANK_MODEL
        _local_cross_encoder = await loop.run_in_executor(None, CrossEncoder, model_name)
    pairs = [(query, passage) for passage in passages]
    raw_scores = await loop.run_in_executor(None, _local_cross_encoder.predict, pairs)
    return [float(score) for score in raw_scores]

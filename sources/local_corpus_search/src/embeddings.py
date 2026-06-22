# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA NIM embedding client for the local scrape-artifact corpus.

Talks to an OpenAI-compatible embeddings endpoint (NVIDIA Integrate by
default). When ``NVIDIA_API_KEY`` is not configured the corpus falls back to a
pure-python BM25 lexical index, so this module must stay import-safe without a
key.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

DEFAULT_EMBED_ENDPOINT = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_BATCH_SIZE = 64

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """Raised when the embedding endpoint cannot produce vectors."""


def embed_endpoint() -> str:
    return os.environ.get("AIQ_CORPUS_EMBED_ENDPOINT", DEFAULT_EMBED_ENDPOINT)


def embed_model() -> str:
    return os.environ.get("AIQ_CORPUS_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def embed_timeout_seconds() -> float:
    raw = os.environ.get("AIQ_CORPUS_EMBED_TIMEOUT_SECONDS", "")
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        logger.warning("Invalid AIQ_CORPUS_EMBED_TIMEOUT_SECONDS=%r; using default", raw)
        return DEFAULT_TIMEOUT_SECONDS


class NimEmbeddingClient:
    """Minimal OpenAI-compatible embeddings client with batching and retries."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        batch_size: int = MAX_BATCH_SIZE,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        self.endpoint = endpoint or embed_endpoint()
        self.model = model or embed_model()
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else embed_timeout_seconds()
        self.batch_size = max(1, min(batch_size, MAX_BATCH_SIZE))
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = backoff_seconds
        if not self._api_key:
            raise EmbeddingError("NVIDIA_API_KEY is not configured; embeddings unavailable")

    def embed(self, texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
        """Embed ``texts`` and return one vector per input, preserving order."""
        if input_type not in ("passage", "query"):
            raise ValueError(f"input_type must be 'passage' or 'query', got {input_type!r}")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, input_type=input_type))
        return vectors

    def _embed_batch(self, batch: list[str], *, input_type: str) -> list[list[float]]:
        if not batch:
            return []
        payload = {
            "model": self.model,
            "input": batch,
            "input_type": input_type,
            "truncate": "END",
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
            try:
                response = httpx.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("Embedding request failed (attempt %d): %s", attempt + 1, exc)
                continue
            if response.status_code in _RETRYABLE_STATUS:
                last_error = EmbeddingError(f"Embedding endpoint returned HTTP {response.status_code}")
                logger.warning(
                    "Embedding endpoint returned HTTP %d (attempt %d)", response.status_code, attempt + 1
                )
                continue
            if response.status_code != 200:
                raise EmbeddingError(
                    f"Embedding endpoint returned HTTP {response.status_code}: {response.text[:300]}"
                )
            try:
                data = response.json()["data"]
                rows = sorted(data, key=lambda item: item.get("index", 0))
                vectors = [list(map(float, row["embedding"])) for row in rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingError(f"Malformed embedding response: {exc}") from exc
            if len(vectors) != len(batch):
                raise EmbeddingError(f"Embedding count mismatch: sent {len(batch)}, got {len(vectors)}")
            return vectors
        raise EmbeddingError(f"Embedding request failed after {self.max_retries + 1} attempts: {last_error}")


def get_default_embedder() -> NimEmbeddingClient | None:
    """Return a NIM embedding client, or None when no NVIDIA_API_KEY is set."""
    if not os.environ.get("NVIDIA_API_KEY", "").strip():
        return None
    try:
        return NimEmbeddingClient()
    except EmbeddingError:
        return None

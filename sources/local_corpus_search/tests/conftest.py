# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test fixtures for local_corpus_search.

Makes the package importable even when it has not been installed into the
environment yet (e.g. before the integrator adds it to the workspace dev
dependency group).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _ensure_package_importable() -> None:
    if "local_corpus_search" in sys.modules:
        return
    try:
        import local_corpus_search  # noqa: F401

        return
    except ImportError:
        pass
    package_root = Path(__file__).resolve().parent.parent / "src"
    spec = importlib.util.spec_from_file_location(
        "local_corpus_search",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_corpus_search"] = module
    spec.loader.exec_module(module)


_ensure_package_importable()


@pytest.fixture(autouse=True)
def _no_nvidia_key(monkeypatch: pytest.MonkeyPatch):
    """Keep tests hermetic: never pick up a real NVIDIA_API_KEY or env overrides."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("AIQ_CORPUS_DB", raising=False)
    monkeypatch.delenv("AIQ_CORPUS_EMBED_ENDPOINT", raising=False)
    monkeypatch.delenv("AIQ_CORPUS_EMBED_MODEL", raising=False)
    monkeypatch.delenv("AIQ_CORPUS_EMBED_TIMEOUT_SECONDS", raising=False)


def write_artifact(
    root: Path,
    *,
    job_id: str = "job-1",
    url: str = "https://example.com/page",
    title: str = "Example Page",
    content: str = "",
    extraction_status: str = "extracted",
    source_class: str = "news",
    created_at: str = "2026-06-01T00:00:00+00:00",
) -> dict[str, Any]:
    """Write a gzip JSON scrape artifact matching scrape_artifacts.py output."""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    host = url.split("//", 1)[-1].split("/", 1)[0].replace(".", "_")
    payload = {
        "job_id": job_id,
        "url": url,
        "title": title,
        "extraction_status": extraction_status,
        "content_hash": content_hash,
        "researcher": "researcher-1",
        "tool": "web_search_tool",
        "source_class": source_class,
        "content": content,
        "content_length": len(content),
        "created_at": created_at,
    }
    job_dir = root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"{host}-{url_hash}-{content_hash[:16]}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False)
    return payload


def make_words(topic_words: list[str], total_words: int = 200) -> str:
    """Build deterministic content with at least ``total_words`` words, sentence-shaped."""
    filler = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    words: list[str] = []
    index = 0
    while len(words) < total_words:
        words.extend(topic_words)
        words.append(filler[index % len(filler)])
        index += 1
        if index % 12 == 0:
            words[-1] += "."
    return " ".join(words) + "."


class FakeEmbedder:
    """Deterministic keyword-presence embedder for tests (no network)."""

    model = "fake-test-embedder"

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary
        self.calls: list[tuple[int, str]] = []

    def embed(self, texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
        self.calls.append((len(texts), input_type))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(term)) for term in self.vocabulary]
            vector.append(1.0)  # avoid zero vectors
            vectors.append(vector)
        return vectors


@pytest.fixture
def artifact_writer():
    """Callable that writes scrape artifacts shaped like scrape_artifacts.py output."""
    return write_artifact


@pytest.fixture
def words_maker():
    """Callable that builds deterministic >=N word content around topic words."""
    return make_words


@pytest.fixture
def fake_embedder_factory():
    """Factory for deterministic no-network embedders."""
    return FakeEmbedder

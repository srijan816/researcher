# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NAT tool registration config."""

from __future__ import annotations

import pytest

pytest.importorskip("nat", reason="NVIDIA NAT is required for registration tests")

from local_corpus_search.register import LocalCorpusSearchToolConfig  # noqa: E402


def test_config_defaults() -> None:
    config = LocalCorpusSearchToolConfig()
    assert config.db_path == "./data/corpus_index.db"
    assert config.top_k == 6
    assert config.max_snippet_chars == 1500


def test_config_db_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIQ_CORPUS_DB", "/tmp/custom_corpus.db")
    config = LocalCorpusSearchToolConfig()
    assert config.db_path == "/tmp/custom_corpus.db"


def test_config_explicit_values() -> None:
    config = LocalCorpusSearchToolConfig(db_path="/data/corpus.db", top_k=3, max_snippet_chars=900)
    assert config.db_path == "/data/corpus.db"
    assert config.top_k == 3
    assert config.max_snippet_chars == 900


def test_config_rejects_out_of_range_top_k() -> None:
    with pytest.raises(ValueError):
        LocalCorpusSearchToolConfig(top_k=0)

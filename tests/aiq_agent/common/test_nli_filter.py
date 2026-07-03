# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local ONNX NLI pre-filter. No network, no real model."""

import numpy as np
import pytest

from aiq_agent.common import nli_filter
from aiq_agent.common.nli_filter import DEFAULT_NLI_MODEL
from aiq_agent.common.nli_filter import nli_enabled
from aiq_agent.common.nli_filter import nli_model_repo
from aiq_agent.common.nli_filter import score_entailment
from aiq_agent.common.nli_filter import score_entailment_batch

_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}


class _FakeEncoding:
    def __init__(self, length: int = 4):
        self.ids = [1] * length
        self.attention_mask = [1] * length
        self.type_ids = [0] * length


class _FakeTokenizer:
    def __init__(self):
        self.batches: list[list[tuple[str, str]]] = []

    def encode_batch(self, pairs):
        self.batches.append(list(pairs))
        return [_FakeEncoding() for _ in pairs]


class _FakeSession:
    """Mock ONNX session returning queued logits rows (or raising)."""

    def __init__(self, logits_rows, input_names=("input_ids", "attention_mask")):
        self._logits_rows = list(logits_rows)
        self._input_names = input_names
        self.feeds: list[dict] = []

    def run(self, _output_names, feed):
        self.feeds.append(feed)
        batch_size = feed["input_ids"].shape[0]
        rows = [self._logits_rows.pop(0) for _ in range(batch_size)]
        for row in rows:
            if isinstance(row, Exception):
                raise row
        return [np.asarray(rows, dtype=np.float32)]


def _install_runtime(monkeypatch, session, tokenizer=None, id2label=None):
    runtime = nli_filter._NliRuntime(
        tokenizer=tokenizer or _FakeTokenizer(),
        session=session,
        input_names=frozenset(session._input_names),
        id2label=id2label or dict(_ID2LABEL),
    )
    monkeypatch.setattr(nli_filter, "_get_runtime", lambda: runtime)
    return runtime


@pytest.fixture(autouse=True)
def _clean_runtime_cache(monkeypatch):
    monkeypatch.delenv("AIQ_NLI_ENABLED", raising=False)
    monkeypatch.delenv("AIQ_NLI_MODEL", raising=False)
    nli_filter.reset_runtime()
    yield
    nli_filter.reset_runtime()


class TestEnvKnobs:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AIQ_NLI_ENABLED", raising=False)
        assert nli_enabled() is True

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AIQ_NLI_ENABLED", "0")
        assert nli_enabled() is False

    def test_model_repo_default_and_override(self, monkeypatch):
        monkeypatch.delenv("AIQ_NLI_MODEL", raising=False)
        assert nli_model_repo() == DEFAULT_NLI_MODEL
        monkeypatch.setenv("AIQ_NLI_MODEL", "someorg/some-nli-onnx")
        assert nli_model_repo() == "someorg/some-nli-onnx"


class TestScoring:
    def test_softmax_probabilities_use_config_label_order(self, monkeypatch):
        # Logit peak at index 1, which config.json maps to "entailment".
        _install_runtime(monkeypatch, _FakeSession([[0.0, 5.0, 0.0]]))
        result = score_entailment("evidence text", "claim text")
        assert result is not None
        assert set(result) == {"entailment", "neutral", "contradiction"}
        assert result["entailment"] > 0.9
        assert result["entailment"] > result["neutral"] > 0
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_permuted_id2label_respected(self, monkeypatch):
        # Same logits, different label mapping: index 1 is now contradiction.
        _install_runtime(
            monkeypatch,
            _FakeSession([[0.0, 5.0, 0.0]]),
            id2label={0: "entailment", 1: "contradiction", 2: "neutral"},
        )
        result = score_entailment("evidence text", "claim text")
        assert result["contradiction"] > 0.9

    def test_batch_returns_one_score_per_pair(self, monkeypatch):
        _install_runtime(monkeypatch, _FakeSession([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
        results = score_entailment_batch([("e1", "c1"), ("e2", "c2")])
        assert len(results) == 2
        assert results[0]["contradiction"] > 0.9
        assert results[1]["neutral"] > 0.9

    def test_blank_pairs_score_none_without_inference(self, monkeypatch):
        session = _FakeSession([[0.0, 5.0, 0.0]])
        _install_runtime(monkeypatch, session)
        results = score_entailment_batch([("", "claim"), ("evidence", ""), ("evidence", "claim")])
        assert results[0] is None
        assert results[1] is None
        assert results[2] is not None
        assert len(session.feeds) == 1
        assert session.feeds[0]["input_ids"].shape[0] == 1

    def test_large_batches_are_chunked(self, monkeypatch):
        count = nli_filter._MAX_BATCH_SIZE + 4
        session = _FakeSession([[0.0, 5.0, 0.0]] * count)
        _install_runtime(monkeypatch, session)
        results = score_entailment_batch([(f"e{i}", f"c{i}") for i in range(count)])
        assert all(score is not None for score in results)
        assert len(session.feeds) == 2

    def test_token_type_ids_fed_when_model_expects_them(self, monkeypatch):
        session = _FakeSession([[0.0, 5.0, 0.0]], input_names=("input_ids", "attention_mask", "token_type_ids"))
        _install_runtime(monkeypatch, session)
        assert score_entailment("evidence", "claim") is not None
        assert "token_type_ids" in session.feeds[0]


class TestFailOpen:
    def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("AIQ_NLI_ENABLED", "0")
        _install_runtime(monkeypatch, _FakeSession([[0.0, 5.0, 0.0]]))
        assert score_entailment("evidence", "claim") is None
        assert score_entailment_batch([("evidence", "claim")]) == [None]

    def test_inference_error_returns_none(self, monkeypatch):
        _install_runtime(monkeypatch, _FakeSession([RuntimeError("onnx boom")]))
        assert score_entailment("evidence", "claim") is None

    def test_load_failure_returns_none_and_is_not_retried(self, monkeypatch):
        calls = {"count": 0}

        def _boom(repo):
            calls["count"] += 1
            raise RuntimeError("download failed")

        monkeypatch.setattr(nli_filter, "_load_runtime", _boom)
        assert score_entailment("evidence", "claim") is None
        assert score_entailment("evidence", "claim") is None
        assert calls["count"] == 1

    def test_empty_input(self):
        assert score_entailment_batch([]) == []


class TestConfigParsing:
    def test_parse_id2label_lowercases_and_casts(self):
        parsed = nli_filter._parse_id2label(
            {"id2label": {"0": "CONTRADICTION", "1": "Entailment", "2": "neutral"}}
        )
        assert parsed == {0: "contradiction", 1: "entailment", 2: "neutral"}

    def test_parse_id2label_rejects_non_nli_heads(self):
        with pytest.raises(ValueError):
            nli_filter._parse_id2label({"id2label": {"0": "positive", "1": "negative"}})
        with pytest.raises(ValueError):
            nli_filter._parse_id2label({})

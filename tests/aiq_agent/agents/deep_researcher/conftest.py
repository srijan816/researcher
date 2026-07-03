# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for deep-researcher agent tests."""

import pytest


@pytest.fixture(autouse=True)
def _disable_nli_prefilter(monkeypatch):
    """Keep verifier tests hermetic: the NLI pre-filter is on by default in
    production and would otherwise download a real ONNX model from Hugging
    Face during tests. Tests that exercise the cascade opt back in and mock
    the scorer at the ``nli_filter`` module boundary."""
    monkeypatch.setenv("AIQ_NLI_ENABLED", "0")

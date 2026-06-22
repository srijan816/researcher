# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prompt-inspection routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiq_api.routes.prompts import register_prompt_routes


def test_prompts_endpoint_lists_research_templates() -> None:
    app = FastAPI()
    register_prompt_routes(app)
    client = TestClient(app)

    response = client.get("/prompts")

    assert response.status_code == 200
    data = response.json()
    prompt_ids = {prompt["id"] for prompt in data["prompts"]}
    assert data["missing"] == []
    assert "deep_researcher.orchestrator" in prompt_ids
    assert "deep_researcher.planner" in prompt_ids
    assert "deep_researcher.researcher" in prompt_ids
    assert "clarifier.plan_generation" in prompt_ids
    assert all(prompt["template"].strip() for prompt in data["prompts"])


def test_v1_prompts_endpoint_is_alias() -> None:
    app = FastAPI()
    register_prompt_routes(app)
    client = TestClient(app)

    response = client.get("/v1/prompts")

    assert response.status_code == 200
    assert response.json()["count"] == len(response.json()["prompts"])

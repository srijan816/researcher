# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Endpoint tests for the /v1/jobs/queue research queue API."""

import httpx
import pytest
from fastapi import FastAPI

from aiq_api.routes.queue import register_queue_routes


@pytest.fixture
async def client(tmp_path, monkeypatch):
    # No Dask scheduler in tests: routes register, background scheduler does not start.
    monkeypatch.delenv("NAT_DASK_SCHEDULER_ADDRESS", raising=False)
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)

    app = FastAPI()
    await register_queue_routes(app, f"sqlite:///{tmp_path}/queue.db")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client


async def test_add_list_approve_cancel_delete_flow(client):
    created = await client.post(
        "/v1/jobs/queue",
        json={
            "agent_type": "deep_researcher",
            "input": "Track EU AI Act enforcement weekly.",
            "research_depth": "deeper",
            "priority": 3,
            "auto_approve": False,
        },
    )
    assert created.status_code == 200, created.text
    item = created.json()
    assert item["status"] == "queued"
    assert item["priority"] == 3
    item_id = item["id"]

    listed = await client.get("/v1/jobs/queue", params={"status": "queued"})
    assert listed.status_code == 200
    assert [entry["id"] for entry in listed.json()["items"]] == [item_id]

    approved = await client.post(f"/v1/jobs/queue/{item_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    cancelled = await client.post(f"/v1/jobs/queue/{item_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    deleted = await client.delete(f"/v1/jobs/queue/{item_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": item_id, "deleted": True}

    empty = await client.get("/v1/jobs/queue")
    assert empty.json()["items"] == []


async def test_auto_approve_defaults_to_approved(client):
    created = await client.post(
        "/v1/jobs/queue",
        json={"agent_type": "deep_researcher", "input": "One-shot research task."},
    )
    assert created.status_code == 200
    assert created.json()["status"] == "approved"


async def test_unknown_agent_type_rejected(client):
    created = await client.post(
        "/v1/jobs/queue",
        json={"agent_type": "not_a_real_agent", "input": "whatever"},
    )
    assert created.status_code == 400


async def test_unknown_status_filter_rejected(client):
    response = await client.get("/v1/jobs/queue", params={"status": "nonsense"})
    assert response.status_code == 400


async def test_secret_is_never_echoed(client):
    created = await client.post(
        "/v1/jobs/queue",
        json={
            "agent_type": "deep_researcher",
            "input": "Webhook research item",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "super-secret-value",  # pragma: allowlist secret
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["has_webhook_secret"] is True
    assert "super-secret-value" not in created.text


async def test_missing_item_is_404(client):
    assert (await client.post("/v1/jobs/queue/424242/approve")).status_code == 404
    assert (await client.post("/v1/jobs/queue/424242/cancel")).status_code == 404
    assert (await client.delete("/v1/jobs/queue/424242")).status_code == 404

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for claim-table fragments in the DeepAgents virtual filesystem."""

from __future__ import annotations

import fnmatch
import json
import re
from typing import Any

from .claim_table import AtomicClaim
from .claim_table import ClaimProfile
from .claim_table import ClaimTable
from .claim_table import merge_claim_tables_json

CLAIMS_DIR = "/shared/claims"
CLAIM_TABLE_PATH = "/shared/claim_table.json"


async def write_researcher_claims(task_id: str, claims: list[AtomicClaim], filesystem: Any) -> str:
    """Write a researcher-specific claim fragment and return the virtual path."""
    safe_task_id = _safe_task_id(task_id)
    path = f"{CLAIMS_DIR}/claims_{safe_task_id}.json"
    payload = {
        "atomic_claims": [claim.model_dump(mode="json") for claim in claims],
    }
    await _write_file(filesystem, path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


async def merge_claim_files(
    filesystem: Any,
    profile: ClaimProfile | None = None,
    job_id: str | None = None,
) -> ClaimTable:
    """Merge all per-researcher claim fragments into `/shared/claim_table.json`."""
    paths = await _list_claim_paths(filesystem)
    contents = [await _read_file(filesystem, path) for path in paths]
    table, errors = merge_claim_tables_json([content for content in contents if content])
    if table is None:
        table = ClaimTable()
    if profile is not None:
        table.profile = profile
        existing_ids = {claim.claim_id for claim in table.atomic_claims}
        for claim in profile.claims:
            if claim.claim_id not in existing_ids:
                table.atomic_claims.append(claim)
    if job_id:
        table.job_id = job_id
    if errors:
        table.model_extra["merge_errors"] = errors
    await write_claim_table(job_id or table.job_id or "", table, filesystem)
    return table


async def write_claim_table(job_id: str, table: ClaimTable, filesystem: Any) -> str:
    """Write the merged claim table to the canonical virtual path."""
    table.job_id = job_id or table.job_id
    await _write_file(filesystem, CLAIM_TABLE_PATH, table.model_dump_json(indent=2))
    return CLAIM_TABLE_PATH


async def _list_claim_paths(filesystem: Any) -> list[str]:
    if hasattr(filesystem, "list_files"):
        result = filesystem.list_files(CLAIMS_DIR)
        if hasattr(result, "__await__"):
            result = await result
        return sorted(path for path in result if fnmatch.fnmatch(path, f"{CLAIMS_DIR}/claims_*.json"))
    files = getattr(filesystem, "files", None)
    if isinstance(files, dict):
        return sorted(path for path in files if fnmatch.fnmatch(path, f"{CLAIMS_DIR}/claims_*.json"))
    return []


async def _write_file(filesystem: Any, path: str, content: str) -> None:
    if hasattr(filesystem, "write_file"):
        result = filesystem.write_file(path, content)
        if hasattr(result, "__await__"):
            await result
        return
    files = getattr(filesystem, "files", None)
    if isinstance(files, dict):
        files[path] = content
        return
    raise TypeError("filesystem must provide write_file() or a dict-like files attribute")


async def _read_file(filesystem: Any, path: str) -> str:
    if hasattr(filesystem, "read_file"):
        result = filesystem.read_file(path)
        if hasattr(result, "__await__"):
            result = await result
        return str(result)
    files = getattr(filesystem, "files", None)
    if isinstance(files, dict):
        return str(files.get(path, ""))
    raise TypeError("filesystem must provide read_file() or a dict-like files attribute")


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")[:80] or "researcher"

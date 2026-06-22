# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded gap-fill round policy for the deep research orchestrator.

Gap-fill used to be a single pass. The orchestrator may now launch up to N
targeted gap-fill researcher rounds, where N is derived from the research tier
and may be overridden with ``AIQ_GAPFILL_MAX_ROUNDS``. Gap-fill spends only
the adaptive reserve budget — total search budgets are never raised here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aiq_agent.common.research_depth import normalize_research_depth

logger = logging.getLogger(__name__)

GAPFILL_ROUNDS_BY_TIER: dict[str, int] = {
    "shallow": 0,
    "medium": 1,
    "deeper": 2,
    "deep": 3,
}

_MAX_GAPFILL_ROUNDS = 10


def gapfill_max_rounds(tier: Any) -> int:
    """Return allowed targeted gap-fill rounds for a research tier.

    Defaults: shallow 0, medium 1, deeper 2, deep 3. The environment variable
    ``AIQ_GAPFILL_MAX_ROUNDS`` overrides the tier default for every tier.
    Invalid overrides fall back to the tier default (fail-open).
    """
    normalized = normalize_research_depth(tier)
    default_rounds = GAPFILL_ROUNDS_BY_TIER.get(normalized, 1)
    raw = os.getenv("AIQ_GAPFILL_MAX_ROUNDS")
    if raw is None or not raw.strip():
        return default_rounds
    try:
        return max(0, min(_MAX_GAPFILL_ROUNDS, int(raw.strip())))
    except (TypeError, ValueError):
        logger.debug("Invalid AIQ_GAPFILL_MAX_ROUNDS=%r; using tier default %d", raw, default_rounds)
        return default_rounds


def gapfill_snapshot(
    *,
    tier: Any,
    rounds_used: int,
    budget_snapshot: dict[str, Any] | None = None,
    budget_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the gap-fill progress block for ``get_research_progress_snapshot``.

    Reports rounds used vs allowed and the remaining reserve search budget so
    the orchestrator can decide deterministically whether another targeted
    gap-fill researcher task is permitted.
    """
    rounds_allowed = gapfill_max_rounds(tier)
    rounds_used = max(0, int(rounds_used or 0))

    search_remaining: int | None = None
    for entry in (budget_snapshot or {}).get("budgets") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("key")) == "search" and entry.get("remaining") is not None:
            try:
                search_remaining = max(0, int(entry["remaining"]))
            except (TypeError, ValueError):
                search_remaining = None
            break

    reserve_total: int | None = None
    if budget_profile:
        try:
            reserve_total = max(0, int(budget_profile.get("reserve_search_calls") or 0))
        except (TypeError, ValueError):
            reserve_total = None

    reserve_remaining: int | None = None
    if search_remaining is not None and reserve_total is not None:
        reserve_remaining = min(reserve_total, search_remaining)
    elif search_remaining is not None:
        reserve_remaining = search_remaining

    rounds_remaining = max(0, rounds_allowed - rounds_used)
    may_launch = rounds_remaining > 0 and (reserve_remaining is None or reserve_remaining > 0)
    return {
        "rounds_used": rounds_used,
        "rounds_allowed": rounds_allowed,
        "rounds_remaining": rounds_remaining,
        "reserve_search_calls_total": reserve_total,
        "reserve_search_calls_remaining": reserve_remaining,
        "search_calls_remaining": search_remaining,
        "may_launch_gap_fill": may_launch,
        "policy": (
            "Gap-fill rounds spend only the adaptive reserve. Stop when rounds_remaining is 0 "
            "or the reserve is exhausted, then move to synthesis."
        ),
    }

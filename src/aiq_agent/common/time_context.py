# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for grounding model prompts in the current local date/time."""

from __future__ import annotations

from datetime import datetime


def current_datetime_context() -> str:
    """Return a prompt-friendly local timestamp with timezone when available."""
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    timezone_name = now.tzname()
    if timezone_name:
        return f"{timestamp} {timezone_name}"
    return timestamp

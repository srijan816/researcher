# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave 3 W3.2 — section-aware final-report streaming.

Tests the section splitter and the cumulative emission shape. The full
AgentEventCallback is a thin emitter on top of an EventStore, so we
exercise just the static splitter helper to keep these tests fast and
free of database dependencies.
"""

from __future__ import annotations

from aiq_api.jobs.callbacks import AgentEventCallback


class TestSplitReportIntoSections:
    def test_no_headings_returns_whole(self) -> None:
        text = "Just one paragraph.\nWith a line break."
        assert AgentEventCallback._split_report_into_sections(text) == [text]

    def test_two_h2_sections(self) -> None:
        text = (
            "# Title\n\nIntro.\n\n"
            "## First\n\nOne.\n\n"
            "## Second\n\nTwo.\n"
        )
        chunks = AgentEventCallback._split_report_into_sections(text)
        assert len(chunks) == 3
        assert chunks[0].startswith("# Title")
        assert chunks[1].startswith("## First")
        assert chunks[2].startswith("## Second")
        # Cumulative join should reconstruct the original
        assert "".join(chunks) == text

    def test_prelude_before_first_heading(self) -> None:
        text = "Some intro prose.\n\n## Section A\n\nBody.\n"
        chunks = AgentEventCallback._split_report_into_sections(text)
        assert len(chunks) == 2
        assert "Some intro prose" in chunks[0]
        assert "## Section A" in chunks[1]
        assert "".join(chunks) == text

    def test_h1_h2_h3_all_supported(self) -> None:
        text = "## A\n\nA body.\n\n### A.1\n\nNested.\n\n## B\n\nB body.\n"
        chunks = AgentEventCallback._split_report_into_sections(text)
        # 3 chunks: A (with A.1 nested inside), B
        assert len(chunks) == 2
        assert "## A" in chunks[0] and "### A.1" in chunks[0]
        assert "## B" in chunks[1]
        assert "".join(chunks) == text

    def test_empty_returns_empty_list(self) -> None:
        assert AgentEventCallback._split_report_into_sections("") == []

    def test_cumulative_join_is_lossless(self) -> None:
        text = (
            "# Title\n\n"
            "Intro paragraph.\n\n"
            "## Section One\n\nBody one.\n\n"
            "## Section Two\n\nBody two.\n\n"
            "## Section Three\n\nBody three.\n"
        )
        chunks = AgentEventCallback._split_report_into_sections(text)
        assert "".join(chunks) == text, "round-trip must be lossless"

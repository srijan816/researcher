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

"""Tests for markdown -> DOCX report export."""

import pytest

from aiq_api.jobs.docx_export import parse_inline
from aiq_api.jobs.docx_export import parse_markdown_blocks
from aiq_api.jobs.docx_export import plain_text
from aiq_api.jobs.docx_export import render_links_as_text

SAMPLE_MARKDOWN = """# Research Report

Intro paragraph with **bold**, *italic*, and `code` plus a [link](https://example.com).

## Findings

- First bullet
- Second bullet

1. Numbered one
2. Numbered two

---

### Details

Some closing paragraph
spread over two lines.

```
raw code block
```
"""


class TestMarkdownParsing:
    def test_heading_levels(self):
        blocks = parse_markdown_blocks("# One\n## Two\n### Three")
        assert blocks == [("heading", 1, "One"), ("heading", 2, "Two"), ("heading", 3, "Three")]

    def test_paragraph_joins_wrapped_lines(self):
        blocks = parse_markdown_blocks("line one\nline two\n\nnext para")
        assert blocks == [("paragraph", "line one line two"), ("paragraph", "next para")]

    def test_bullet_and_numbered_lists(self):
        blocks = parse_markdown_blocks("- a\n* b\n+ c\n1. one\n2) two")
        assert blocks == [
            ("bullet", "a"),
            ("bullet", "b"),
            ("bullet", "c"),
            ("number", "one"),
            ("number", "two"),
        ]

    def test_horizontal_rule_and_code_fence(self):
        blocks = parse_markdown_blocks("---\n```\ncode here\n```")
        assert blocks == [("hr",), ("code", "code here")]

    def test_full_sample_structure(self):
        kinds = [block[0] for block in parse_markdown_blocks(SAMPLE_MARKDOWN)]
        assert kinds == [
            "heading",
            "paragraph",
            "heading",
            "bullet",
            "bullet",
            "number",
            "number",
            "hr",
            "heading",
            "paragraph",
            "code",
        ]


class TestInlineParsing:
    def test_bold_italic_code_runs(self):
        runs = parse_inline("plain **bold** *ital* `code` end")
        assert runs == [
            ("text", "plain "),
            ("bold", "bold"),
            ("text", " "),
            ("italic", "ital"),
            ("text", " "),
            ("code", "code"),
            ("text", " end"),
        ]

    def test_links_render_as_text_with_url(self):
        assert render_links_as_text("see [docs](https://example.com/x)") == "see docs (https://example.com/x)"

    def test_plain_text_strips_markers(self):
        assert plain_text("**Bold** and [link](https://e.co)") == "Bold and link (https://e.co)"


class TestDocxRendering:
    def test_markdown_to_docx_bytes_roundtrip(self):
        docx = pytest.importorskip("docx", reason="python-docx not installed")
        import io

        from aiq_api.jobs.docx_export import markdown_to_docx_bytes

        payload = markdown_to_docx_bytes(SAMPLE_MARKDOWN, title="Research Report test")
        assert isinstance(payload, bytes)
        assert payload[:2] == b"PK"  # OOXML zip container

        document = docx.Document(io.BytesIO(payload))
        texts = [paragraph.text for paragraph in document.paragraphs]

        assert "Research Report" in texts
        assert "Findings" in texts
        assert "First bullet" in texts
        assert "Numbered one" in texts
        # Links are flattened to "text (url)".
        assert any("link (https://example.com)" in text for text in texts)

        styles = {paragraph.text: paragraph.style.name for paragraph in document.paragraphs}
        assert styles["Research Report"] == "Heading 1"
        assert styles["Findings"] == "Heading 2"
        assert styles["First bullet"] == "List Bullet"
        assert styles["Numbered one"] == "List Number"

    def test_bold_runs_survive_render(self):
        docx = pytest.importorskip("docx", reason="python-docx not installed")
        import io

        from aiq_api.jobs.docx_export import markdown_to_docx_bytes

        payload = markdown_to_docx_bytes("Body with **important** text.")
        document = docx.Document(io.BytesIO(payload))
        runs = [run for paragraph in document.paragraphs for run in paragraph.runs]
        bold_runs = [run.text for run in runs if run.bold]
        assert bold_runs == ["important"]

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

from aiq_agent.common.scrape_artifacts import persist_scrape_artifacts


def test_persist_scrape_artifacts_stores_compressed_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("AIQ_SCRAPE_ARTIFACT_DIR", str(tmp_path))
    output = """
    <document idx="0">
      <title>Example Result</title>
      <url>https://example.com/report</url>
      <content>Full extracted page text.</content>
    </document>
    """

    artifacts = persist_scrape_artifacts(
        job_id="job-1",
        tool_name="advanced_web_search_tool",
        tool_output=output,
        researcher="researcher-agent",
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.url == "https://example.com/report"
    assert artifact.title == "Example Result"
    assert artifact.extraction_status == "extracted"
    assert artifact.researcher == "researcher-agent"

    with gzip.open(artifact.artifact_path, "rt", encoding="utf-8") as fp:
        payload = json.load(fp)
    assert payload["url"] == "https://example.com/report"
    assert payload["content"] == "Full extracted page text."
    assert payload["content_hash"] == artifact.content_hash

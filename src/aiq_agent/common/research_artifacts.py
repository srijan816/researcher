# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable audit artifacts for deep research runs.

DeepAgents uses an in-memory virtual filesystem for live work. This module
mirrors compact, synthesis-ready artifacts to deterministic markdown/JSON files
so a run can be inspected, resumed, and debugged without copying full scraped
pages multiple times.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field

from .claim_table import ClaimTable
from .claim_table import validate_claim_table_json
from .evidence_packet import EvidencePacket
from .source_scoring import SourceScore
from .source_scoring import score_source

_DEFAULT_RUNS_DIR = Path(".deep-research-runtime/research_runs")
_MAX_NOTE_CHARS = 12000
_MAX_SOURCE_SUMMARY_EXTRACTS = 5


class ResearchRunManifest(BaseModel):
    """Manifest describing the durable files written for one research run."""

    schema_version: str = "1.0"
    job_id: str | None = None
    slug: str
    run_dir: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    files: dict[str, str] = Field(default_factory=dict)
    source_count: int = 0
    claim_count: int = 0
    gap_count: int = 0
    contradiction_count: int = 0


def coerce_file_content(value: Any) -> str:
    """Convert DeepAgents file data into text."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(line) for line in value)
    if isinstance(value, dict):
        return coerce_file_content(value.get("content", ""))
    return ""


def build_virtual_research_artifacts(
    *,
    files: dict[str, Any],
    registry_sources: list[Any] | None = None,
    job_id: str | None = None,
    request_text: str = "",
    final_report: str | None = None,
) -> dict[str, str]:
    """Build compact artifacts suitable for writing back to `/shared`."""

    registry_sources = registry_sources or []
    plan = _read_json_file(files, "shared/plan.json")
    claim_table_content = _read_text_file(files, "shared/claim_table.json")
    evidence_packet = _read_evidence_packet(files)
    claim_table = _read_claim_table(claim_table_content)
    source_scores = _build_source_scores(
        evidence_packet=evidence_packet,
        registry_sources=registry_sources,
        request_text=request_text,
    )
    gaps = _build_gaps_markdown(plan=plan, claim_table=claim_table, evidence_packet=evidence_packet)
    contradictions = _build_contradictions_markdown(evidence_packet=evidence_packet, claim_table=claim_table)
    research = _build_research_markdown(
        request_text=request_text,
        plan=plan,
        claim_table=claim_table,
        evidence_packet=evidence_packet,
        source_scores=source_scores,
        gaps_md=gaps,
        contradictions_md=contradictions,
    )
    sources_json = json.dumps([score.model_dump() for score in source_scores], indent=2, ensure_ascii=False)
    return {
        "/shared/sources.json": sources_json,
        "/shared/gaps.md": gaps,
        "/shared/contradictions.md": contradictions,
        "/shared/research.md": research,
        **({"/shared/final.md": final_report} if final_report else {}),
    }


def mirror_run_artifacts(
    *,
    files: dict[str, Any],
    registry_sources: list[Any] | None = None,
    job_id: str | None = None,
    request_text: str = "",
    final_report: str | None = None,
    runs_dir: Path | None = None,
) -> ResearchRunManifest:
    """Write the durable per-run audit folder and return its manifest."""

    registry_sources = registry_sources or []
    root = runs_dir or Path(os.environ.get("AIQ_RESEARCH_RUNS_DIR", str(_DEFAULT_RUNS_DIR)))
    slug = _slugify(request_text or job_id or "research-run")
    run_name = f"{job_id or 'local'}-{slug}"[:160]
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "notes").mkdir(exist_ok=True)
    (run_dir / "source_summaries").mkdir(exist_ok=True)

    plan = _read_json_file(files, "shared/plan.json")
    claim_table_content = _read_text_file(files, "shared/claim_table.json")
    evidence_packet = _read_evidence_packet(files)
    claim_table = _read_claim_table(claim_table_content)
    source_scores = _build_source_scores(
        evidence_packet=evidence_packet,
        registry_sources=registry_sources,
        request_text=request_text,
    )
    virtual_artifacts = build_virtual_research_artifacts(
        files=files,
        registry_sources=registry_sources,
        job_id=job_id,
        request_text=request_text,
        final_report=final_report,
    )

    written: dict[str, str] = {}
    _write(run_dir / "plan.json", json.dumps(plan or {}, indent=2, ensure_ascii=False), written, run_dir)
    _write(run_dir / "plan.md", _plan_markdown(plan, request_text=request_text), written, run_dir)
    queries_json = json.dumps((plan or {}).get("queries", []), indent=2, ensure_ascii=False)
    _write(run_dir / "queries.json", queries_json, written, run_dir)
    budget_payload = (plan or {}).get("budget_profile") or (plan or {}).get("task_analysis", {}).get(
        "budget_profile", {}
    )
    _write(
        run_dir / "budget.json",
        json.dumps(budget_payload, indent=2, ensure_ascii=False),
        written,
        run_dir,
    )
    _write(run_dir / "sources.json", virtual_artifacts["/shared/sources.json"], written, run_dir)
    _write(run_dir / "claims.json", claim_table_content or "{}", written, run_dir)
    _write(run_dir / "gaps.md", virtual_artifacts["/shared/gaps.md"], written, run_dir)
    _write(run_dir / "contradictions.md", virtual_artifacts["/shared/contradictions.md"], written, run_dir)
    _write(run_dir / "research.md", virtual_artifacts["/shared/research.md"], written, run_dir)
    if final_report:
        _write(run_dir / "final.md", final_report, written, run_dir)

    for path, value in sorted(files.items()):
        normalized = str(path).lstrip("/")
        if not _is_note_path(normalized):
            continue
        content = coerce_file_content(value).strip()
        if len(content) < 120:
            continue
        note_name = _safe_filename(normalized.replace("/", "__"))
        _write(run_dir / "notes" / note_name, _truncate(content, _MAX_NOTE_CHARS), written, run_dir)

    for index, score in enumerate(source_scores, start=1):
        summary = _source_summary_for_score(score, evidence_packet=evidence_packet)
        _write(
            run_dir / "source_summaries" / f"source_{index:03d}.json",
            json.dumps(summary, indent=2, ensure_ascii=False),
            written,
            run_dir,
        )

    manifest = ResearchRunManifest(
        job_id=job_id,
        slug=slug,
        run_dir=str(run_dir),
        files=written,
        source_count=len(source_scores),
        claim_count=len(claim_table.all_claims()) if claim_table else 0,
        gap_count=_count_markdown_items(virtual_artifacts["/shared/gaps.md"]),
        contradiction_count=_count_markdown_items(virtual_artifacts["/shared/contradictions.md"]),
    )
    _write(run_dir / "manifest.json", manifest.model_dump_json(indent=2), written, run_dir)
    manifest.files = written
    return manifest


def _read_text_file(files: dict[str, Any], normalized_path: str) -> str:
    candidates = {normalized_path, f"/{normalized_path}"}
    for path, value in files.items():
        if str(path).lstrip("/") in candidates or str(path) in candidates:
            return coerce_file_content(value).strip()
    return ""


def _read_json_file(files: dict[str, Any], normalized_path: str) -> dict[str, Any] | None:
    content = _read_text_file(files, normalized_path)
    if not content:
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_claim_table(content: str) -> ClaimTable | None:
    if not content:
        return None
    table, _errors = validate_claim_table_json(content)
    return table


def _read_evidence_packet(files: dict[str, Any]) -> EvidencePacket | None:
    content = _read_text_file(files, "shared/evidence_packet.json")
    if not content:
        return None
    try:
        return EvidencePacket.model_validate_json(content)
    except Exception:
        return None


def _build_source_scores(
    *,
    evidence_packet: EvidencePacket | None,
    registry_sources: list[Any],
    request_text: str = "",
) -> list[SourceScore]:
    scores: dict[str, SourceScore] = {}
    if evidence_packet:
        for source in evidence_packet.sources:
            score = score_source(
                url=source.url,
                title=source.title,
                source_class=source.source_class,
                used_for=source.claim_ids,
                extract_count=len(source.extracts),
                relevance_text=request_text,
                evidence_text=" ".join(extract.text for extract in source.extracts[:5]),
                generated_at=evidence_packet.generated_at,
            )
            scores[_normalize_url(score.url)] = score
    for entry in registry_sources:
        url = str(getattr(entry, "url", "") or "")
        if not url:
            continue
        key = _normalize_url(url)
        if key in scores:
            continue
        score = score_source(
            url=url,
            title=getattr(entry, "title", None),
            source_class=getattr(entry, "source_class", None),
            relevance_text=request_text,
        )
        if score.relevance <= 1:
            continue
        scores[key] = score
    return sorted(scores.values(), key=lambda score: (-score.authority, score.bias_risk, -score.relevance, score.url))


def _build_gaps_markdown(
    *,
    plan: dict[str, Any] | None,
    claim_table: ClaimTable | None,
    evidence_packet: EvidencePacket | None,
) -> str:
    lines = ["# Gaps", ""]
    gaps: list[str] = []
    if not evidence_packet or evidence_packet.source_count == 0:
        gaps.append("No ranked evidence packet was available.")
    if not claim_table or not claim_table.all_claims():
        gaps.append("No usable claim table was available.")
    elif claim_table.unverified_count():
        for claim in claim_table.all_claims():
            if getattr(claim, "status", "") == "unverified":
                claim_text = getattr(claim, "claim_text", getattr(claim, "claim", ""))
                gaps.append(f"Unverified claim {getattr(claim, 'claim_id', '?')}: {claim_text}")
    for query in (plan or {}).get("queries", []) or []:
        if isinstance(query, dict) and not query.get("target_claims"):
            gaps.append(
                f"Plan query {query.get('task_id') or '?'} lacks explicit target claims: {query.get('query', '')}"
            )
    if not gaps:
        lines.append("No deterministic gaps were detected from the current claim table and evidence packet.")
    else:
        lines.extend(f"- {gap}" for gap in gaps)
    return "\n".join(lines).strip() + "\n"


def _build_contradictions_markdown(
    *,
    evidence_packet: EvidencePacket | None,
    claim_table: ClaimTable | None,
) -> str:
    lines = ["# Contradictions", ""]
    contradictions: list[str] = []
    if claim_table:
        by_id: dict[str, set[str]] = {}
        for claim in claim_table.all_claims():
            claim_id = getattr(claim, "claim_id", "")
            value = str(getattr(claim, "resolved_value", getattr(claim, "value", "")) or "").strip().lower()
            if claim_id and value:
                by_id.setdefault(claim_id, set()).add(value)
        for claim_id, values in by_id.items():
            if len(values) > 1:
                contradictions.append(f"Claim {claim_id} has multiple resolved values: {', '.join(sorted(values)[:5])}")
    if evidence_packet:
        support_map: dict[str, set[str]] = {}
        for source in evidence_packet.sources:
            for extract in source.extracts:
                text = extract.text.lower()
                polarity = (
                    "negative"
                    if any(term in text for term in ("not ", "no ", "failed", "declined", "decrease"))
                    else "positive"
                )
                for claim_id in extract.claim_ids:
                    support_map.setdefault(claim_id, set()).add(polarity)
        for claim_id, polarities in support_map.items():
            if {"positive", "negative"}.issubset(polarities):
                contradictions.append(
                    f"Claim {claim_id} has both positive and negative evidence polarity; review source extracts."
                )
    if not contradictions:
        lines.append("No deterministic contradictions were detected. This does not replace human/source review.")
    else:
        lines.extend(f"- {item}" for item in contradictions)
    return "\n".join(lines).strip() + "\n"


def _build_research_markdown(
    *,
    request_text: str,
    plan: dict[str, Any] | None,
    claim_table: ClaimTable | None,
    evidence_packet: EvidencePacket | None,
    source_scores: list[SourceScore],
    gaps_md: str,
    contradictions_md: str,
) -> str:
    title = (plan or {}).get("report_title") or _title_from_request(request_text)
    lines = [
        f"# Research Compile: {title}",
        "",
        "## Executive Evidence Summary",
        "",
        f"- Sources scored: {len(source_scores)}",
        f"- Evidence packet sources: {evidence_packet.source_count if evidence_packet else 0}",
        f"- Claim count: {len(claim_table.all_claims()) if claim_table else 0}",
        f"- Verified claims: {claim_table.verified_count() if claim_table else 0}",
        f"- Partially verified claims: {claim_table.partially_verified_count() if claim_table else 0}",
        f"- Unverified claims: {claim_table.unverified_count() if claim_table else 0}",
        "",
        "## Source Table",
        "",
        "| Source | Class | Authority | Recency | Relevance | Bias Risk | Used For |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for score in source_scores[:80]:
        title_or_host = score.title or urlparse(score.url).hostname or score.url
        lines.append(
            f"| [{_escape_table(title_or_host)}]({score.url}) | {score.source_class} | {score.authority} | "
            f"{score.recency} | {score.relevance} | {score.bias_risk} | {', '.join(score.used_for[:6]) or '-'} |"
        )
    lines.extend(["", "## Claim-by-Claim Evidence", ""])
    if claim_table and claim_table.all_claims():
        for claim in claim_table.all_claims()[:120]:
            claim_id = getattr(claim, "claim_id", "?")
            claim_text = getattr(claim, "claim_text", getattr(claim, "claim", ""))
            status = getattr(claim, "status", "unknown")
            value = getattr(claim, "resolved_value", getattr(claim, "value", None))
            lines.append(f"### {claim_id} — {status}")
            lines.append("")
            lines.append(str(claim_text))
            if value:
                lines.append(f"\nResolved value: {value}")
            evidence_items = list(getattr(claim, "evidence", []) or [])
            for evidence in evidence_items[:3]:
                lines.append(f"- {getattr(evidence, 'source_url', '')}: {getattr(evidence, 'extract', '')}")
            lines.append("")
    else:
        lines.append("No claim table was available.")
    lines.extend(["", "## Disagreements", "", contradictions_md.replace("# Contradictions", "").strip()])
    lines.extend(["", "## Unknowns and Gaps", "", gaps_md.replace("# Gaps", "").strip()])
    lines.extend(
        [
            "",
            "## Recommended Synthesis Direction",
            "",
            "Write the final report from verified and partially verified claims first. Hedge weakly sourced "
            "numeric or causal claims, and omit unverified claims unless the limitation is important to the answer.",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _plan_markdown(plan: dict[str, Any] | None, *, request_text: str) -> str:
    if not plan:
        return f"# Plan\n\nNo plan JSON was available.\n\n## Original Request\n\n{request_text.strip()}\n"
    lines = [f"# {plan.get('report_title') or 'Research Plan'}", "", "## Main Question", ""]
    task_analysis = plan.get("task_analysis") or {}
    lines.append(str(task_analysis.get("user_intent") or request_text or "Not recorded."))
    lines.extend(["", "## Sections", ""])
    for item in plan.get("report_toc", []) or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('title', 'Untitled section')}")
    lines.extend(["", "## Search Strategy", ""])
    for query in plan.get("queries", []) or []:
        if isinstance(query, dict):
            lines.append(
                f"- **{query.get('task_id') or '?'}** ({query.get('budget_percent', '?')}%, "
                f"{query.get('search_budget', '?')} calls): {query.get('query', '')}"
            )
    lines.extend(["", "## Risky Claims / Claim Targets", ""])
    claims = (task_analysis.get("claim_profile") or {}).get("claims") or []
    for claim in claims[:80]:
        if isinstance(claim, dict):
            lines.append(f"- {claim.get('claim_id', '?')}: {claim.get('claim_text') or claim.get('claim')}")
    return "\n".join(lines).strip() + "\n"


def _source_summary_for_score(score: SourceScore, *, evidence_packet: EvidencePacket | None) -> dict[str, Any]:
    summary = score.model_dump()
    if evidence_packet:
        for source in evidence_packet.sources:
            if _normalize_url(source.url) == _normalize_url(score.url):
                summary["extracts"] = [
                    extract.model_dump() for extract in source.extracts[:_MAX_SOURCE_SUMMARY_EXTRACTS]
                ]
                break
    return summary


def _is_note_path(normalized_path: str) -> bool:
    name = normalized_path.lower()
    if not name.startswith("shared/"):
        return False
    skip_names = ("claims/", "extracts/", "plan.json", "evidence_packet", "claim_table", "sources.json")
    if any(skip in name for skip in skip_names):
        return False
    return name.endswith((".md", ".txt"))


def _write(path: Path, content: str, written: dict[str, str], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    written[str(path.relative_to(root))] = str(path)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug or "research-run")[:80]


def _safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)[:160] or "note.md"


def _truncate(content: str, limit: int) -> str:
    content = content.strip()
    if len(content) <= limit:
        return content
    return content[:limit].rstrip() + f"\n\n[truncated from {len(content)} characters]\n"


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _count_markdown_items(content: str) -> int:
    return sum(1 for line in content.splitlines() if line.startswith("- "))


def _title_from_request(request_text: str) -> str:
    first = next((line.strip() for line in request_text.splitlines() if line.strip()), "Research Report")
    return first[:100].rstrip(" .,:;") or "Research Report"


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")

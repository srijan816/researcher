# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in generated illustrations for finalized deep-research reports.

When a job is submitted with ``include_images=true`` the agent runs one fast
LLM call over the finished report's outline to pick up to three genuinely
visual opportunities (concepts, comparisons, scenes — never charts of specific
data), generates each via the MiniMax image-generation API, saves the bytes
under a per-job artifact directory, and inserts standard markdown image tags
right after the chosen section headings.

Fail-open philosophy: any error at any step — LLM parse failure, API error,
download failure, disk trouble — leaves the report exactly as it was, with a
single warning log. The stage runs under a hard total time budget and never
delays or breaks report finalization. Kill switch: AIQ_REPORT_IMAGES_ENABLED
(default on).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_REPORT_IMAGES = 3
IMAGE_STAGE_BUDGET_SECONDS = 60.0

_MINIMAX_IMAGE_URL = "https://api.minimax.io/v1/image_generation"
_MINIMAX_IMAGE_MODEL = "image-01"
_IMAGE_REQUEST_TIMEOUT_SECONDS = 45.0
_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_SECTION_SNIPPET_CHARS = 200
_MAX_PROMPT_CHARS = 1200
_MAX_CAPTION_CHARS = 300

# Prefer the mounted data volume (/app/data in the container) so images
# survive container recreates and the path is absolute regardless of cwd;
# fall back to a cwd-relative dir for host/dev runs.
_DEFAULT_IMAGES_DIR = (
    Path("/app/data/report_images")
    if Path("/app/data").is_dir()
    else Path(".deep-research-runtime/report_images")
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SAFE_JOB_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_FALSEY = {"0", "false", "no", "off"}

_PLANNING_PROMPT = """You are choosing illustration opportunities for a finished research report.

Below is the report outline: its title, section headings, and the first ~200 characters of each section.

Pick the sections where a generated illustration would genuinely help the reader visualize a concept, comparison, or scene. Do NOT propose charts, graphs, plots, or any image containing specific numbers, data, logos, or text — image generators cannot render those accurately. Prefer 2 images; never more than 3. If nothing is genuinely visual, return an empty list.

Respond with ONLY a JSON array (no prose, no code fences) of objects:
[{{"after_heading": "<exact heading text from the outline>", "prompt": "<detailed English image-generation prompt, photorealistic or clean editorial illustration style>", "caption": "<short figure caption>"}}]

REPORT OUTLINE:
{outline}
"""


@dataclass(frozen=True)
class ReportImageSpec:
    """One planned illustration: where it goes, how to generate it, its caption."""

    after_heading: str
    prompt: str
    caption: str


def report_images_enabled() -> bool:
    """Kill switch: AIQ_REPORT_IMAGES_ENABLED (default on)."""
    return os.getenv("AIQ_REPORT_IMAGES_ENABLED", "1").strip().lower() not in _FALSEY


def report_images_dir(job_id: str) -> Path:
    """Per-job directory where generated report images are persisted.

    Shared between the agent worker (writer) and the API image route (reader)
    via the AIQ_REPORT_IMAGES_DIR env var; both run with the same working
    directory in the in-process executor deployment.
    """
    root = Path(os.environ.get("AIQ_REPORT_IMAGES_DIR", str(_DEFAULT_IMAGES_DIR)))
    safe_job_id = _SAFE_JOB_ID_RE.sub("_", job_id or "job")[:120] or "job"
    return root / safe_job_id


def build_image_planning_outline(report: str, *, snippet_chars: int = _SECTION_SNIPPET_CHARS) -> str:
    """Compact outline of the report: title + headings + leading snippet per section."""
    lines: list[str] = []
    matches = list(_HEADING_RE.finditer(report))
    if not matches:
        return report[: snippet_chars * 4].strip()
    for index, match in enumerate(matches):
        level, heading = match.group(1), match.group(2).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        snippet = re.sub(r"\s+", " ", report[body_start:body_end]).strip()[:snippet_chars]
        lines.append(f"{level} {heading}")
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines)


def parse_image_specs(text: str, *, max_images: int = MAX_REPORT_IMAGES) -> list[ReportImageSpec]:
    """Parse the planning LLM's JSON reply into validated specs (tolerant, pure).

    Accepts raw JSON arrays, code-fenced JSON, or JSON embedded in prose.
    Invalid entries are skipped; at most ``max_images`` specs are returned.
    """
    if not text or not text.strip():
        return []
    payload = _extract_json_array(text)
    if payload is None:
        return []
    specs: list[ReportImageSpec] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        after_heading = str(entry.get("after_heading") or "").strip()
        prompt = str(entry.get("prompt") or "").strip()[:_MAX_PROMPT_CHARS]
        caption = str(entry.get("caption") or "").strip()[:_MAX_CAPTION_CHARS]
        if not after_heading or not prompt:
            continue
        specs.append(ReportImageSpec(after_heading=after_heading, prompt=prompt, caption=caption or after_heading))
        if len(specs) >= max_images:
            break
    return specs


def insert_report_images(report: str, images: list[tuple[ReportImageSpec, str]]) -> tuple[str, int]:
    """Insert ``![caption](url)`` after each spec's heading (pure; returns new text).

    ``images`` pairs each spec with its final (already-servable) URL. Headings
    are matched case-insensitively against the heading text, ignoring leading
    ``#`` markers in the spec. Specs whose heading cannot be found are skipped.
    Returns the updated report and the number of images inserted.
    """
    if not images:
        return report, 0
    lines = report.split("\n")
    inserted = 0
    for spec, url in images[:MAX_REPORT_IMAGES]:
        wanted = _normalize_heading(spec.after_heading)
        if not wanted or not url:
            continue
        for index, line in enumerate(lines):
            match = _HEADING_RE.match(line)
            if match and _normalize_heading(match.group(2)) == wanted:
                caption = spec.caption.replace("]", ")").replace("[", "(").strip()
                lines.insert(index + 1, "")
                lines.insert(index + 2, f"![{caption}]({url})")
                inserted += 1
                break
    return "\n".join(lines), inserted


async def generate_and_save_report_images(
    *,
    report: str,
    job_id: str,
    llm: Any,
    api_key: str | None = None,
    image_url_prefix: str,
    budget_seconds: float = IMAGE_STAGE_BUDGET_SECONDS,
) -> tuple[str, int]:
    """Full image stage: plan → generate → save → insert. Fully fail-open.

    Returns ``(possibly-updated report, images inserted)``. On any error the
    original report is returned unchanged with a single warning log.
    """
    try:
        return await asyncio.wait_for(
            _run_image_stage(
                report=report,
                job_id=job_id,
                llm=llm,
                api_key=api_key or os.environ.get("MINIMAX_API_KEY"),
                image_url_prefix=image_url_prefix,
            ),
            timeout=budget_seconds,
        )
    except Exception:  # noqa: BLE001 - the image stage must never break the report
        logger.warning("Report image stage failed (fail-open); report unchanged", exc_info=True)
        return report, 0


async def _run_image_stage(
    *,
    report: str,
    job_id: str,
    llm: Any,
    api_key: str | None,
    image_url_prefix: str,
) -> tuple[str, int]:
    if llm is None or not api_key:
        logger.warning("Report image stage skipped: missing %s", "LLM" if llm is None else "MINIMAX_API_KEY")
        return report, 0

    from langchain_core.messages import HumanMessage

    outline = build_image_planning_outline(report)
    response = await llm.ainvoke([HumanMessage(content=_PLANNING_PROMPT.format(outline=outline))])
    specs = parse_image_specs(_coerce_llm_text(response))
    if not specs:
        logger.info("Report image stage: planner chose no image opportunities")
        return report, 0

    target_dir = report_images_dir(job_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[ReportImageSpec, str]] = []
    async with httpx.AsyncClient(timeout=_IMAGE_REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for index, spec in enumerate(specs, start=1):
            try:
                image_bytes = await _generate_minimax_image(client, spec.prompt, api_key)
            except Exception:  # noqa: BLE001 - skip this image, keep the rest
                logger.warning("Report image %d generation failed; skipping", index, exc_info=True)
                continue
            if not image_bytes:
                continue
            filename = f"image-{index}.{_image_extension(image_bytes)}"
            (target_dir / filename).write_bytes(image_bytes)
            images.append((spec, f"{image_url_prefix.rstrip('/')}/{filename}"))

    updated, inserted = insert_report_images(report, images)
    if inserted:
        logger.info("Report image stage inserted %d generated image(s) for job %s", inserted, job_id)
    return updated, inserted


async def _generate_minimax_image(client: httpx.AsyncClient, prompt: str, api_key: str) -> bytes | None:
    """Generate one 16:9 image via MiniMax and return its raw bytes.

    Validated live against ``POST https://api.minimax.io/v1/image_generation``:
    response is ``{"data": {"image_urls": [...]}, "base_resp": {"status_code": 0}}``.
    """
    endpoint = os.environ.get("AIQ_MINIMAX_IMAGE_URL", _MINIMAX_IMAGE_URL)
    response = await client.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("AIQ_MINIMAX_IMAGE_MODEL", _MINIMAX_IMAGE_MODEL),
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "n": 1,
            "response_format": "url",
        },
    )
    response.raise_for_status()
    payload = response.json()
    base_resp = payload.get("base_resp") or {}
    if base_resp.get("status_code") not in (0, None):
        raise RuntimeError(f"MiniMax image generation error: {base_resp.get('status_msg')}")
    data = payload.get("data") or {}
    if data.get("image_base64"):
        import base64

        first = data["image_base64"][0] if isinstance(data["image_base64"], list) else data["image_base64"]
        return base64.b64decode(first)
    urls = data.get("image_urls") or []
    if not urls:
        raise RuntimeError("MiniMax image generation returned no image URLs")
    download = await client.get(str(urls[0]), timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    download.raise_for_status()
    return download.content


def _image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "png"
    return "jpg"


def _normalize_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading.lstrip("#").strip()).lower()


def _extract_json_array(text: str) -> list[Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    start, end = stripped.find("["), stripped.rfind("]")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, list):
            return payload
    return None


def _coerce_llm_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content or "")

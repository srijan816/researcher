# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in generated illustrations for finalized deep-research reports.

When a job is submitted with ``include_images=true`` the agent runs one fast
LLM call over the finished report's outline to pick 1-4 (default 3, via the
optional ``image_count`` request field) genuinely visual opportunities
(concepts, comparisons, scenes — never charts of specific data), generates
each as a hyper-realistic photograph via the provider chain, saves the bytes
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

MAX_REPORT_IMAGES = 4
DEFAULT_REPORT_IMAGES = 3
def _stage_budget_default() -> float:
    """Grok CLI generations take 1-3 min each; API providers are fast."""
    return 480.0 if grok_cli_available() and os.environ.get("AIQ_IMAGE_PROVIDER", "").lower() != "minimax" else 60.0


IMAGE_STAGE_BUDGET_SECONDS = 60.0

_MINIMAX_IMAGE_URL = "https://api.minimax.io/v1/image_generation"
_MINIMAX_IMAGE_MODEL = "image-01"
_XAI_IMAGE_URL = "https://api.x.ai/v1/images/generations"
_XAI_IMAGE_MODEL = "grok-2-image"
_MINIMAX_ANTHROPIC_MESSAGES_URL = "https://api.minimax.io/anthropic/v1/messages"
_PLANNING_MODEL = "MiniMax-M2.7-highspeed"
_PLANNING_MAX_TOKENS = 1024
_PLANNING_TIMEOUT_SECONDS = 45.0
BACKFILL_BUDGET_SECONDS = 90.0
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
# Appended to every per-image prompt sent to the Grok CLI so Grok Imagine's
# natural expressiveness is steered toward photography, not illustration.
_GROK_STYLE_SUFFIX = (
    ". Style: hyper-realistic, natural expressive lighting, photographic detail, "
    "realistic materials and textures, shallow depth of field where appropriate, "
    "no illustration or sketch style, no vector art, no cartoon, no diagram."
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SAFE_JOB_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_FALSEY = {"0", "false", "no", "off"}

_PLANNING_PROMPT = """You are choosing photographic illustration opportunities for a finished research report.

Below is the report outline: its title, section headings, and the first ~200 characters of each section.

The reader explicitly asked for this report to include visuals, so you MUST select the {preferred_count} best illustration opportunities (never more than {max_count}). Almost every research topic has visualizable moments: a process or mechanism in action, a place or scene, a comparison of physical things, people or equipment at work. Do NOT propose charts, graphs, plots, diagrams, or any image containing specific numbers, data, logos, or text — image generators cannot render those accurately; recast such ideas as real photographable scenes instead. Return an empty list ONLY if the report is so abstract that any photograph would be decoration with zero explanatory value (this should be rare).

STYLE — every "prompt" you write MUST describe a hyper-realistic photograph, never an illustration. Each prompt must:
- Explicitly say "hyper-realistic photograph" and describe a real, physical scene.
- Specify cinematic photography language: natural light, realistic materials and textures, believable environment, and shallow depth of field where it suits the subject.
- Explicitly end with: "photorealistic, NOT an illustration, NOT a vector, NOT a sketch, NOT a cartoon, NOT a diagram".

Respond with ONLY a JSON array (no prose, no code fences) of objects:
[{{"after_heading": "<exact heading text from the outline>", "prompt": "<detailed English image-generation prompt for a hyper-realistic photograph as specified above>", "caption": "<short figure caption>"}}]

REPORT OUTLINE:
{outline}
"""


def clamp_image_count(image_count: int | None) -> int:
    """Resolve an optional requested image count to the effective 1..MAX range (default 3)."""
    if image_count is None:
        return DEFAULT_REPORT_IMAGES
    return max(1, min(int(image_count), MAX_REPORT_IMAGES))


def build_planning_prompt(outline: str, *, image_count: int | None = None) -> str:
    """Render the planning prompt for the requested (clamped) image count."""
    count = clamp_image_count(image_count)
    return _PLANNING_PROMPT.format(outline=outline, preferred_count=count, max_count=count)


class ReportImagesError(RuntimeError):
    """Raised by the non-fail-open backfill path when image work cannot proceed."""


class ReportImagesTimeoutError(ReportImagesError):
    """Raised when the backfill image stage exceeds its time budget."""


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
    budget_seconds: float | None = None,
    image_count: int | None = None,
) -> tuple[str, int]:
    """Full image stage: plan → generate → save → insert. Fully fail-open.

    Returns ``(possibly-updated report, images inserted)``. On any error the
    original report is returned unchanged with a single warning log.
    """
    if budget_seconds is None:
        budget_seconds = _stage_budget_default()
    try:
        return await asyncio.wait_for(
            _run_image_stage(
                report=report,
                job_id=job_id,
                llm=llm,
                api_key=api_key or os.environ.get("MINIMAX_API_KEY"),
                image_url_prefix=image_url_prefix,
                image_count=image_count,
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
    image_count: int | None = None,
) -> tuple[str, int]:
    if llm is None or not (api_key or os.environ.get("XAI_API_KEY")):
        logger.warning("Report image stage skipped: missing %s", "LLM" if llm is None else "image API key")
        return report, 0

    from langchain_core.messages import HumanMessage

    outline = build_image_planning_outline(report)
    response = await llm.ainvoke([HumanMessage(content=build_planning_prompt(outline, image_count=image_count))])
    specs = parse_image_specs(_coerce_llm_text(response), max_images=clamp_image_count(image_count))
    if not specs:
        logger.info("Report image stage: planner chose no image opportunities")
        return report, 0

    images = await _generate_and_save_images(
        specs, job_id=job_id, image_url_prefix=image_url_prefix, minimax_api_key=api_key
    )
    updated, inserted = insert_report_images(report, images)
    if inserted:
        logger.info("Report image stage inserted %d generated image(s) for job %s", inserted, job_id)
    return updated, inserted


async def _generate_and_save_images(
    specs: list[ReportImageSpec],
    *,
    job_id: str,
    image_url_prefix: str,
    minimax_api_key: str | None,
) -> list[tuple[ReportImageSpec, str]]:
    """Generate each spec's image via the resolved provider, save to disk, return (spec, url) pairs.

    Per-image failures are logged and skipped so one bad generation never
    voids the rest; callers decide whether zero successes is an error.
    """
    target_dir = report_images_dir(job_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[ReportImageSpec, str]] = []
    async with httpx.AsyncClient(timeout=_IMAGE_REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for index, spec in enumerate(specs, start=1):
            try:
                image_bytes = await _generate_image(client, spec.prompt, minimax_api_key=minimax_api_key)
            except Exception:  # noqa: BLE001 - skip this image, keep the rest
                logger.warning("Report image %d generation failed; skipping", index, exc_info=True)
                continue
            if not image_bytes:
                continue
            filename = f"image-{index}.{_image_extension(image_bytes)}"
            (target_dir / filename).write_bytes(image_bytes)
            images.append((spec, f"{image_url_prefix.rstrip('/')}/{filename}"))
    return images


def grok_cli_available() -> bool:
    """True when the Grok Build CLI (authenticated session) is mounted and executable."""
    cli = os.environ.get("AIQ_GROK_CLI", "").strip()
    return bool(cli) and os.access(cli, os.X_OK)


def resolve_image_provider() -> str:
    """Pick the image provider.

    AIQ_IMAGE_PROVIDER forces one of minimax|xai|grok. Auto order:
    grok CLI (subscription, free) -> xAI (if XAI_API_KEY) -> MiniMax.
    Auto-selected providers fall back down the chain on failure.
    """
    forced = os.environ.get("AIQ_IMAGE_PROVIDER", "").strip().lower()
    if forced in {"minimax", "xai", "grok"}:
        return forced
    if grok_cli_available():
        return "grok"
    return "xai" if os.environ.get("XAI_API_KEY") else "minimax"


async def _generate_image(
    client: httpx.AsyncClient, prompt: str, *, minimax_api_key: str | None
) -> bytes | None:
    """Provider-dispatching image generation.

    - provider "xai" (forced): xAI only; errors propagate.
    - provider "xai" (auto, key present): try xAI, fall back to MiniMax on any error.
    - provider "minimax": MiniMax only (original behavior).
    """
    provider = resolve_image_provider()
    xai_key = os.environ.get("XAI_API_KEY")
    forced = os.environ.get("AIQ_IMAGE_PROVIDER", "").strip().lower() in {"minimax", "xai", "grok"}

    if provider == "grok":
        if forced:
            return await _generate_grok_image(prompt)
        try:
            data = await _generate_grok_image(prompt)
            if data:
                return data
            logger.warning("Grok image generation returned nothing; falling back")
        except Exception:  # noqa: BLE001 - auto mode falls back down the chain
            logger.warning("Grok image generation failed; falling back", exc_info=True)
        if xai_key:
            try:
                return await _generate_xai_image(client, prompt, xai_key)
            except Exception:  # noqa: BLE001
                logger.warning("xAI fallback failed; falling back to MiniMax", exc_info=True)
        if not minimax_api_key:
            raise ReportImagesError("Grok failed and MINIMAX_API_KEY is not set")
        return await _generate_minimax_image(client, prompt, minimax_api_key)

    if provider == "xai":
        if not xai_key:
            raise ReportImagesError("AIQ_IMAGE_PROVIDER=xai but XAI_API_KEY is not set")
        if forced:
            return await _generate_xai_image(client, prompt, xai_key)
        try:
            return await _generate_xai_image(client, prompt, xai_key)
        except Exception:  # noqa: BLE001 - auto mode falls back to MiniMax
            logger.warning("xAI image generation failed; falling back to MiniMax", exc_info=True)

    if not minimax_api_key:
        raise ReportImagesError("MINIMAX_API_KEY is not set")
    return await _generate_minimax_image(client, prompt, minimax_api_key)


async def _generate_xai_image(client: httpx.AsyncClient, prompt: str, api_key: str) -> bytes | None:
    """Generate one image via xAI's OpenAI-compatible image API and return its raw bytes.

    Contract (docs.x.ai): ``POST https://api.x.ai/v1/images/generations`` with
    ``{"model", "prompt", "n", "response_format"}``; response is OpenAI-shaped:
    ``{"data": [{"url": ...}]}`` (or ``b64_json`` when requested).
    """
    endpoint = os.environ.get("AIQ_XAI_IMAGE_URL", _XAI_IMAGE_URL)
    response = await client.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.environ.get("AIQ_XAI_IMAGE_MODEL", _XAI_IMAGE_MODEL),
            "prompt": prompt,
            "n": 1,
            "response_format": "url",
        },
    )
    response.raise_for_status()
    payload = response.json()
    entries = payload.get("data") or []
    if not entries or not isinstance(entries[0], dict):
        raise RuntimeError("xAI image generation returned no image data")
    first = entries[0]
    if first.get("b64_json"):
        import base64

        return base64.b64decode(first["b64_json"])
    url = first.get("url")
    if not url:
        raise RuntimeError("xAI image generation returned neither url nor b64_json")
    download = await client.get(str(url), timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    download.raise_for_status()
    return download.content


# ---------------------------------------------------------------------------
# Backfill path: add images to an already-persisted report (explicit user
# action; NOT fail-open — errors are raised so the API can report them).
# ---------------------------------------------------------------------------

_GENERATED_IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\((?:/api|/v1)/jobs/async/job/[^)]*/images/")


def report_contains_generated_images(report: str) -> bool:
    """True when the report already embeds generated job-image markdown (idempotency guard)."""
    return bool(report) and bool(_GENERATED_IMAGE_MARKDOWN_RE.search(report))


async def plan_image_specs_via_minimax(
    report: str,
    *,
    api_key: str,
    model: str | None = None,
    timeout_seconds: float = _PLANNING_TIMEOUT_SECONDS,
    image_count: int | None = None,
) -> list[ReportImageSpec]:
    """Run the image-planning prompt via a plain httpx call to MiniMax's Anthropic-compatible endpoint.

    Keeps the API layer free of agent LLM internals. Raises ReportImagesError
    on transport/HTTP errors; parse tolerance matches the agent path.
    """
    endpoint = os.environ.get("AIQ_MINIMAX_ANTHROPIC_URL", _MINIMAX_ANTHROPIC_MESSAGES_URL)
    payload = {
        "model": os.environ.get("AIQ_IMAGE_PLANNING_MODEL", model or _PLANNING_MODEL),
        "max_tokens": _PLANNING_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": build_planning_prompt(build_image_planning_outline(report), image_count=image_count),
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise ReportImagesError(f"Image planning LLM call failed: {exc}") from exc
    parts = body.get("content") or []
    text = "".join(
        str(part.get("text") or "") for part in parts if isinstance(part, dict) and part.get("type") == "text"
    )
    return parse_image_specs(text, max_images=clamp_image_count(image_count))


async def backfill_report_images(
    *,
    report: str,
    job_id: str,
    image_url_prefix: str,
    budget_seconds: float | None = None,
    image_count: int | None = None,
) -> tuple[str, int]:
    """Add generated images to an existing report. NOT fail-open: raises ReportImagesError.

    Returns ``(updated report, images inserted)``. ``(report, 0)`` when the
    planner legitimately picks no opportunities.
    """
    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if not minimax_key:
        raise ReportImagesError("MINIMAX_API_KEY is not configured on the server")
    if budget_seconds is None:
        budget_seconds = 480.0 if resolve_image_provider() == "grok" else BACKFILL_BUDGET_SECONDS
    try:
        return await asyncio.wait_for(
            _run_backfill(
                report=report,
                job_id=job_id,
                image_url_prefix=image_url_prefix,
                minimax_key=minimax_key,
                image_count=image_count,
            ),
            timeout=budget_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ReportImagesTimeoutError(
            f"Image backfill exceeded its {budget_seconds:.0f}s time budget"
        ) from exc


async def _run_backfill(
    *, report: str, job_id: str, image_url_prefix: str, minimax_key: str, image_count: int | None = None
) -> tuple[str, int]:
    specs = await plan_image_specs_via_minimax(report, api_key=minimax_key, image_count=image_count)
    if not specs:
        return report, 0
    images = await _generate_and_save_images(
        specs, job_id=job_id, image_url_prefix=image_url_prefix, minimax_api_key=minimax_key
    )
    if not images:
        raise ReportImagesError("All image generations failed; report left unchanged")
    updated, inserted = insert_report_images(report, images)
    if not inserted:
        raise ReportImagesError("Generated images could not be matched to any report heading")
    return updated, inserted


async def _generate_grok_image(prompt: str, *, timeout_seconds: float | None = None) -> bytes | None:
    """Generate an image via the mounted Grok Build CLI (authenticated subscription).

    Runs a single-turn headless prompt instructing the agent to call its
    native image tool and save the file to a temp path we control, then
    reads the bytes back. Slower than a direct API (~1-3 min) but free.
    """
    import tempfile

    cli = os.environ.get("AIQ_GROK_CLI", "").strip()
    if not cli or not os.access(cli, os.X_OK):
        raise ReportImagesError("Grok CLI is not available (AIQ_GROK_CLI)")
    if timeout_seconds is None:
        try:
            timeout_seconds = float(os.environ.get("AIQ_GROK_IMAGE_TIMEOUT_SECONDS", "210"))
        except ValueError:
            timeout_seconds = 210.0
    grok_home = os.environ.get("AIQ_GROK_HOME", "").strip() or os.path.expanduser("~")
    with tempfile.TemporaryDirectory(prefix="grok-img-") as tmp:
        out_path = os.path.join(tmp, "image.jpg")
        instruction = (
            "Call your image generation tool with this exact prompt and aspect_ratio '16:9', "
            f"then save/copy the resulting image file to {out_path} and stop. "
            "Use the image tool only - no code, no matplotlib, no SVG. Prompt: "
            + prompt
            + _GROK_STYLE_SUFFIX
        )
        env = {**os.environ, "HOME": grok_home}
        proc = await asyncio.create_subprocess_exec(
            cli,
            "-p",
            instruction,
            "--always-approve",
            "--cwd",
            tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            proc.kill()
            raise ReportImagesError(f"Grok image generation timed out after {timeout_seconds:.0f}s")
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            # The agent sometimes writes a differently-named file in cwd.
            candidates = [
                os.path.join(tmp, f)
                for f in os.listdir(tmp)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and os.path.getsize(os.path.join(tmp, f)) >= 1024
            ]
            if not candidates:
                raise ReportImagesError("Grok run finished but produced no image file")
            out_path = candidates[0]
        with open(out_path, "rb") as fh:
            return fh.read()


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

"""NotebookLM source-harvest tool for AI-Q shallow research.

The tool treats NotebookLM as an external discovery and scraping worker. It
does not use NotebookLM's generated research report as an answer; it imports
the discovered sources, pulls source full text where available, and returns
citation-ready documents for the AI-Q researcher to synthesize.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

GENERATED_REPORT_TITLE_RE = re.compile(
    r"(generated|research)\s+(markdown\s+)?(deep[-\s]?research\s+)?report|deep[-\s]?research\s+report",
    re.IGNORECASE,
)


class NotebookLMSourceHarvestConfig(FunctionBaseConfig, name="notebooklm_source_harvest"):
    """Harvest sources through NotebookLM web/deep research."""

    python_executable: str = Field(
        default_factory=lambda: os.getenv("NOTEBOOKLM_PYTHON", sys.executable),
        description="Python executable that can run `-m notebooklm`.",
    )
    auth_home: str | None = Field(
        default_factory=lambda: os.getenv("NOTEBOOKLM_HOME"),
        description="Optional NotebookLM auth directory containing `profiles/default/storage_state.json`.",
    )
    mode: str = Field(default="deep", description="NotebookLM research mode: fast or deep.")
    query_count: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Sequential NotebookLM research queries to run per tool call.",
    )
    timeout_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Per-phase NotebookLM research/import timeout.",
    )
    max_sources: int = Field(default=36, ge=1, le=120, description="Maximum unique sources returned.")
    max_fulltext_sources: int = Field(
        default=24,
        ge=0,
        le=80,
        description="Maximum ready sources whose full text should be fetched.",
    )
    max_content_length: int = Field(
        default=4500,
        ge=500,
        le=20000,
        description="Maximum characters of full text/snippet returned per source.",
    )
    early_stop_ready_sources: int = Field(
        default=24,
        ge=0,
        le=120,
        description=(
            "If >0, stop waiting on a blocking NotebookLM deep-research import once this many ready "
            "sources are visible in the notebook."
        ),
    )
    early_stop_poll_seconds: int = Field(
        default=20,
        ge=5,
        le=300,
        description="How often to inspect NotebookLM sources while add-research is still running.",
    )
    early_stop_stable_polls: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of consecutive ready-source polls required before early-stopping add-research.",
    )
    source_inspection_timeout_seconds: int = Field(
        default=15,
        ge=5,
        le=120,
        description=(
            "Short timeout for source-list inspections while add-research is still running. "
            "This prevents NotebookLM CLI locks from hiding already-imported sources."
        ),
    )
    post_timeout_source_poll_seconds: int = Field(
        default=300,
        ge=0,
        le=1800,
        description=(
            "After NotebookLM reports a research timeout, keep polling briefly because sources can appear "
            "after the CLI timeout message while the notebook import settles."
        ),
    )
    post_timeout_source_poll_interval_seconds: int = Field(
        default=20,
        ge=5,
        le=300,
        description="Polling interval used during the post-timeout source visibility window.",
    )
    include_generated_reports: bool = Field(
        default=False,
        description="Include NotebookLM generated report artifacts as sources. Disabled for source-only harvesting.",
    )
    delete_notebook_after: bool = Field(
        default=False,
        description="Delete the temporary NotebookLM notebook after harvesting. Disabled by default for auditability.",
    )
    work_dir: str = Field(
        default_factory=lambda: os.getenv("NOTEBOOKLM_HARVEST_WORK_DIR", "/tmp/aiq-notebooklm-harvest"),
        description="Directory for temporary prompt and fulltext files.",
    )
    query_templates: list[str] = Field(
        default_factory=lambda: [
            "{query} current authoritative sources primary evidence recent data",
            "{query} expert analysis case studies tradeoffs risks counterarguments",
            "{query} examples implementation lessons criticism unresolved questions",
        ],
        description="Templates used to split one broad user query into sequential NotebookLM harvests.",
    )


@dataclass
class HarvestSource:
    source_id: str
    title: str
    url: str
    status: str
    source_type: str
    content: str = ""
    error: str = ""


def _truncate(text: str, max_length: int) -> str:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(normalized) <= max_length:
        return normalized
    cutoff = normalized.rfind(" ", 0, max_length - 3)
    if cutoff < max_length // 2:
        cutoff = max_length - 3
    return normalized[:cutoff].rstrip() + "..."


def _json_loads_lenient(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def _first_string_by_key(value: Any, keys: set[str]) -> str:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, str) and child.strip():
                return child.strip()
        for child in value.values():
            found = _first_string_by_key(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _first_string_by_key(child, keys)
            if found:
                return found
    return ""


def _extract_notebook_id(payload: Any) -> str:
    return _first_string_by_key(payload, {"notebook_id", "notebookId", "id", "project_id", "projectId"})


def _iter_source_like(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return value
        return []
    if not isinstance(value, dict):
        return []
    for key in ("sources", "items", "data", "results"):
        child = value.get(key)
        if isinstance(child, list):
            return [item for item in child if isinstance(item, dict)]
    if any(key in value for key in ("source_id", "sourceId", "id", "title", "url")):
        return [value]
    return []


def _normalize_status(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"ready", "done", "complete", "completed", "success", "processed"}:
        return "ready"
    if text in {"error", "failed", "failure"}:
        return "error"
    if text:
        return text
    return "unknown"


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _is_generated_report(source: HarvestSource) -> bool:
    if source.source_type.lower() in {"generated_report", "report", "artifact"}:
        return True
    return bool(GENERATED_REPORT_TITLE_RE.search(source.title))


def _source_from_dict(raw: dict[str, Any]) -> HarvestSource:
    source_id = _first_string_by_key(raw, {"source_id", "sourceId", "id"})
    title = _first_string_by_key(raw, {"title", "displayTitle", "name"}) or source_id or "NotebookLM source"
    url = _first_string_by_key(raw, {"url", "sourceUrl", "source_url", "uri", "webUrl"})
    status = _normalize_status(raw.get("status") or raw.get("state") or raw.get("sourceStatus"))
    source_type = str(raw.get("source_type") or raw.get("sourceType") or raw.get("type") or "notebooklm").strip()
    error = _first_string_by_key(raw, {"error", "errorMessage", "failureReason"})
    return HarvestSource(source_id=source_id, title=title, url=url, status=status, source_type=source_type, error=error)


def _dedupe_sources(sources: list[HarvestSource]) -> list[HarvestSource]:
    unique: dict[str, HarvestSource] = {}
    for source in sources:
        key = source.url.rstrip("/") if source.url else f"id:{source.source_id}"
        if not key or key == "id:":
            key = source.title.lower()
        existing = unique.get(key)
        if existing is None:
            unique[key] = source
            continue
        if existing.status != "ready" and source.status == "ready":
            unique[key] = source
        elif not existing.content and source.content:
            unique[key] = source
    return list(unique.values())


def _render_sources(
    query: str,
    notebook_id: str,
    subqueries: list[str],
    sources: list[HarvestSource],
    warnings: list[str],
    max_content_length: int,
) -> str:
    ready_count = sum(1 for source in sources if source.status == "ready")
    error_count = sum(1 for source in sources if source.status == "error")
    docs: list[str] = [
        f'<notebooklm_harvest provider="notebooklm" mode="source_harvest" '
        f'notebook_id="{escape(notebook_id)}" ready_sources="{ready_count}" error_sources="{error_count}">',
        f"<original_query>{escape(query)}</original_query>",
    ]
    for idx, subquery in enumerate(subqueries, 1):
        docs.append(f'<harvest_query idx="{idx}">{escape(subquery)}</harvest_query>')
    for warning in warnings:
        docs.append(f"<harvest_warning>{escape(warning)}</harvest_warning>")

    for idx, source in enumerate(sources):
        content = _truncate(source.content, max_content_length) if source.content else ""
        docs.append(
            f'<document idx="{idx}" provider="notebooklm" status="{escape(source.status)}" '
            f'source_id="{escape(source.source_id)}" source_type="{escape(source.source_type)}">\n'
            f"<title>{escape(source.title)}</title>\n"
            f"<url>{escape(source.url)}</url>\n"
            f"<domain>{escape(_hostname(source.url))}</domain>\n"
            f"<content>{escape(content)}</content>\n"
            f"<error>{escape(source.error)}</error>\n"
            f"</document>"
        )
    docs.append("</notebooklm_harvest>")
    return "\n\n".join(docs)


class NotebookLMCLI:
    def __init__(self, config: NotebookLMSourceHarvestConfig):
        self.config = config
        self.work_dir = Path(config.work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.auth_home:
            env["NOTEBOOKLM_HOME"] = self.config.auth_home
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def run(self, args: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        command = [self.config.python_executable, "-m", "notebooklm", *args]
        logger.info("Running NotebookLM command: %s", " ".join(command[:4] + ["..."]))
        return subprocess.run(
            command,
            cwd=self.work_dir,
            env=self._env(),
            text=True,
            capture_output=True,
            timeout=timeout or self.config.timeout_seconds + 60,
            check=False,
        )

    def _popen(self, args: list[str]) -> subprocess.Popen[str]:
        command = [self.config.python_executable, "-m", "notebooklm", *args]
        logger.info("Running NotebookLM command: %s", " ".join(command[:4] + ["..."]))
        return subprocess.Popen(
            command,
            cwd=self.work_dir,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def create_notebook(self, title: str) -> str:
        result = self.run(["create", title, "--json"], timeout=120)
        if result.returncode != 0:
            raise RuntimeError(_truncate(result.stderr or result.stdout or "NotebookLM create failed", 1000))
        notebook_id = _extract_notebook_id(_json_loads_lenient(result.stdout))
        if not notebook_id:
            raise RuntimeError("NotebookLM create did not return a notebook id")
        return notebook_id

    def add_research(self, notebook_id: str, query: str) -> str:
        prompt_file = self.work_dir / f"notebooklm_query_{time.time_ns()}.txt"
        prompt_file.write_text(query, encoding="utf-8")
        args = [
            "source",
            "add-research",
            "--prompt-file",
            str(prompt_file),
            "--notebook",
            notebook_id,
            "--from",
            "web",
            "--mode",
            self.config.mode,
            "--import-all",
            "--timeout",
            str(self.config.timeout_seconds),
        ]
        process = self._popen(args)

        try:
            timeout_seconds = (self.config.timeout_seconds * 2) + 120
            deadline = time.monotonic() + timeout_seconds
            last_ready_count = -1
            stable_ready_polls = 0

            while True:
                remaining = max(1, int(deadline - time.monotonic()))
                poll_timeout = min(self.config.early_stop_poll_seconds, remaining)
                try:
                    stdout, stderr = process.communicate(timeout=poll_timeout)
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        process.kill()
                        stdout, stderr = process.communicate()
                        return _truncate(stderr or stdout or "NotebookLM research timed out", 1500)

                    early_stop_target = self.config.early_stop_ready_sources
                    if early_stop_target <= 0:
                        continue

                    try:
                        visible_sources = self.list_sources(
                            notebook_id,
                            timeout=self.config.source_inspection_timeout_seconds,
                        )
                    except Exception as exc:  # pragma: no cover - depends on NotebookLM runtime state
                        logger.debug("NotebookLM early-stop source inspection failed: %s", exc)
                        continue

                    ready_count = sum(1 for source in visible_sources if source.status == "ready")
                    if ready_count >= early_stop_target:
                        if ready_count == last_ready_count:
                            stable_ready_polls += 1
                        else:
                            stable_ready_polls = 1
                            last_ready_count = ready_count

                        if stable_ready_polls >= self.config.early_stop_stable_polls:
                            process.terminate()
                            try:
                                stdout, stderr = process.communicate(timeout=20)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                stdout, stderr = process.communicate()
                            warning = (
                                "NotebookLM add-research was stopped early after "
                                f"{ready_count} ready sources were visible."
                            )
                            if stderr:
                                warning = f"{warning} CLI stderr: {_truncate(stderr, 500)}"
                            return warning
                    continue

                if process.returncode != 0:
                    return _truncate(stderr or stdout or "NotebookLM research failed", 1500)
                return ""
        finally:
            try:
                prompt_file.unlink(missing_ok=True)
            except OSError:
                pass

    def list_sources(self, notebook_id: str, *, timeout: int = 120) -> list[HarvestSource]:
        result = self.run(["source", "list", "--notebook", notebook_id, "--json"], timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(_truncate(result.stderr or result.stdout or "NotebookLM source list failed", 1000))
        payload = _json_loads_lenient(result.stdout)
        return [_source_from_dict(item) for item in _iter_source_like(payload)]

    def fulltext(self, notebook_id: str, source_id: str) -> str:
        result, output_path = self._fulltext(notebook_id, source_id, output_format="markdown")
        if result.returncode != 0 and "markdownify" in (result.stderr or "").lower():
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
            result, output_path = self._fulltext(notebook_id, source_id, output_format="text")

        if result.returncode != 0:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(_truncate(result.stderr or result.stdout or "NotebookLM fulltext failed", 500))
        try:
            return output_path.read_text(encoding="utf-8", errors="replace")
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _fulltext(
        self,
        notebook_id: str,
        source_id: str,
        *,
        output_format: str,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        with tempfile.NamedTemporaryFile(dir=self.work_dir, suffix=".md", delete=False) as tmp:
            output_path = Path(tmp.name)
        result = self.run(
            [
                "source",
                "fulltext",
                source_id,
                "--notebook",
                notebook_id,
                "--format",
                output_format,
                "--output",
                str(output_path),
            ],
            timeout=180,
        )
        return result, output_path

    def delete_notebook(self, notebook_id: str) -> None:
        self.run(["delete", notebook_id, "--yes"], timeout=120)


def _build_subqueries(query: str, config: NotebookLMSourceHarvestConfig) -> list[str]:
    templates = config.query_templates or ["{query}"]
    selected = templates[: config.query_count]
    return [template.format(query=query).strip() for template in selected]


def _wait_for_sources_after_timeout(
    client: NotebookLMCLI,
    notebook_id: str,
    config: NotebookLMSourceHarvestConfig,
) -> int:
    """Give NotebookLM imports a short grace period after a CLI timeout."""
    if config.post_timeout_source_poll_seconds <= 0:
        return 0

    deadline = time.monotonic() + config.post_timeout_source_poll_seconds
    while time.monotonic() < deadline:
        try:
            visible_sources = client.list_sources(
                notebook_id,
                timeout=config.source_inspection_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - depends on NotebookLM runtime state
            logger.debug("NotebookLM post-timeout source inspection failed: %s", exc)
            visible_sources = []

        ready_count = sum(1 for source in visible_sources if source.status == "ready")
        if ready_count > 0:
            return ready_count

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(config.post_timeout_source_poll_interval_seconds, remaining))

    return 0


def harvest_notebooklm_sources(query: str, config: NotebookLMSourceHarvestConfig) -> str:
    client = NotebookLMCLI(config)
    title_query = re.sub(r"\s+", " ", query).strip()[:80] or "Research"
    notebook_id = client.create_notebook(f"AIQ shallow harvest - {title_query}")
    subqueries = _build_subqueries(query, config)
    warnings: list[str] = []

    try:
        for subquery in subqueries:
            warning = client.add_research(notebook_id, subquery)
            if warning:
                warnings.append(f"Research query failed or partially completed: {warning}")
                if "timed out" in warning.lower():
                    ready_after_timeout = _wait_for_sources_after_timeout(client, notebook_id, config)
                    if ready_after_timeout:
                        warnings.append(
                            "NotebookLM sources became visible after timeout grace polling: "
                            f"{ready_after_timeout} ready sources."
                        )
            if config.early_stop_ready_sources > 0:
                try:
                    visible_sources = client.list_sources(notebook_id)
                except Exception as exc:  # pragma: no cover - depends on NotebookLM runtime state
                    logger.debug("NotebookLM source count after subquery failed: %s", exc)
                else:
                    ready_count = sum(1 for source in visible_sources if source.status == "ready")
                    if ready_count >= config.early_stop_ready_sources:
                        warnings.append(f"Stopped additional NotebookLM subqueries after {ready_count} ready sources.")
                        break

        sources = client.list_sources(notebook_id)
        if not config.include_generated_reports:
            sources = [source for source in sources if not _is_generated_report(source)]
        sources = _dedupe_sources(sources)[: config.max_sources]

        ready_sources = [source for source in sources if source.status == "ready" and source.source_id]
        for source in ready_sources[: config.max_fulltext_sources]:
            try:
                source.content = client.fulltext(notebook_id, source.source_id)
            except Exception as exc:  # pragma: no cover - depends on NotebookLM account state
                source.error = str(exc)
                logger.warning("NotebookLM fulltext failed for %s: %s", source.source_id, exc)

        return _render_sources(query, notebook_id, subqueries, sources, warnings, config.max_content_length)
    finally:
        if config.delete_notebook_after:
            try:
                client.delete_notebook(notebook_id)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("NotebookLM notebook cleanup failed for %s: %s", notebook_id, exc)


@register_function(config_type=NotebookLMSourceHarvestConfig)
async def notebooklm_source_harvest(tool_config: NotebookLMSourceHarvestConfig, builder: Builder):
    async def _harvest(query: str) -> str:
        """Use NotebookLM deep research to harvest web sources and extracted source text.

        Use this for shallow research when broad web discovery would otherwise require many
        search/fetch calls. Treat the returned sources as evidence candidates, not as a final
        report. Cite only returned source URLs.
        """
        try:
            return await asyncio.to_thread(harvest_notebooklm_sources, query, tool_config)
        except FileNotFoundError:
            return (
                "Error: NotebookLM source harvest is unavailable because the configured Python executable "
                "could not be found."
            )
        except subprocess.TimeoutExpired:
            return "Error: NotebookLM source harvest timed out before sources were ready."
        except Exception as exc:
            logger.exception("NotebookLM source harvest failed")
            return f"Error: NotebookLM source harvest failed - {exc}"

    yield FunctionInfo.from_fn(_harvest, description=_harvest.__doc__)

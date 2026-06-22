# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial post-research claim verification.

After the research dossier is compiled (and again as a post-run fallback), the
runtime selects high-risk claims from ``/shared/claim_table.json`` and asks a
narrow, non-thinking LLM to actively try to refute each claim from its own
evidence extracts. The result is written to
``/shared/verification_report.json`` and contradicted claims are downgraded in
the claim table so final synthesis can drop or hedge them.

Everything here is fail-open: any parse failure, timeout, or LLM error
degrades to a ``not_addressed`` verdict or a skipped verification pass. A
quality feature must never crash or hang a research job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage

from aiq_agent.common.claim_table import validate_claim_table_json
from aiq_agent.common.json_utils import extract_json

logger = logging.getLogger(__name__)

VERIFICATION_REPORT_PATH = "/shared/verification_report.json"

VALID_VERDICTS = ("supported", "partially_supported", "contradicted", "not_addressed")

# Claim types that carry the highest hallucination/precision risk. The claim
# schema uses "quantitative"; external callers may use "numeric"/"date".
HIGH_RISK_CLAIM_TYPES = frozenset({"quantitative", "numeric", "date", "superlative", "causal"})

_PERCENT_OR_MONEY_RE = re.compile(r"\d+(?:[.,]\d+)?\s?%|[$€£¥]\s?\d")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_SUPERLATIVE_RE = re.compile(
    r"\b(?:largest|biggest|smallest|first|only|fastest|slowest|highest|lowest|best|worst|"
    r"most|least|leading|top|record|unprecedented)\b",
    re.IGNORECASE,
)

_MAX_EVIDENCE_SNIPPETS = 6
_MAX_EVIDENCE_CHARS = 1400
_MAX_CLAIM_CHARS = 600
_MAX_NOTE_CHARS = 320

_FALSEY = {"0", "false", "no", "off"}


def verifier_enabled() -> bool:
    """Kill switch: AIQ_ADVERSARIAL_VERIFIER_ENABLED (default on)."""
    return os.getenv("AIQ_ADVERSARIAL_VERIFIER_ENABLED", "1").strip().lower() not in _FALSEY


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 1000) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, min(maximum, int(raw.strip())))
    except (TypeError, ValueError):
        return default


def verifier_max_claims() -> int:
    """Cap on verified claims per run: AIQ_VERIFIER_MAX_CLAIMS (default 24)."""
    return _env_int("AIQ_VERIFIER_MAX_CLAIMS", 24)


def verifier_concurrency() -> int:
    """Bounded LLM-call concurrency: AIQ_VERIFIER_CONCURRENCY (default 6)."""
    return _env_int("AIQ_VERIFIER_CONCURRENCY", 6, maximum=32)


def verifier_timeout_seconds() -> float:
    """Per-claim LLM call timeout: AIQ_VERIFIER_TIMEOUT_SECONDS (default 45)."""
    return float(_env_int("AIQ_VERIFIER_TIMEOUT_SECONDS", 45, minimum=5, maximum=600))


@dataclass
class SelectedClaim:
    """One high-risk claim plus the evidence extracts available to test it."""

    claim_id: str
    claim_text: str
    claim_type: str = "other"
    status: str = "unverified"
    evidence: list[str] = field(default_factory=list)
    risk_score: int = 0


@dataclass
class VerificationOutcome:
    """Result of one adversarial verification pass."""

    report: dict[str, Any]
    report_json: str
    updated_claim_table_json: str | None
    summary: dict[str, int]


def claim_risk_score(claim_text: str, claim_type: str = "other") -> int:
    """Deterministic risk score used to prioritize claims under the cap."""
    score = 0
    if str(claim_type or "").strip().lower() in HIGH_RISK_CLAIM_TYPES:
        score += 3
    text = str(claim_text or "")
    if _PERCENT_OR_MONEY_RE.search(text):
        score += 3
    if _SUPERLATIVE_RE.search(text):
        score += 2
    if _YEAR_RE.search(text):
        score += 1
    if _NUMBER_RE.search(text):
        score += 1
    return score


def is_high_risk_claim(claim_text: str, claim_type: str = "other") -> bool:
    """High-risk = risky claim type OR numeric/%/$/year/superlative wording."""
    if str(claim_type or "").strip().lower() in HIGH_RISK_CLAIM_TYPES:
        return True
    text = str(claim_text or "")
    return bool(
        _PERCENT_OR_MONEY_RE.search(text)
        or _SUPERLATIVE_RE.search(text)
        or _YEAR_RE.search(text)
        or _NUMBER_RE.search(text)
    )


def _evidence_by_claim_from_packet(evidence_packet_content: str | None) -> dict[str, list[str]]:
    """Map claim_id -> evidence snippets pulled from the evidence packet."""
    mapping: dict[str, list[str]] = {}
    if not evidence_packet_content:
        return mapping
    try:
        payload = json.loads(evidence_packet_content)
    except (TypeError, ValueError):
        return mapping
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list):
        return mapping
    for source in sources:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        for extract in source.get("extracts") or []:
            if not isinstance(extract, dict):
                continue
            text = str(extract.get("text") or "").strip()
            if not text:
                continue
            snippet = f"[{url}] {text}" if url else text
            for claim_id in extract.get("claim_ids") or []:
                claim_key = str(claim_id).strip()
                if claim_key:
                    mapping.setdefault(claim_key, []).append(snippet)
    return mapping


def _claim_evidence_snippets(claim: Any) -> list[str]:
    """Pull evidence extracts off an AtomicClaim or ClaimResolution."""
    snippets: list[str] = []
    for evidence in getattr(claim, "evidence", None) or []:
        text = str(getattr(evidence, "extract", "") or "").strip()
        url = str(getattr(evidence, "source_url", "") or "").strip()
        if text:
            snippets.append(f"[{url}] {text}" if url else text)
    extract = str(getattr(claim, "source_extract", "") or "").strip()
    if extract:
        url = str(getattr(claim, "source_url", "") or "").strip()
        snippets.append(f"[{url}] {extract}" if url else extract)
    return snippets


def select_high_risk_claims(
    claim_table_content: str,
    *,
    evidence_packet_content: str | None = None,
    max_claims: int | None = None,
) -> list[SelectedClaim]:
    """Select high-risk claims (numeric/date/superlative/causal) with evidence.

    Claims are prioritized by deterministic risk score and capped at
    ``max_claims`` (default AIQ_VERIFIER_MAX_CLAIMS). Fail-open: invalid claim
    tables select nothing.
    """
    cap = max_claims if max_claims is not None else verifier_max_claims()
    table, errors = validate_claim_table_json(claim_table_content or "")
    if table is None:
        if errors:
            logger.debug("Adversarial verifier: claim table invalid: %s", "; ".join(errors))
        return []

    packet_evidence = _evidence_by_claim_from_packet(evidence_packet_content)

    selected: list[SelectedClaim] = []
    seen_ids: set[str] = set()
    for index, claim in enumerate(table.all_claims()):
        claim_id = str(getattr(claim, "claim_id", "") or f"claim-{index}")
        if claim_id in seen_ids:
            continue
        claim_text = str(getattr(claim, "claim_text", None) or getattr(claim, "claim", "") or "").strip()
        claim_type = str(getattr(claim, "claim_type", "other") or "other")
        if not claim_text or not is_high_risk_claim(claim_text, claim_type):
            continue
        evidence = _claim_evidence_snippets(claim)
        evidence.extend(packet_evidence.get(claim_id, []))
        deduped: list[str] = []
        seen_snippets: set[str] = set()
        for snippet in evidence:
            key = snippet[:200]
            if key not in seen_snippets:
                seen_snippets.add(key)
                deduped.append(snippet[:_MAX_EVIDENCE_CHARS])
        seen_ids.add(claim_id)
        selected.append(
            SelectedClaim(
                claim_id=claim_id,
                claim_text=claim_text[:_MAX_CLAIM_CHARS],
                claim_type=claim_type,
                status=str(getattr(claim, "status", "unverified") or "unverified"),
                evidence=deduped[:_MAX_EVIDENCE_SNIPPETS],
                risk_score=claim_risk_score(claim_text, claim_type),
            )
        )

    selected.sort(key=lambda claim: (-claim.risk_score, claim.claim_id))
    return selected[: max(0, cap)]


def _verdict_prompt(claim: SelectedClaim) -> str:
    evidence_block = "\n".join(f"{index + 1}. {snippet}" for index, snippet in enumerate(claim.evidence))
    return (
        "You are an adversarial fact-checker. Your job is to actively try to REFUTE the claim "
        "below using ONLY the evidence extracts provided. Do not use outside knowledge. "
        "Look for mismatched numbers, dates, units, scope, entities, or direction of effect.\n\n"
        f"CLAIM ({claim.claim_type}):\n{claim.claim_text}\n\n"
        f"EVIDENCE EXTRACTS:\n{evidence_block or '(none)'}\n\n"
        "Verdict definitions:\n"
        '- "supported": the evidence clearly states the claim (numbers/dates/entities match).\n'
        '- "partially_supported": the evidence is related and consistent but does not fully state the claim.\n'
        '- "contradicted": the evidence states something incompatible with the claim.\n'
        '- "not_addressed": the evidence does not speak to the claim.\n\n'
        "Respond with STRICT JSON only, no prose, exactly this shape:\n"
        '{"verdict": "supported"|"partially_supported"|"contradicted"|"not_addressed", '
        '"confidence": 0.0-1.0, "note": "<short reason, name the contradiction when present>"}'
    )


def _coerce_message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return str(content or "")


def parse_verdict_payload(raw_text: str) -> dict[str, Any]:
    """Parse one strict-JSON verdict response defensively.

    Any unparseable or out-of-contract response degrades to ``not_addressed``.
    """
    payload = extract_json(raw_text or "") or {}
    verdict = str(payload.get("verdict") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if verdict not in VALID_VERDICTS:
        return {"verdict": "not_addressed", "confidence": 0.0, "note": "unparseable_verifier_response"}
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    note = str(payload.get("note") or "").strip()[:_MAX_NOTE_CHARS]
    return {"verdict": verdict, "confidence": confidence, "note": note}


async def verify_claim(llm: Any, claim: SelectedClaim, *, timeout: float | None = None) -> dict[str, Any]:
    """Run one narrow adversarial verification call. Failure → not_addressed."""
    base = {
        "claim_id": claim.claim_id,
        "claim_text": claim.claim_text,
        "claim_type": claim.claim_type,
    }
    if not claim.evidence:
        return {**base, "verdict": "not_addressed", "confidence": 0.0, "note": "no_evidence_extracts"}

    # Wave 2 W2.2 — Redis-backed verdict cache. Keyed by (model, claim, evidence)
    # so a model upgrade invalidates old verdicts automatically. Fail-open: if
    # the cache is unreachable, we still make the LLM call.
    cache_key: str | None = None
    try:
        from aiq_agent.common.llm_cache import LLMResponseCache

        cache = LLMResponseCache.instance()
        if cache.enabled:
            model_id = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
            cache_key = LLMResponseCache.make_key({
                "role": "verifier",
                "model": str(model_id),
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "evidence": tuple(claim.evidence),
            })
            cached = cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "aiq.metrics llm_cache_hit role=verifier claim_id=%s model=%s",
                    claim.claim_id,
                    model_id,
                )
                return {**base, **cached}
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        logger.debug("Verifier cache lookup failed for %s: %s", claim.claim_id, exc)

    try:
        effective_timeout = timeout if timeout is not None else verifier_timeout_seconds()
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=_verdict_prompt(claim))]),
            timeout=effective_timeout,
        )
        parsed = parse_verdict_payload(_coerce_message_text(response))
        # Write-through: only cache parseable verdicts so we never poison the
        # cache with malformed responses.
        if cache_key is not None:
            try:
                from aiq_agent.common.llm_cache import LLMResponseCache

                LLMResponseCache.instance().set(cache_key, parsed)
                logger.info(
                    "aiq.metrics llm_cache_miss role=verifier claim_id=%s",
                    claim.claim_id,
                )
            except Exception:  # noqa: BLE001 - cache write is best-effort
                logger.debug("Verifier cache write failed for %s", claim.claim_id, exc_info=True)
        return {**base, **parsed}
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.debug("Adversarial verifier call failed for %s: %s", claim.claim_id, exc)
        return {**base, "verdict": "not_addressed", "confidence": 0.0, "note": f"verifier_call_failed: {exc}"[:200]}


async def verify_claims(
    llm: Any,
    claims: list[SelectedClaim],
    *,
    concurrency: int | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Verify claims with bounded asyncio concurrency."""
    semaphore = asyncio.Semaphore(concurrency if concurrency is not None else verifier_concurrency())

    async def _bounded(claim: SelectedClaim) -> dict[str, Any]:
        async with semaphore:
            return await verify_claim(llm, claim, timeout=timeout)

    return list(await asyncio.gather(*(_bounded(claim) for claim in claims)))


def apply_verdicts_to_claim_table(claim_table_content: str, verdicts: list[dict[str, Any]]) -> str | None:
    """Annotate the claim table with verdicts; downgrade contradicted claims.

    Contradicted claims get ``status: "unverified"`` plus a visible
    ``verification_verdict: "contradicted"`` flag so synthesis must drop or
    hedge them. Returns updated JSON, or None when nothing changed / on error.
    """
    try:
        payload = json.loads(claim_table_content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    by_id = {str(v.get("claim_id") or ""): v for v in verdicts if v.get("claim_id")}
    if not by_id:
        return None

    changed = False
    for key in ("entries", "atomic_claims", "claims"):
        claim_list = payload.get(key)
        if not isinstance(claim_list, list):
            continue
        for claim in claim_list:
            if not isinstance(claim, dict):
                continue
            verdict = by_id.get(str(claim.get("claim_id") or ""))
            if verdict is None:
                continue
            claim["verification_verdict"] = verdict.get("verdict")
            claim["verification_confidence"] = verdict.get("confidence")
            if verdict.get("note"):
                claim["verification_note"] = verdict.get("note")
            changed = True
            if verdict.get("verdict") == "contradicted":
                claim["status"] = "unverified"
                reason = f"contradicted_by_adversarial_verification: {verdict.get('note') or 'evidence conflict'}"
                claim.setdefault("downgrade_reason", reason)
                claim.setdefault("notes", reason)
                claim["hedge_required"] = True
                claim.setdefault("hedge_phrase", "Sources conflict on this point")
    if not changed:
        return None
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def build_verification_report(verdicts: list[dict[str, Any]], *, tier: str | None = None) -> dict[str, Any]:
    """Build the /shared/verification_report.json payload."""
    summary = {key: 0 for key in VALID_VERDICTS}
    for verdict in verdicts:
        key = str(verdict.get("verdict") or "not_addressed")
        summary[key] = summary.get(key, 0) + 1
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "tier": tier,
        "summary": summary,
        "claims": [
            {
                "claim_id": verdict.get("claim_id"),
                "claim_text": verdict.get("claim_text"),
                "claim_type": verdict.get("claim_type"),
                "verdict": verdict.get("verdict"),
                "confidence": verdict.get("confidence"),
                "note": verdict.get("note"),
            }
            for verdict in verdicts
        ],
    }


async def run_adversarial_verification(
    *,
    llm: Any,
    claim_table_content: str,
    evidence_packet_content: str | None = None,
    tier: str | None = None,
    max_claims: int | None = None,
    concurrency: int | None = None,
    timeout: float | None = None,
) -> VerificationOutcome | None:
    """Run the full adversarial verification pass. Fail-open: returns None when
    disabled, when the tier is shallow, when no LLM is available, or when no
    high-risk claim has evidence extracts to test.
    """
    if not verifier_enabled():
        logger.info("Adversarial verifier disabled via AIQ_ADVERSARIAL_VERIFIER_ENABLED")
        return None
    if str(tier or "").strip().lower() == "shallow":
        logger.debug("Adversarial verifier skipped for shallow tier")
        return None
    if llm is None:
        logger.warning("Adversarial verifier skipped: no LLM available")
        return None

    try:
        claims = select_high_risk_claims(
            claim_table_content,
            evidence_packet_content=evidence_packet_content,
            max_claims=max_claims,
        )
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.warning("Adversarial verifier claim selection failed", exc_info=True)
        return None
    if not claims:
        logger.debug("Adversarial verifier: no high-risk claims selected")
        return None
    if not any(claim.evidence for claim in claims):
        logger.debug("Adversarial verifier skipped: no evidence extracts available")
        return None

    try:
        verdicts = await verify_claims(llm, claims, concurrency=concurrency, timeout=timeout)
    except Exception:  # noqa: BLE001 - fail-open by design
        logger.warning("Adversarial verifier pass failed", exc_info=True)
        return None

    report = build_verification_report(verdicts, tier=tier)
    updated_claim_table = apply_verdicts_to_claim_table(claim_table_content, verdicts)
    logger.info(
        "Adversarial verification complete: %s",
        json.dumps(report["summary"], sort_keys=True),
    )
    return VerificationOutcome(
        report=report,
        report_json=json.dumps(report, indent=2, ensure_ascii=False),
        updated_claim_table_json=updated_claim_table,
        summary=dict(report["summary"]),
    )

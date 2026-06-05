# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic fact-alignment checks for final research reports.

These checks are intentionally conservative. They do not try to prove every
sentence true; they catch high-cost failures that are cheap to detect:
duplicate reference numbers, arXiv ID mismatches, and precise numeric claims
whose cited evidence extract does not contain the reported value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_REFERENCE_SECTION_RE = re.compile(
    r"^(?:#{2,3}\s+(?:(?:\d+|[A-Z])[\).:-]?\s+)?(?:Sources|References)|Reference\s+List|\*\*References:?\*\*)",
    re.MULTILINE | re.IGNORECASE,
)
_CITATION_LINE_RE = re.compile(r"^\s*[-*]?\s*\[(\d+)\]\s*(.+)$", re.MULTILINE)
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_URL_RE = re.compile(r"https?://\S+")
_ARXIV_ID_RE = re.compile(r"\b(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(
    r"(?<!\[)\b(?:\d+(?:\.\d+)?\s?%|\d+\.\d+|\d+(?:\.\d+)?\s?(?:x|times|million|billion|trillion|k)\b|\$\s?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SOURCE_COUNT_CLAIM_RE = re.compile(
    r"\b(\d[\d,]*)\+?\s+(?:verified|vetted|supporting|cited)\s+sources\b",
    re.IGNORECASE,
)
_POSITIVE_DIRECTION_RE = re.compile(
    r"\b(?:improv(?:e|es|ed|ing)|help(?:s|ed)?|benefit(?:s|ed)?|boost(?:s|ed)?|"
    r"lower(?:s|ed)?\s+(?:runtime|costs?|tokens?)|fewer\s+(?:tokens?|errors?)|"
    r"reduce(?:s|d)?\s+(?:runtime|costs?|tokens?|latency)|higher\s+(?:success|accuracy|quality)|"
    r"faster|more\s+effective)\b",
    re.IGNORECASE,
)
_NEGATIVE_DIRECTION_RE = re.compile(
    r"\b(?:hurt(?:s|ing)?|harm(?:s|ed|ing)?|worse|reduce(?:s|d)?\s+(?:success|accuracy|quality)|"
    r"lower\s+(?:success|accuracy|quality)|increase(?:s|d)?\s+(?:costs?|runtime|tokens?|latency)|"
    r"higher\s+(?:costs?|runtime|tokens?|latency)|do(?:es)?\s+not\s+help|mixed-to-negative|"
    r"fail(?:s|ed)?\s+to\s+improve)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FactAuditIssue:
    """One deterministic report integrity issue."""

    severity: str
    code: str
    message: str
    sentence: str = ""
    citation_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class FactAuditResult:
    """Report fact-alignment audit result."""

    hard_failed: bool
    issues: tuple[FactAuditIssue, ...] = ()

    @property
    def hard_issues(self) -> tuple[FactAuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[FactAuditIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def evaluate_report_fact_audit(report_text: str, evidence_packet_content: str | None = None) -> FactAuditResult:
    """Evaluate final-report fact integrity without blocking broad synthesis.

    Hard failures are reserved for cases where the report is actively
    misleading: duplicate reference numbers, arXiv IDs that disagree with the
    cited URL, or precise numeric claims contradicted by available cited
    extracts. Missing extracts are warnings unless the sentence contains a
    high-precision metric such as a correlation/range/percentage.
    """
    if not report_text:
        return FactAuditResult(hard_failed=False)

    ref_match = _REFERENCE_SECTION_RE.search(report_text)
    if not ref_match:
        return FactAuditResult(hard_failed=False)

    body = report_text[: ref_match.start()]
    ref_section = report_text[ref_match.start() :]
    references = _parse_references(ref_section)
    evidence_by_url = _parse_evidence_packet(evidence_packet_content)
    issues: list[FactAuditIssue] = []

    issues.extend(_duplicate_reference_number_issues(references))
    issues.extend(_sentence_fact_issues(body, references, evidence_by_url))
    issues.extend(_source_count_claim_issues(body, references, evidence_by_url))

    return FactAuditResult(
        hard_failed=any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
    )


def fact_audit_note(audit: FactAuditResult, *, max_items: int = 4) -> str:
    """Return a reader-facing note for residual fact-audit issues."""
    if not audit.issues:
        return ""

    lines = [
        "## Source Accuracy Notes",
        "",
        "Some high-precision claims could not be fully reconciled against the captured source extracts. "
        "The report preserves the best available synthesis, but the following items should be treated with caution:",
    ]
    for issue in audit.issues[:max_items]:
        cited = f" citations {list(issue.citation_numbers)}" if issue.citation_numbers else ""
        lines.append(f"- {issue.message}{cited}.")
    return "\n".join(lines).strip()


def _parse_references(ref_section: str) -> dict[int, list[str]]:
    references: dict[int, list[str]] = {}
    for line_match in _CITATION_LINE_RE.finditer(ref_section):
        num = int(line_match.group(1))
        url_match = _URL_RE.search(line_match.group(2))
        if not url_match:
            continue
        url = url_match.group(0).rstrip(".,;)")
        references.setdefault(num, []).append(url)
    return references


def _duplicate_reference_number_issues(references: dict[int, list[str]]) -> list[FactAuditIssue]:
    issues: list[FactAuditIssue] = []
    for num, urls in references.items():
        unique_urls = list(dict.fromkeys(_normalize_url(url) for url in urls))
        if len(unique_urls) > 1:
            issues.append(
                FactAuditIssue(
                    severity="error",
                    code="duplicate_reference_number",
                    message=f"Reference [{num}] maps to multiple different URLs",
                    citation_numbers=(num,),
                )
            )
    return issues


def _sentence_fact_issues(
    body: str,
    references: dict[int, list[str]],
    evidence_by_url: dict[str, str],
) -> list[FactAuditIssue]:
    issues: list[FactAuditIssue] = []
    for raw_sentence in _SENTENCE_SPLIT_RE.split(body):
        sentence = " ".join(raw_sentence.split())
        if not sentence:
            continue
        cited_nums = tuple(int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(sentence))
        if not cited_nums:
            continue

        arxiv_ids = _arxiv_ids(sentence)
        numeric_tokens = _numeric_tokens(sentence)
        has_directional_claim = bool(_POSITIVE_DIRECTION_RE.search(sentence) or _NEGATIVE_DIRECTION_RE.search(sentence))
        if not arxiv_ids and not numeric_tokens and not has_directional_claim:
            continue

        cited_urls = [url for num in cited_nums for url in references.get(num, [])]
        evidence_texts = [_evidence_for_url(url, evidence_by_url) for url in cited_urls]
        evidence_texts = [text for text in evidence_texts if text]
        if evidence_texts and has_directional_claim:
            direction_issue = _claim_direction_issue(sentence, cited_nums, "\n".join(evidence_texts))
            if direction_issue is not None:
                issues.append(direction_issue)

        if arxiv_ids:
            issues.extend(_arxiv_mismatch_issues(sentence, cited_nums, cited_urls, arxiv_ids))

        if numeric_tokens:
            if evidence_texts:
                combined = "\n".join(evidence_texts)
                missing = [token for token in numeric_tokens if not _number_present(token, combined)]
                if missing:
                    issues.append(
                        FactAuditIssue(
                            severity="error",
                            code="numeric_value_not_in_cited_evidence",
                            message=f"Numeric value(s) {', '.join(missing)} were not found in cited evidence extracts",
                            sentence=sentence,
                            citation_numbers=cited_nums,
                        )
                    )
            elif _requires_extract_support(sentence, numeric_tokens):
                issues.append(
                    FactAuditIssue(
                        severity="warning",
                        code="numeric_claim_missing_extract",
                        message="High-precision numeric claim lacks captured cited-source extract support",
                        sentence=sentence,
                        citation_numbers=cited_nums,
                    )
                )
    return issues


def _source_count_claim_issues(
    body: str,
    references: dict[int, list[str]],
    evidence_by_url: dict[str, str],
) -> list[FactAuditIssue]:
    reference_count = len({num for num, urls in references.items() if urls})
    evidence_count = len(evidence_by_url)
    supportable_count = max(reference_count, evidence_count)
    issues: list[FactAuditIssue] = []
    for match in _SOURCE_COUNT_CLAIM_RE.finditer(body):
        claimed = int(match.group(1).replace(",", ""))
        if supportable_count and claimed > supportable_count:
            issues.append(
                FactAuditIssue(
                    severity="error",
                    code="inflated_verified_source_count",
                    message=(
                        f"Report claims {claimed} verified/supporting sources, "
                        f"but only {supportable_count} cited or extracted sources are auditable"
                    ),
                )
            )
    return issues


def _parse_evidence_packet(content: str | None) -> dict[str, str]:
    if not content:
        return {}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list):
        return {}
    evidence: dict[str, str] = {}
    for item in sources:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not url:
            continue
        parts: list[str] = []
        for extract in item.get("extracts") or []:
            if isinstance(extract, dict) and extract.get("text"):
                parts.append(str(extract["text"]))
        if item.get("full_text"):
            parts.append(str(item["full_text"]))
        if parts:
            evidence[_normalize_url(url)] = "\n".join(parts)
    return evidence


def _evidence_for_url(url: str, evidence_by_url: dict[str, str]) -> str:
    normalized = _normalize_url(url)
    if normalized in evidence_by_url:
        return evidence_by_url[normalized]
    host_path = _host_path(url)
    for evidence_url, text in evidence_by_url.items():
        if _host_path(evidence_url) == host_path:
            return text
    return ""


def _claim_direction_issue(
    sentence: str,
    cited_nums: tuple[int, ...],
    evidence_text: str,
) -> FactAuditIssue | None:
    sentence_positive = bool(_POSITIVE_DIRECTION_RE.search(sentence))
    sentence_negative = bool(_NEGATIVE_DIRECTION_RE.search(sentence))
    evidence_positive = bool(_POSITIVE_DIRECTION_RE.search(evidence_text))
    evidence_negative = bool(_NEGATIVE_DIRECTION_RE.search(evidence_text))
    if sentence_positive and evidence_negative and not evidence_positive:
        return FactAuditIssue(
            severity="error",
            code="claim_direction_contradicted_by_evidence",
            message="Cited evidence appears directionally negative while the report states a positive effect",
            sentence=sentence,
            citation_numbers=cited_nums,
        )
    if sentence_negative and evidence_positive and not evidence_negative:
        return FactAuditIssue(
            severity="error",
            code="claim_direction_contradicted_by_evidence",
            message="Cited evidence appears directionally positive while the report states a negative effect",
            sentence=sentence,
            citation_numbers=cited_nums,
        )
    return None


def _numeric_tokens(sentence: str) -> list[str]:
    tokens: list[str] = []
    for match in _NUMERIC_RE.finditer(sentence):
        token = match.group(0).strip()
        normalized = _normalize_number_token(token)
        if not normalized or _YEAR_RE.match(normalized):
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens


def _arxiv_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in _ARXIV_ID_RE.finditer(text):
        value = match.group(1)
        if value not in ids:
            ids.append(value)
    return ids


def _arxiv_mismatch_issues(
    sentence: str,
    cited_nums: tuple[int, ...],
    cited_urls: list[str],
    arxiv_ids: list[str],
) -> list[FactAuditIssue]:
    issues: list[FactAuditIssue] = []
    cited_arxiv_ids = [arxiv_id for url in cited_urls for arxiv_id in _arxiv_ids(url)]
    if cited_arxiv_ids and not any(arxiv_id in cited_arxiv_ids for arxiv_id in arxiv_ids):
        issues.append(
            FactAuditIssue(
                severity="error",
                code="arxiv_id_citation_mismatch",
                message=(
                    f"Sentence names arXiv ID(s) {', '.join(arxiv_ids)} but cited URL contains "
                    f"{', '.join(cited_arxiv_ids)}"
                ),
                sentence=sentence,
                citation_numbers=cited_nums,
            )
        )
    return issues


def _requires_extract_support(sentence: str, numeric_tokens: list[str]) -> bool:
    lowered = sentence.lower()
    return bool(
        any(
            "%" in token or "." in token or token.startswith("$") or token.endswith("x") or " times" in token
            for token in numeric_tokens
        )
        or any(term in lowered for term in ("spearman", "correlation", "benchmark", "leaderboard", "sample"))
    )


def _number_present(token: str, evidence: str) -> bool:
    normalized_evidence = evidence.lower().replace(",", "")
    variants = {token.lower(), token.lower().replace(" ", "")}
    if token.endswith("%"):
        variants.add(token[:-1].strip() + " percent")
    if token.startswith("$"):
        variants.add(token[1:].strip())
    return any(variant and variant in normalized_evidence.replace(" ", "") for variant in variants) or any(
        variant and variant in normalized_evidence for variant in variants
    )


def _normalize_number_token(token: str) -> str:
    token = token.lower().replace(",", "").strip()
    token = re.sub(r"\s+", " ", token)
    return token


def _normalize_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{host}{path}"


def _host_path(url: str) -> str:
    parsed = urlparse(url)
    return f"{(parsed.hostname or '').lower().removeprefix('www.')}{parsed.path.rstrip('/') or '/'}"

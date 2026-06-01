# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared checks for distinguishing real reports from terminal error text."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.source_classification import AUTHORITATIVE_CLASSES
from aiq_agent.common.source_classification import WEAK_DERIVATIVE_CLASSES
from aiq_agent.common.source_classification import classify_source
from aiq_agent.common.source_classification import normalize_source_class

_CONTEXT_MARKERS = ("## Clarification Context", "## Resume Context", "**Approved Research Plan**")
_WORD_RE = re.compile(r"[a-z][a-z0-9-]+", re.IGNORECASE)
_REFERENCE_SECTION_RE = re.compile(
    r"^(?:#{2,3}\s+(?:Sources|References)|Reference\s+List|\*\*References:?\*\*)",
    re.MULTILINE | re.IGNORECASE,
)
_CITATION_LINE_RE = re.compile(r"^\s*[-*]?\s*\[(\d+)\]\s*(.+)$", re.MULTILINE)
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_URL_RE = re.compile(r"https?://\S+")
_NUMERIC_CLAIM_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s?%|\$\s?\d|(?:\b\d+(?:\.\d+)?\s?(?:million|billion|trillion|x|times)\b))",
    re.IGNORECASE,
)
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "answer",
    "analysis",
    "background",
    "based",
    "before",
    "between",
    "brief",
    "build",
    "case",
    "compare",
    "conclusion",
    "context",
    "criteria",
    "current",
    "deeper",
    "detail",
    "details",
    "direct",
    "discussion",
    "education",
    "evidence",
    "explain",
    "findings",
    "focus",
    "focused",
    "for",
    "from",
    "given",
    "guide",
    "how",
    "important",
    "in",
    "include",
    "introduction",
    "investigate",
    "latest",
    "lesson",
    "more",
    "notes",
    "overview",
    "practical",
    "provide",
    "question",
    "recommendations",
    "research",
    "report",
    "results",
    "risks",
    "section",
    "sections",
    "summary",
    "the",
    "this",
    "through",
    "topic",
    "trade",
    "tradeoffs",
    "use",
    "using",
    "what",
    "with",
    "write",
}
_DRIFT_MARKERS = (
    "alternative pathways",
    "alternate ways",
    "alternatives to university",
    "alternatives to college",
    "college alternatives",
    "career pathways",
    "career alternative",
    "career alternatives",
    "career path",
    "get to your career",
    "without a degree",
    "no degree",
    "vocational education",
    "vocational training",
    "apprenticeship",
    "trade school",
    "jobs without a degree",
    "career first",
)


@dataclass(frozen=True)
class SourceQualityResult:
    """Deterministic quality signal for final report source usage."""

    passed: bool
    reason: str
    distinct_cited_domains: int = 0
    top_domain: str = ""
    top_domain_share: float = 0.0
    blog_share: float = 0.0
    authoritative_citation_count: int = 0
    numeric_claim_count: int = 0
    weak_numeric_claim_count: int = 0


_MODEL_CALL_FAILED_RE = re.compile(
    r"^\s*model call failed after \d+ attempts? with [a-z_]*error\b",
    re.IGNORECASE,
)
_REPETITION_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]*", re.IGNORECASE)


def _meaningful_token_matches(content: str):
    return [match for match in _REPETITION_TOKEN_RE.finditer(content) if len(match.group(0)) >= 3]


def repeated_token_run(content: str | None, *, min_run: int = 24) -> tuple[str, int] | None:
    """Return the first suspicious same-token run in generated text.

    Long MiniMax streams can occasionally degenerate into loops such as
    ``actors actors actors ...``. The threshold is intentionally high so normal
    rhetoric, tables, and repeated headings are not flagged.
    """
    if not content or min_run <= 1:
        return None

    last_token = ""
    run_length = 0
    for match in _REPETITION_TOKEN_RE.finditer(content.lower()):
        token = match.group(0)
        if len(token) < 3:
            last_token = ""
            run_length = 0
            continue
        if token == last_token:
            run_length += 1
        else:
            last_token = token
            run_length = 1
        if run_length >= min_run:
            return token, run_length
    return None


def low_diversity_token_loop(
    content: str | None,
    *,
    min_tokens: int = 40,
    max_unique_tokens: int = 5,
    min_top_token_coverage: float = 0.85,
) -> tuple[str, int] | None:
    """Return true for alternating low-diversity loops like ``puppet specialists``."""
    window = _low_diversity_loop_window(
        content,
        min_tokens=min_tokens,
        max_unique_tokens=max_unique_tokens,
        min_top_token_coverage=min_top_token_coverage,
    )
    if not window:
        return None
    label, _start, _end = window
    return label, min_tokens


def _low_diversity_loop_window(
    content: str | None,
    *,
    min_tokens: int = 40,
    max_unique_tokens: int = 5,
    min_top_token_coverage: float = 0.85,
) -> tuple[str, int, int] | None:
    if not content or min_tokens <= 1:
        return None

    matches = _meaningful_token_matches(content.lower())
    if len(matches) < min_tokens:
        return None

    for end_idx in range(min_tokens, len(matches) + 1):
        window = matches[end_idx - min_tokens : end_idx]
        counts = Counter(match.group(0) for match in window)
        if len(counts) > max_unique_tokens:
            continue
        if counts[window[0].group(0)] < 2:
            continue
        top_total = sum(count for _token, count in counts.most_common(3))
        if top_total / min_tokens >= min_top_token_coverage:
            label = ",".join(token for token, count in counts.most_common(3) if count > 1)
            return label, window[0].start(), window[-1].end()
    return None


def strip_degenerate_repetition(content: str | None, *, min_run: int = 10) -> str:
    """Remove obvious repeated-token tails from streamed UI chunks.

    This is a display/event-stream guardrail. It keeps useful prefix text from a
    chunk but suppresses the pathological tail. Full model-call protection lives
    in the LLM wrapper, which aborts and lets the agent retry.
    """
    if not content:
        return ""

    matches = list(_REPETITION_TOKEN_RE.finditer(content))
    if len(matches) < min_run:
        return content

    run_start = 0
    run_length = 0
    last_token = ""
    for idx, match in enumerate(matches):
        token = match.group(0).lower()
        if len(token) < 3:
            run_start = idx + 1
            run_length = 0
            last_token = ""
            continue
        if token == last_token:
            run_length += 1
        else:
            run_start = idx
            run_length = 1
            last_token = token
        if run_length < min_run:
            continue

        prefix = content[: matches[run_start].start()].rstrip()
        suffix = content[match.end() :].strip()
        suffix_tokens = [m.group(0).lower() for m in _REPETITION_TOKEN_RE.finditer(suffix)]

        if not suffix_tokens or all(t == token for t in suffix_tokens):
            return prefix

        # If the entire chunk is dominated by a repeated token, it is safer to
        # suppress it than to render noise into the thinking log.
        token_count = len(matches)
        run_to_end = 1
        for later in matches[idx + 1 :]:
            if later.group(0).lower() == token:
                run_to_end += 1
        if (run_length + run_to_end) / max(token_count, 1) >= 0.75:
            return prefix

    low_diversity_window = _low_diversity_loop_window(content, min_tokens=max(min_run * 4, 40))
    if low_diversity_window:
        _label, start, _end = low_diversity_window
        return content[:start].rstrip()

    return content


def has_degenerate_repetition(content: str | None, *, min_run: int = 24) -> bool:
    """Return true when text contains an obvious model repetition loop."""
    return (
        repeated_token_run(content, min_run=min_run) is not None
        or low_diversity_token_loop(
            content,
            min_tokens=max(min_run * 2, 40),
        )
        is not None
    )


def is_model_failure_report(content: str | None) -> bool:
    """Return true when content is an infrastructure failure, not a report."""
    if not content:
        return False

    text = content.strip()
    lowered = text.lower()
    return (
        bool(_MODEL_CALL_FAILED_RE.search(text))
        or lowered == "research failed to produce a report."
        or lowered.startswith("research failed: no sources were captured")
        or has_degenerate_repetition(text)
    )


def evaluate_report_source_quality(
    report_text: str | None,
    registry: SourceRegistry,
    *,
    min_distinct_cited_domains: int = 4,
    max_top_domain_share: float = 0.60,
    max_blog_share: float = 0.65,
) -> SourceQualityResult:
    """Evaluate whether a deep report uses sources broadly enough.

    This is intentionally a coarse gate. It catches the high-cost failure mode
    where a long report cites many claims to one derivative blog while the
    references list merely creates the appearance of breadth.
    """
    if not report_text:
        return SourceQualityResult(False, "empty_report")
    if not registry.all_sources():
        return SourceQualityResult(False, "no_registered_sources")

    ref_match = _REFERENCE_SECTION_RE.search(report_text)
    if not ref_match:
        return SourceQualityResult(False, "missing_references_section")

    body = report_text[: ref_match.start()]
    ref_section = report_text[ref_match.start() :]
    citation_sources = _citation_sources_from_references(ref_section, registry)
    if not citation_sources:
        return SourceQualityResult(False, "no_verifiable_reference_sources")

    inline_numbers = [int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(body)]
    cited_entries = [citation_sources[num] for num in inline_numbers if num in citation_sources]
    if len(cited_entries) < 4:
        return SourceQualityResult(False, f"too_few_body_citations:{len(cited_entries)}")

    domains = [_hostname(entry.url) for entry in cited_entries if entry.url]
    domain_counts = Counter(domain for domain in domains if domain)
    distinct_domains = len(domain_counts)
    top_domain = ""
    top_domain_share = 0.0
    if domain_counts:
        top_domain, top_count = domain_counts.most_common(1)[0]
        top_domain_share = top_count / len(cited_entries)

    class_counts = Counter(_entry_source_class(entry) for entry in cited_entries)
    blog_share = sum(class_counts[source_class] for source_class in WEAK_DERIVATIVE_CLASSES) / len(cited_entries)
    authoritative_count = sum(class_counts[source_class] for source_class in AUTHORITATIVE_CLASSES)
    numeric_claim_count, weak_numeric_claim_count = _numeric_claim_counts(body, citation_sources)

    available_domains = len({_hostname(source.url) for source in registry.all_sources() if source.url})
    if available_domains >= min_distinct_cited_domains and distinct_domains < min_distinct_cited_domains:
        return SourceQualityResult(
            False,
            f"too_few_distinct_cited_domains:{distinct_domains}",
            distinct_domains,
            top_domain,
            top_domain_share,
            blog_share,
            authoritative_count,
            numeric_claim_count,
            weak_numeric_claim_count,
        )

    if len(cited_entries) >= 8 and top_domain_share > max_top_domain_share:
        return SourceQualityResult(
            False,
            f"single_source_dependency:{top_domain}:{top_domain_share:.2f}",
            distinct_domains,
            top_domain,
            top_domain_share,
            blog_share,
            authoritative_count,
            numeric_claim_count,
            weak_numeric_claim_count,
        )

    if len(cited_entries) >= 8 and blog_share > max_blog_share and authoritative_count < 2:
        return SourceQualityResult(
            False,
            f"blog_source_dominance:{blog_share:.2f}",
            distinct_domains,
            top_domain,
            top_domain_share,
            blog_share,
            authoritative_count,
            numeric_claim_count,
            weak_numeric_claim_count,
        )

    if numeric_claim_count >= 3 and weak_numeric_claim_count / numeric_claim_count > 0.5:
        return SourceQualityResult(
            False,
            f"numeric_claims_need_primary_sources:{weak_numeric_claim_count}/{numeric_claim_count}",
            distinct_domains,
            top_domain,
            top_domain_share,
            blog_share,
            authoritative_count,
            numeric_claim_count,
            weak_numeric_claim_count,
        )

    return SourceQualityResult(
        True,
        "source_quality_ok",
        distinct_domains,
        top_domain,
        top_domain_share,
        blog_share,
        authoritative_count,
        numeric_claim_count,
        weak_numeric_claim_count,
    )


def _citation_sources_from_references(ref_section: str, registry: SourceRegistry):
    sources = {}
    for line_match in _CITATION_LINE_RE.finditer(ref_section):
        url_match = _URL_RE.search(line_match.group(2))
        if not url_match:
            continue
        canonical_url = registry.resolve_url(url_match.group(0).rstrip(".,;)"))
        if not canonical_url:
            continue
        entry = _registry_entry_for_url(registry, canonical_url)
        if entry is not None:
            sources[int(line_match.group(1))] = entry
    return sources


def _registry_entry_for_url(registry: SourceRegistry, url: str):
    normalized_target = _normalize_host_path(url)
    for entry in registry.all_sources():
        if entry.url and _normalize_host_path(entry.url) == normalized_target:
            return entry
    for entry in registry.all_sources():
        if entry.url and (entry.url == url or entry.url.startswith(url) or url.startswith(entry.url)):
            return entry
    return None


def _normalize_host_path(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.netloc.lower().removeprefix('www.')}{parsed.path.rstrip('/') or '/'}"


def _hostname(url: str | None) -> str:
    if not url:
        return ""
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _entry_source_class(entry) -> str:
    source_class = getattr(entry, "source_class", "unknown") or "unknown"
    if source_class == "unknown" and getattr(entry, "url", None):
        source_class = classify_source(entry.url)
    return normalize_source_class(source_class)


def _numeric_claim_counts(body: str, citation_sources: dict[int, object]) -> tuple[int, int]:
    numeric_claim_count = 0
    weak_numeric_claim_count = 0
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if not _NUMERIC_CLAIM_RE.search(sentence):
            continue
        numeric_claim_count += 1
        cited_classes = []
        for match in _INLINE_CITATION_RE.finditer(sentence):
            entry = citation_sources.get(int(match.group(1)))
            if entry is not None:
                cited_classes.append(_entry_source_class(entry))
        if not cited_classes or not any(source_class in AUTHORITATIVE_CLASSES for source_class in cited_classes):
            weak_numeric_claim_count += 1
    return numeric_claim_count, weak_numeric_claim_count


def _strip_request_context(request_text: str) -> str:
    """Remove appended clarification or resume context from the original request."""
    clean = request_text
    for marker in _CONTEXT_MARKERS:
        if marker in clean:
            clean = clean.split(marker, 1)[0]
    return clean.strip() or request_text.strip()


def _scope_terms(text: str | None) -> set[str]:
    """Return meaningful scope terms from a request or report body."""
    if not text:
        return set()

    terms: set[str] = set()
    for token in _WORD_RE.findall(_strip_request_context(text).lower()):
        if len(token) >= 4 and token not in _STOPWORDS:
            terms.add(token)
    return terms


def report_matches_request_scope(
    report_text: str | None,
    request_text: str | None,
    *,
    candidate_name: str | None = None,
) -> tuple[bool, str]:
    """Return whether a recovered report still matches the submitted request scope."""
    if not report_text or not request_text:
        report_basis = report_text or ""
        if candidate_name:
            report_basis = f"{candidate_name}\n{report_basis}"
        lead_text = re.sub(r"[_-]+", " ", report_basis.lower())[:500]
        if any(marker in lead_text for marker in _DRIFT_MARKERS):
            return False, "recovered report shows a clear alternatives/career-paths drift"
        return True, "no_scope_check"

    request_terms = _scope_terms(request_text)
    if not request_terms:
        return True, "no_scope_terms"

    report_basis = report_text
    if candidate_name:
        report_basis = f"{candidate_name}\n{report_basis}"
    report_basis_lower = report_basis.lower()
    normalized_basis = re.sub(r"[_-]+", " ", report_basis_lower)
    lead_text = normalized_basis[:500]

    comparison_markers = ("alternative", "compare", "comparison", "vocational", "apprenticeship", "trade")
    if not any(marker in _strip_request_context(request_text).lower() for marker in comparison_markers):
        if any(marker in lead_text for marker in _DRIFT_MARKERS):
            return False, "recovered report shows a clear alternatives/career-paths drift"

    matched_terms = sorted(term for term in request_terms if term in report_basis_lower)
    if not matched_terms:
        return False, "no request-scope terms found in recovered report"

    required_matches = 1 if len(request_terms) == 1 else min(2, len(request_terms))
    if len(matched_terms) < required_matches:
        return (
            False,
            f"recovered report only matched {len(matched_terms)} of {len(request_terms)} scope term(s)",
        )

    return True, f"matched scope terms: {', '.join(matched_terms[:5])}"

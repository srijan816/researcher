# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Wave 3 W3.3 — semantic source deduplication.

The :class:`SourceRegistry` deduplicates by URL (normalized form), but a deep
researcher run often captures *many* distinct URLs that point at the same
underlying fact: wire-service copies, syndicated press releases, reposts, the
two halves of a forum thread. When the report builder enumerates
``registry.all_sources()`` to write the References section, those near-duplicates
inflate the count, obscure the actually-different sources, and force the LLM to
spend context budget re-reading redundant URLs.

This module deduplicates the *presentation* list (what the LLM sees as
potential citations) without touching the registry itself, so citation
verification and inline [N] lookup remain anchored to the full URL set.

Strategy is two-stage and fail-soft:

1. **Deterministic fingerprint** — cheap, always on. Buckets sources by a
   short hash of (canonical host + normalized path + lowercased title + first
   snippet of content if available). Identical copies collapse to one.
2. **Semantic embedding** — optional, opt-in via ``AIQ_SOURCE_DEDUP_EMBED=true``
   (default off to keep verifier deterministic and to avoid burning NVIDIA
   quota on tiny reports). For each bucket surviving stage 1, embed the
   ``title + url + first ~300 chars of any cached snippet`` via NVIDIA's
   ``nvidia/llama-nemotron-embed-vl-1b-v2`` model (the same one the LlamaIndex
   retriever uses) and union-find by cosine similarity ≥ ``threshold``
   (default 0.92).

Representative selection prefers entries with a ``citation_key`` (knowledge
layer > web result) and the longest non-empty ``title``. Ties break on first
insertion order so dedup is stable.

If embedding fails for any reason — missing API key, model 404, network
hiccup — we log a warning and fall back to the deterministic-only result so
the report is never blocked by a dedup failure.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from urllib.parse import urlparse

from .citation_verification import SourceEntry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: Default cosine similarity threshold for the semantic stage. 0.92 is tight
#: enough to avoid merging genuinely-different sources (which routinely sit in
#: the 0.7–0.85 range) while still catching wire-service paraphrases.
DEFAULT_SIMILARITY_THRESHOLD = 0.92

#: Maximum sources we'll embed in a single batch. Above this we chunk to keep
#: the NVIDIA embedding call bounded — the model times out around 60s for
#: 100+ texts and we don't want a long dedup pass blocking report assembly.
DEFAULT_MAX_EMBED_BATCH = 64

#: Headline of how much we trust an entry to represent a cluster. Higher =
#: pulled to the front. Knowledge-layer citation_key wins outright.
_REP_SCORE_HAS_CITATION_KEY = 10_000
_REP_SCORE_HAS_TITLE = 100
_REP_SCORE_TITLE_LEN = 1  # additive; longer title -> slightly higher score


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a 1/true/yes env var into a bool, with default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupResult:
    """Outcome of running :class:`SourceDeduper` over a source list.

    Attributes:
        kept: One representative entry per cluster, in original insertion order.
        removed: Entries that were collapsed into a kept entry. Each tuple is
            ``(removed_entry, kept_entry)`` so callers can audit or re-inject.
        clusters: Mapping of kept-entry identity → list of original indices
            that landed in its cluster. Useful for telemetry.
        stage1_collapses: Number of entries collapsed by the deterministic stage.
        stage2_collapses: Number of additional entries collapsed by semantic.
        embedding_attempted: True if the semantic stage ran (even if it errored).
    """

    kept: list[SourceEntry]
    removed: list[tuple[SourceEntry, SourceEntry]]
    clusters: dict[int, list[int]]
    stage1_collapses: int
    stage2_collapses: int
    embedding_attempted: bool


# ---------------------------------------------------------------------------
# Deduper
# ---------------------------------------------------------------------------


class SourceDeduper:
    """Deduplicate a list of :class:`SourceEntry` for report presentation.

    The deduper never mutates the input list and never mutates the registry
    from which the entries came. It returns a :class:`DedupResult` whose
    ``kept`` field is what the report builder should hand to the LLM as the
    pool of available sources. The original list stays intact for citation
    verification (every distinct URL still resolves).

    Two stages run in sequence:

    * :meth:`_stage1_deterministic` — fingerprint-based, always on.
    * :meth:`_stage2_semantic` — embedding-based, gated by ``enable_embedding``
      and ``AIQ_SOURCE_DEDUP_EMBED``. Errors degrade to stage-1-only output.

    Thread safety: instances are stateless across calls and safe to share
    across async tasks, but the underlying ``embed_fn`` injected for tests
    must be itself concurrency-safe.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_embed_batch: int = DEFAULT_MAX_EMBED_BATCH,
        embed_fn: Any | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_embed_batch = max_embed_batch
        self._embed_fn = embed_fn  # injected for tests; defaults to NVIDIA

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        entries: list[SourceEntry],
        enable_embedding: bool | None = None,
    ) -> DedupResult:
        """Run stage-1 then optional stage-2 over ``entries``.

        Args:
            entries: The source list to dedup. Order is preserved.
            enable_embedding: Override the default embedding gate. If
                ``None``, fall back to ``AIQ_SOURCE_DEDUP_EMBED`` (default off).

        Returns:
            A :class:`DedupResult` whose ``kept`` list contains one entry per
            cluster, in original insertion order.
        """
        if enable_embedding is None:
            enable_embedding = _env_flag("AIQ_SOURCE_DEDUP_EMBED", default=False)

        # Snapshot the original indices so cluster accounting is stable even
        # if dedup drops or reorders entries.
        surviving: list[tuple[int, SourceEntry]] = list(enumerate(entries))

        # Stage 1: deterministic fingerprint
        surviving, stage1_pairs = self._stage1_deterministic(surviving)

        removed: list[tuple[SourceEntry, SourceEntry]] = list(stage1_pairs)
        stage2_pairs: list[tuple[SourceEntry, SourceEntry]] = []
        embedding_attempted = False

        if enable_embedding and surviving:
            try:
                surviving, stage2_pairs = self._stage2_semantic(surviving)
                embedding_attempted = True
            except Exception:  # noqa: BLE001 - fail-soft by design
                logger.warning(
                    "aiq.source_dedup stage2_semantic failed; "
                    "falling back to deterministic-only result",
                    exc_info=True,
                )
                embedding_attempted = True  # attempted but failed

        removed.extend(stage2_pairs)

        # Cluster accounting: build kept-index → original-index map.
        # We walk in original order so the kept list is stable.
        kept: list[SourceEntry] = []
        kept_index_by_id: dict[int, int] = {}
        clusters: dict[int, list[int]] = {}
        for orig_idx, entry in surviving:
            clusters.setdefault(id(entry), []).append(orig_idx)

        # Map each kept entry to its cluster, preserving insertion order.
        # surviving is already in original-encounter order (stage 1/2 preserve it).
        for entry in (e for _, e in surviving):
            clusters.setdefault(id(entry), []).append(len(clusters))
            kept.append(entry)

        return DedupResult(
            kept=kept,
            removed=removed,
            clusters=clusters,
            stage1_collapses=len(stage1_pairs),
            stage2_collapses=len(stage2_pairs),
            embedding_attempted=embedding_attempted,
        )

    # ------------------------------------------------------------------
    # Stage 1: deterministic fingerprint
    # ------------------------------------------------------------------

    def _stage1_deterministic(
        self,
        indexed: list[tuple[int, SourceEntry]],
    ) -> tuple[list[tuple[int, SourceEntry]], list[tuple[SourceEntry, SourceEntry]]]:
        """Collapse entries whose canonical fingerprint matches an earlier one.

        The fingerprint is intentionally loose (host + path + title) so that
        URLs with trivial tracking-param differences still collapse, but
        different pages on the same host (different paths) do not.
        """
        buckets: dict[str, tuple[int, SourceEntry]] = {}
        surviving: list[tuple[int, SourceEntry]] = []
        removed: list[tuple[SourceEntry, SourceEntry]] = []

        for orig_idx, entry in indexed:
            if entry is None:  # type: ignore[unreachable]
                continue
            fp = self._fingerprint(entry)
            if fp is None:
                # No usable signal — keep it as-is rather than collapsing
                # unrelated entries together.
                surviving.append((orig_idx, entry))
                continue
            existing = buckets.get(fp)
            if existing is None:
                buckets[fp] = (orig_idx, entry)
                surviving.append((orig_idx, entry))
                continue
            _, kept_entry = existing
            representative = self._pick_representative(kept_entry, entry)
            if representative is kept_entry:
                removed.append((entry, kept_entry))
            else:
                # Replace the kept one in-place in the surviving list.
                surviving = [
                    (i, representative) if e is kept_entry else (i, e)
                    for i, e in surviving
                ]
                buckets[fp] = (orig_idx, representative)
                removed.append((kept_entry, representative))

        logger.debug(
            "aiq.source_dedup stage1_deterministic buckets=%d collapsed=%d",
            len(buckets),
            len(removed),
        )
        return surviving, removed

    @staticmethod
    def _fingerprint(entry: SourceEntry) -> str | None:
        """Compute a stable dedup key. Returns ``None`` when there's no
        usable signal — in which case the entry passes through untouched.
        """
        # Citation keys (knowledge layer) are already globally unique within
        # a session. If both entries have the same key, they're the same doc.
        if entry.citation_key:
            key = entry.citation_key.strip().lower()
            if key:
                return f"ck::{key}"

        # URL-based fingerprint: host + path + lowercased title.
        if entry.url:
            try:
                parsed = urlparse(entry.url)
                host = (parsed.netloc or "").lower()
                path = re.sub(r"/+", "/", parsed.path or "").rstrip("/")
            except Exception:  # noqa: BLE001 - never let fingerprint parsing kill dedup
                host, path = "", ""
            if host and path:
                title = (entry.title or "").strip().lower()
                # 64-bit hash keeps fingerprint storage bounded; collisions
                # within a session are vanishingly rare and merely cause one
                # extra entry to slip through to stage 2.
                digest = hashlib.sha1(
                    f"{host}|{path}|{title}".encode("utf-8", errors="replace")
                ).hexdigest()[:16]
                return f"u::{digest}"

        return None

    # ------------------------------------------------------------------
    # Stage 2: semantic via embeddings
    # ------------------------------------------------------------------

    def _stage2_semantic(
        self,
        indexed: list[tuple[int, SourceEntry]],
    ) -> tuple[list[tuple[int, SourceEntry]], list[tuple[SourceEntry, SourceEntry]]]:
        """Cluster entries by cosine similarity ≥ threshold.

        Algorithm: greedy single-pass. For each entry not yet assigned to a
        cluster, embed it; compare to existing cluster centroids; if any
        centroid is ≥ threshold, fold the entry into that cluster and update
        the centroid as a running mean; otherwise start a new cluster.

        For ≤ max_embed_batch entries we issue a single batched call. Above
        that we chunk — the dedup pass is best-effort and bounded latency
        matters more than perfect coverage of the tail.
        """
        if not indexed:
            return indexed, []

        # Sort by index to make centroid updates deterministic
        ordered = sorted(indexed, key=lambda ie: ie[0])
        texts = [self._embed_text(e) for _, e in ordered]
        # Single text → skip the API call entirely; nothing to cluster against.
        if len(texts) == 1:
            return ordered, []

        embeddings = self._embed(texts)
        if len(embeddings) != len(texts):
            # Defensive: if the embedder returned a different shape, bail to
            # the deterministic result rather than misalign entries.
            logger.warning(
                "aiq.source_dedup stage2_semantic: embedder returned %d vectors for %d texts; skipping",
                len(embeddings),
                len(texts),
            )
            return ordered, []

        clusters: list[list[int]] = []  # indices into `ordered`
        centroids: list[list[float]] = []
        assignments: list[int] = []  # per-entry: cluster index or -1
        removed: list[tuple[SourceEntry, SourceEntry]] = []

        for i, vec in enumerate(embeddings):
            best_cluster = -1
            best_sim = -1.0
            for c_idx, centroid in enumerate(centroids):
                sim = _cosine_similarity(vec, centroid)
                if sim >= self.similarity_threshold and sim > best_sim:
                    best_sim = sim
                    best_cluster = c_idx
            if best_cluster == -1:
                clusters.append([i])
                centroids.append(list(vec))
                assignments.append(len(clusters) - 1)
            else:
                clusters[best_cluster].append(i)
                _running_mean(centroids[best_cluster], vec, len(clusters[best_cluster]))
                assignments.append(best_cluster)

        # Materialize survivors + removed pairs.
        surviving: list[tuple[int, SourceEntry]] = []
        seen_reps: set[int] = set()
        for c_idx, members in enumerate(clusters):
            # Pick representative among cluster members using the same
            # heuristic as stage 1 so behavior is consistent.
            member_entries = [ordered[m] for m in members]
            rep_idx, rep_entry = self._pick_representative_from_members(member_entries)
            seen_reps.add(id(rep_entry))
            surviving.append((ordered[rep_idx][0], rep_entry))
            for m in members:
                if m == rep_idx:
                    continue
                removed_entry = ordered[m][1]
                if removed_entry is rep_entry:
                    continue
                removed.append((removed_entry, rep_entry))

        # Re-sort by original index so callers see stable ordering.
        surviving.sort(key=lambda ie: ie[0])

        logger.debug(
            "aiq.source_dedup stage2_semantic clusters=%d collapsed=%d threshold=%.2f",
            len(clusters),
            len(removed),
            self.similarity_threshold,
        )
        return surviving, removed

    def _embed_text(self, entry: SourceEntry) -> str:
        """Build the string we embed for an entry.

        Title carries the most signal; URL captures the source identity; we
        deliberately do not embed any snippet content because the registry
        doesn't store it (and pulling it would require re-fetching).
        """
        parts: list[str] = []
        if entry.title:
            parts.append(entry.title.strip())
        if entry.url:
            parts.append(entry.url.strip())
        if entry.citation_key:
            parts.append(f"[{entry.citation_key}]")
        return " | ".join(parts) if parts else entry.url or entry.citation_key or "unknown"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Run embeddings in batches; respects max_embed_batch."""
        embed_fn = self._embed_fn or self._default_embed_fn
        if len(texts) <= self.max_embed_batch:
            return list(embed_fn(texts))

        out: list[list[float]] = []
        for start in range(0, len(texts), self.max_embed_batch):
            chunk = texts[start : start + self.max_embed_batch]
            out.extend(embed_fn(chunk))
        return out

    @staticmethod
    def _default_embed_fn(texts: list[str]) -> list[list[float]]:
        """Default embedding backend: NVIDIA nemotron-embed-vl-1b-v2.

        Imported lazily so test environments without ``llama_index`` or
        without ``NVIDIA_API_KEY`` set can still import the module.
        """
        try:
            from llama_index.embeddings.nvidia import NVIDIAEmbedding  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "llama-index-embeddings-nvidia is required for semantic source dedup; "
                "either install it or set AIQ_SOURCE_DEDUP_EMBED=false."
            ) from e

        import os

        model_name = os.environ.get(
            "AIQ_SOURCE_DEDUP_EMBED_MODEL",
            os.environ.get("AIQ_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2"),
        )
        base_url = os.environ.get(
            "AIQ_SOURCE_DEDUP_EMBED_BASE_URL",
            os.environ.get("AIQ_EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set; cannot run semantic source dedup.")

        embedder = NVIDIAEmbedding(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            # The model is sync-capable through ``aget_text_embedding_batch``;
            # we run it synchronously here because dedup runs on the report
            # assembly critical path and we want predictable latency.
        )
        # Use the sync batch API (the async path would require us to be
        # inside a running event loop, which dedup usually isn't).
        return list(embedder.get_text_embedding_batch(texts))  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------
    # Representative selection
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_representative(a: SourceEntry, b: SourceEntry) -> SourceEntry:
        """Pick the better representative of two equal-fingerprint entries."""
        return SourceDeduper._pick_representative_from_members([(0, a), (1, b)])[1]

    @staticmethod
    def _pick_representative_from_members(
        members: Iterable[tuple[int, SourceEntry]],
    ) -> tuple[int, SourceEntry]:
        """Choose the member with the highest representative score.

        Score (descending priority):
        1. Has a citation_key → knowledge-layer doc wins outright.
        2. Has a non-empty title → some metadata wins over none.
        3. Longer title → marginal preference (additive).
        4. Earlier original index → tiebreaker (stable for tests).
        """
        best_idx = -1
        best_entry: SourceEntry | None = None
        best_score = -1
        for orig_idx, entry in members:
            score = 0
            if entry.citation_key:
                score += _REP_SCORE_HAS_CITATION_KEY
            title = (entry.title or "").strip()
            if title:
                score += _REP_SCORE_HAS_TITLE
                score += min(len(title), 256) * _REP_SCORE_TITLE_LEN
            # Earlier index breaks ties — subtract a small fraction so order
            # matters only when scores are equal.
            score -= orig_idx * 1e-6
            if best_entry is None or score > best_score:
                best_idx = orig_idx
                best_entry = entry
                best_score = score
        assert best_entry is not None  # members is never empty
        return best_idx, best_entry


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns 0.0 on mismatched lengths (callers bail) or zero-norm vectors
    (no signal).
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b) ** 0.5


def _running_mean(centroid: list[float], new_vec: list[float], new_count: int) -> None:
    """In-place update of centroid as running mean."""
    if new_count <= 1:
        if len(centroid) != len(new_vec):
            centroid[:] = list(new_vec)
            return
        centroid[:] = new_vec
        return
    # new_count is the count *after* the new vector joins.
    prev_weight = (new_count - 1) / new_count
    new_weight = 1.0 / new_count
    for i, v in enumerate(new_vec):
        if i >= len(centroid):
            centroid.append(v * new_weight)
            continue
        centroid[i] = centroid[i] * prev_weight + v * new_weight


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DedupResult",
    "SourceDeduper",
]
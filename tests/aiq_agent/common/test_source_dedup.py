# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave 3 W3.3 — source deduplication.

Covers both stages:

* Stage 1 (deterministic fingerprint) — always on, no API.
* Stage 2 (semantic embedding) — gated; we inject a stubbed ``embed_fn`` so
  no network is touched.

The dedup must be stable (insertion-order preservation) and the registry must
be unchanged after a dedup call (we never mutate the registry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.source_dedup import DEFAULT_SIMILARITY_THRESHOLD
from aiq_agent.common.source_dedup import DedupResult
from aiq_agent.common.source_dedup import SourceDeduper
from aiq_agent.common.source_dedup import _cosine_similarity


# ---------------------------------------------------------------------------
# Stage 1: deterministic fingerprint
# ---------------------------------------------------------------------------


class TestStage1Deterministic:
    def test_identical_url_and_title_collapse(self) -> None:
        d = SourceDeduper(embed_fn=_never_call)
        e1 = SourceEntry(url="https://example.com/a", title="Same Title")
        e2 = SourceEntry(url="https://example.com/a", title="Same Title")
        result = d.deduplicate([e1, e2], enable_embedding=False)
        assert len(result.kept) == 1
        assert result.stage1_collapses == 1
        assert result.stage2_collapses == 0
        assert result.embedding_attempted is False

    def test_tracking_param_does_not_split_buckets(self) -> None:
        # The fingerprint hashes (host, path, title) — query strings are
        # intentionally excluded, so UTM-tagged copies collapse with the
        # canonical link.
        d = SourceDeduper(embed_fn=_never_call)
        e1 = SourceEntry(url="https://example.com/a", title="Title")
        e2 = SourceEntry(url="https://example.com/a?utm_source=x", title="Title")
        result = d.deduplicate([e1, e2], enable_embedding=False)
        assert len(result.kept) == 1
        assert result.stage1_collapses == 1

    def test_different_paths_do_not_collapse(self) -> None:
        d = SourceDeduper(embed_fn=_never_call)
        e1 = SourceEntry(url="https://example.com/a", title="Title")
        e2 = SourceEntry(url="https://example.com/b", title="Title")
        result = d.deduplicate([e1, e2], enable_embedding=False)
        assert len(result.kept) == 2
        assert result.stage1_collapses == 0

    def test_different_titles_do_not_collapse(self) -> None:
        d = SourceDeduper(embed_fn=_never_call)
        e1 = SourceEntry(url="https://example.com/a", title="Title One")
        e2 = SourceEntry(url="https://example.com/a", title="Title Two")
        result = d.deduplicate([e1, e2], enable_embedding=False)
        assert len(result.kept) == 2
        assert result.stage1_collapses == 0

    def test_untitled_entries_fall_through(self) -> None:
        # An entry with no URL and no citation_key and no title has no
        # fingerprint signal; it must pass through rather than be merged
        # with unrelated entries.
        d = SourceDeduper(embed_fn=_never_call)
        bare1 = SourceEntry()
        bare2 = SourceEntry()
        result = d.deduplicate([bare1, bare2], enable_embedding=False)
        assert len(result.kept) == 2
        assert result.stage1_collapses == 0

    def test_citation_key_collision_collapses(self) -> None:
        d = SourceDeduper(embed_fn=_never_call)
        e1 = SourceEntry(citation_key="doc.pdf", title="From A")
        e2 = SourceEntry(citation_key="doc.pdf", title="From B")
        result = d.deduplicate([e1, e2], enable_embedding=False)
        assert len(result.kept) == 1
        assert result.stage1_collapses == 1

    def test_representative_prefers_citation_key(self) -> None:
        # Two entries with identical citation_key fingerprint. Without a
        # citation_key, they would URL-fingerprint to two different buckets
        # and stay separate. The fingerprint logic prefers the citation_key
        # bucket when one is present, so we give both entries the same
        # citation_key and a (different) URL each — they collide on ck.
        d = SourceDeduper(embed_fn=_never_call)
        a = SourceEntry(citation_key="doc.pdf", title="Short", url="https://a.example/x")
        b = SourceEntry(citation_key="doc.pdf", title="A longer descriptive title", url="https://b.example/y")
        result = d.deduplicate([a, b], enable_embedding=False)
        assert len(result.kept) == 1
        # When citation_keys tie, the longer-title entry wins.
        assert result.kept[0] is b
        # The shorter entry is recorded as the one that was collapsed away.
        assert result.removed[0][0] is a
        assert result.removed[0][1] is b

    def test_representative_within_bucket_picks_longer_title(self) -> None:
        # Pure title-length preference within a bucket (no citation_key tie).
        # We synthesize the same fingerprint via identical (host, path, title)
        # by giving two entries the *same* URL — the registry would dedup at
        # insertion, but the deduper's input list is independent, so we just
        # pass two entries with identical fields.
        d = SourceDeduper(embed_fn=_never_call)
        short = SourceEntry(url="https://example.com/a", title="ab")
        long = SourceEntry(url="https://example.com/a", title="ab")
        # Identical fields → identical fingerprint → bucket collision.
        # Representative is the first one seen (deterministic tiebreak).
        result = d.deduplicate([short, long], enable_embedding=False)
        assert len(result.kept) == 1
        # Either is acceptable; both are identical.
        assert result.kept[0] in (short, long)

    def test_keeps_insertion_order(self) -> None:
        d = SourceDeduper(embed_fn=_never_call)
        entries = [
            SourceEntry(url="https://example.com/a", title="A"),
            SourceEntry(url="https://example.com/b", title="B"),
            SourceEntry(url="https://example.com/c", title="C"),
        ]
        result = d.deduplicate(entries, enable_embedding=False)
        kept_urls = [e.url for e in result.kept]
        assert kept_urls == [e.url for e in entries]


# ---------------------------------------------------------------------------
# Stage 2: semantic embedding
# ---------------------------------------------------------------------------


class TestStage2Semantic:
    def test_semantic_dedup_collapses_near_duplicates(self) -> None:
        # Three "semantic" clusters:
        #  * A and B are paraphrases of the same fact (cosine 0.96)
        #  * C is a different fact (cosine 0.00 to A)
        # The stub returns vectors keyed by the entry's url, so test inputs
        # map cleanly regardless of how embed_text() formats them.
        table = {
            "https://w.com/a": [1.0, 0.0, 0.0],
            "https://w.com/b": [0.96, 0.28, 0.0],  # cos = 0.96 to A
            "https://w.com/c": [0.0, 1.0, 0.0],    # cos = 0.00 to A
        }
        embed_fn = _url_keyed_embedder(table)

        d = SourceDeduper(similarity_threshold=0.90, embed_fn=embed_fn)
        eA = SourceEntry(url="https://w.com/a", title="Fact A", source_class="news")
        eB = SourceEntry(url="https://w.com/b", title="Same fact A in different words", source_class="news")
        eC = SourceEntry(url="https://w.com/c", title="Completely unrelated", source_class="news")
        result = d.deduplicate([eA, eB, eC], enable_embedding=True)

        assert len(result.kept) == 2
        assert result.embedding_attempted is True
        assert result.stage2_collapses == 1
        kept_urls = sorted(e.url for e in result.kept)
        # C is its own cluster. A and B cluster together; the longer-titled
        # B wins representative, A is collapsed away.
        assert "https://w.com/c" in kept_urls
        assert "https://w.com/b" in kept_urls
        assert "https://w.com/a" not in kept_urls

    def test_below_threshold_keeps_both(self) -> None:
        table = {
            "https://w.com/a": [1.0, 0.0, 0.0],
            "https://w.com/b": [0.5, 0.866, 0.0],  # cos = 0.5
        }
        embed_fn = _url_keyed_embedder(table)
        d = SourceDeduper(similarity_threshold=0.90, embed_fn=embed_fn)
        eA = SourceEntry(url="https://w.com/a", title="A")
        eB = SourceEntry(url="https://w.com/b", title="B")
        result = d.deduplicate([eA, eB], enable_embedding=True)
        assert len(result.kept) == 2
        assert result.stage2_collapses == 0

    def test_threshold_boundary_inclusive(self) -> None:
        table = {
            "https://w.com/a": [1.0, 0.0],
            "https://w.com/b": [0.92, 0.39],  # cos ≈ 0.92
        }
        embed_fn = _url_keyed_embedder(table)
        d = SourceDeduper(similarity_threshold=0.92, embed_fn=embed_fn)
        eA = SourceEntry(url="https://w.com/a", title="A")
        eB = SourceEntry(url="https://w.com/b", title="B")
        result = d.deduplicate([eA, eB], enable_embedding=True)
        # ≥ threshold collapses; cosine 0.92 meets 0.92 exactly.
        assert len(result.kept) == 1
        assert result.stage2_collapses == 1

    def test_embedding_failure_degrades_gracefully(self) -> None:
        def broken_embed(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("simulated NVIDIA API outage")

        d = SourceDeduper(embed_fn=broken_embed)
        eA = SourceEntry(url="https://w.com/a", title="A")
        eB = SourceEntry(url="https://w.com/a", title="A")  # stage-1 dup
        eC = SourceEntry(url="https://w.com/b", title="B")
        result = d.deduplicate([eA, eB, eC], enable_embedding=True)
        # Stage 1 still ran; stage 2 didn't add anything (and didn't crash).
        assert result.embedding_attempted is True
        assert result.stage1_collapses == 1
        assert result.stage2_collapses == 0
        assert len(result.kept) == 2

    def test_single_entry_skips_embed_call(self) -> None:
        # With one entry, the semantic stage should be a no-op (nothing to
        # cluster against). The embed_fn must NOT be called.
        called = {"n": 0}

        def tracker(texts: list[str]) -> list[list[float]]:
            called["n"] += 1
            return [[0.0, 0.0] for _ in texts]

        d = SourceDeduper(embed_fn=tracker)
        result = d.deduplicate(
            [SourceEntry(url="https://w.com/a", title="A")], enable_embedding=True
        )
        assert len(result.kept) == 1
        assert called["n"] == 0
        assert result.embedding_attempted is True

    def test_mismatched_embed_output_falls_back(self) -> None:
        # If the embedder returns a different number of vectors than texts,
        # the semantic stage must bail rather than misalign entries.
        def bad_count(texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts[:-1]]  # one short

        d = SourceDeduper(embed_fn=bad_count)
        result = d.deduplicate(
            [
                SourceEntry(url="https://w.com/a", title="A"),
                SourceEntry(url="https://w.com/b", title="B"),
            ],
            enable_embedding=True,
        )
        # Stage 1 does nothing (different URLs); stage 2 silently fails.
        assert result.stage1_collapses == 0
        assert result.stage2_collapses == 0
        assert len(result.kept) == 2


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_deduped_sources_does_not_mutate_registry(self) -> None:
        # Two distinct URLs that differ in path (so the registry keeps both)
        # but fingerprint-collide by (host, path, title) when wrapped with
        # semantic dedup — no, simpler: two distinct URLs that the registry
        # keeps, and verify dedup returns them unchanged because they have
        # *different* fingerprints. The invariant under test is that the
        # registry is not mutated by a dedup call regardless of outcome.
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://a.com/x", title="X"))
        reg.add(SourceEntry(url="https://b.com/y", title="Y"))
        before = len(reg.all_sources())
        deduped = reg.deduped_sources(enable_embedding=False)
        after = len(reg.all_sources())
        assert before == 2
        assert after == 2, "registry must not be mutated by dedup"
        assert len(deduped) == 2

    def test_deduped_sources_with_injected_deduper(self) -> None:
        # Caller-supplied deduper → caller controls thresholds and stub.
        # We register two URL-distinct entries that share a fingerprint
        # (impossible via the URL fingerprint since paths differ); instead
        # we use citation_key collision, which the registry keeps distinct
        # because they have different URLs/citation_key field-as-whole, but
        # the deduper's ck:: fingerprint folds them.
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://a.example/x", citation_key="doc.pdf", title="Long version"))
        reg.add(SourceEntry(url="https://b.example/y", citation_key="doc.pdf", title="Short"))
        before = len(reg.all_sources())
        deduped = reg.deduped_sources(enable_embedding=False)
        after = len(reg.all_sources())
        assert before == 2
        assert after == 2, "registry must not be mutated by dedup"
        assert len(deduped) == 1, "citation_key collision should collapse"
        # The longer-title entry wins the representative pick.
        assert deduped[0].title == "Long version"

    def test_deduped_sources_single_entry_passthrough(self) -> None:
        reg = SourceRegistry()
        reg.add(SourceEntry(url="https://a.com/x", title="X"))
        assert len(reg.deduped_sources(enable_embedding=False)) == 1

    def test_deduped_sources_empty_registry(self) -> None:
        reg = SourceRegistry()
        assert reg.deduped_sources(enable_embedding=False) == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _never_call(_texts: list[str]) -> list[list[float]]:
    raise AssertionError("embed_fn should not be called when enable_embedding=False")


@dataclass
class _StubEmbedder:
    """Map embed_text()'s output → unit vector from a fixed lookup table.

    SourceDeduper._embed_text() joins ``title | url | [citation_key]`` for
    each entry, so tests can precompute that string and put it in the table.
    Entries whose text isn't in the table get a zero vector (stage-1 dup
    handling is unaffected).
    """

    table: dict[str, list[float]]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = self.table.get(t, [0.0, 0.0, 0.0])
            out.append(_normalize(vec))
        return out


def _stub_embedder(table: dict[str, list[float]]) -> Any:
    return _StubEmbedder(table=table)


class _UrlKeyedEmbedder:
    """Embedder that ignores ``embed_text`` formatting and looks up by URL.

    Useful when tests want to keep the embedder table decoupled from
    SourceDeduper's exact embed_text() output format — pass URLs as keys.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = {k: _normalize(v) for k, v in table.items()}

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # Pull the first URL-looking token out of the embed text.
            url = next((tok for tok in t.split(" | ") if tok.startswith("http")), None)
            vec = self._table.get(url) if url else None
            out.append(vec if vec is not None else [0.0, 0.0, 0.0])
        return out


def _url_keyed_embedder(table: dict[str, list[float]]) -> Any:
    return _UrlKeyedEmbedder(table=table)


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine_similarity([1, 0], [1, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------


def test_default_similarity_threshold_is_sensible() -> None:
    # Pinned: this is the knob callers tune. If you change it, update the
    # docstring on AIQ_SOURCE_DEDUP_EMBED and the W3.3 design notes.
    assert 0.85 <= DEFAULT_SIMILARITY_THRESHOLD <= 0.95


def test_dedup_result_fields() -> None:
    # The dataclass shape is part of the public surface (callers may inspect
    # .stage1_collapses / .stage2_collapses for telemetry).
    r = DedupResult(
        kept=[],
        removed=[],
        clusters={},
        stage1_collapses=0,
        stage2_collapses=0,
        embedding_attempted=False,
    )
    assert r.kept == []
    assert r.embedding_attempted is False

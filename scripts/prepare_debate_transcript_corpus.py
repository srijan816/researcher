#!/usr/bin/env python3
"""Prepare debate transcripts for local research retrieval.

The NotebookLM export uses compact/unrecognizable filenames, but its manifest
contains useful motion metadata. This script creates a clean corpus folder with
canonical documents, retrieval chunks, and lightweight theme indexes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import textwrap
from collections import Counter
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from pathlib import Path

DEFAULT_SOURCE_DIR = Path(
    "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/notebooklm_exports/debate_transcripts"
)
DEFAULT_RECENT_SOURCE_DIR = (
    Path("/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data")
    / "youtube_recent_transcripts_2022_2026"
)  # pragma: allowlist secret
DEFAULT_OUTPUT_DIR = Path(
    "/Volumes/SrijanExt/Users/Srijan/Downloads/code/minimax/debate-yt/data/prepared_debate_corpus"
)

TRANSCRIPT_MARKERS = (
    "This is a transcript from a YouTube video about a debate.",
    "\ufeffThis is a transcript from a YouTube video about a debate.",
    "ï»¿This is a transcript from a YouTube video about a debate.",
    "■This is a transcript from a YouTube video about a debate.",
)

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "education": (
        "university",
        "school",
        "student",
        "teaching",
        "language",
        "admission",
        "standardized",
        "curriculum",
    ),
    "law_criminal_justice": (
        "criminal",
        "prison",
        "convict",
        "court",
        "law",
        "legal",
        "trial",
        "justice",
        "sharia",
        "constitution",
        "police",
    ),
    "civil_liberties_rights": (
        "free speech",
        "freedom",
        "rights",
        "individual freedom",
        "protest",
        "surveillance",
        "privacy",
        "deplatforming",
    ),
    "international_relations_conflict": (
        "war",
        "occupied",
        "occupation",
        "regime",
        "foreign",
        "aid",
        "military",
        "sanction",
        "intervene",
        "china",
        "usa",
        "saudi",
        "tigray",
        "south africa",
        "anc",
        "democracy",
    ),
    "politics_governance": (
        "government",
        "state",
        "election",
        "democratic",
        "legislature",
        "authoritarian",
        "federal",
        "policy",
        "public",
        "leaders",
    ),
    "economics_business_labor": (
        "economy",
        "economic",
        "funding",
        "minimum wage",
        "jobs",
        "work",
        "private equity",
        "company",
        "market",
        "retail investing",
        "interest rates",
        "business",
    ),
    "environment_ecology": (
        "environment",
        "environmental",
        "climate",
        "zoos",
        "animals",
        "agricultural",
        "ecology",
    ),
    "technology_media_ai": (
        "technology",
        "ai",
        "algorithm",
        "big data",
        "newspapers",
        "media",
        "pegasus",
        "spyware",
        "neuralink",
        "transhumanism",
        "brain-computer",
        "space",
        "extraterrestrial",
    ),
    "health_bioethics_wellness": (
        "mindfulness",
        "meditation",
        "health",
        "healthcare",
        "bioethics",
        "reincarnation",
        "memories",
        "body",
    ),
    "culture_religion_identity": (
        "religion",
        "religious",
        "feminists",
        "lgbtq",
        "identity",
        "gender",
        "queer",
        "culture",
        "cultural",
        "minority",
    ),
    "arts_sports_pop_culture": (
        "art",
        "artists",
        "disney",
        "sports",
        "pornography",
        "shakespeare",
        "genius",
        "hobbit",
        "shire",
    ),
    "family_society": (
        "children",
        "child",
        "families",
        "family",
        "parental",
        "women",
        "social",
        "society",
        "humanity",
        "working class",
    ),
}


def normalize_text(value: str) -> str:
    replacements = {
        "ï»¿": "",
        "\ufeff": "",
        "■": " ",
        "\u00a0": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return re.sub(r"\s+", " ", value).strip()


def clean_motion(raw_motion: str) -> str:
    motion = normalize_text(raw_motion)
    motion = re.sub(r"^Motion:\s*", "", motion, flags=re.IGNORECASE).strip()
    for marker in TRANSCRIPT_MARKERS:
        marker = normalize_text(marker)
        if marker and marker in motion:
            motion = motion.split(marker, 1)[0].strip()
    return motion.strip(" -–—")


def clean_transcript(raw_text: str, motion: str) -> str:
    text = normalize_text(raw_text)
    for marker in TRANSCRIPT_MARKERS:
        marker = normalize_text(marker)
        if marker and marker in text:
            text = text.split(marker, 1)[1].strip()
            break
    if text.lower().startswith("motion:"):
        text = re.sub(r"^Motion:\s*", "", text, flags=re.IGNORECASE).strip()
        if motion and text.lower().startswith(motion[:80].lower()):
            text = text[len(motion) :].strip(" .:-–—")
    return normalize_text(text)


def slugify(value: str, *, max_len: int = 90) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_len].strip("-") or "untitled"


def transcript_id(row: dict[str, str]) -> str:
    raw_path = Path(row["raw_text_path"])
    return raw_path.stem


def extract_incoming_motion(raw_text: str, source_path: Path) -> tuple[str, str]:
    text = normalize_text(raw_text)
    motion_match = re.match(r"^Motion:\s*(.+?)(?:\n|$)", raw_text.strip(), flags=re.IGNORECASE)
    if motion_match:
        motion = clean_motion(motion_match.group(1))
        body = re.sub(r"^Motion:\s*.+?(?:\n|$)", "", raw_text.strip(), count=1, flags=re.IGNORECASE)
        return motion, clean_transcript(body, motion)

    heading_match = re.match(r"^#\s+(.+?)(?:\n|$)", raw_text.strip())
    if heading_match:
        motion = clean_motion(heading_match.group(1))
        body = re.sub(r"^#\s+.+?(?:\n|$)", "", raw_text.strip(), count=1)
        return motion, clean_transcript(body, motion)

    motion = clean_motion(source_path.stem.replace("_", " ").replace("-", " "))
    return motion, clean_transcript(text, motion)


def truncate_motion_candidate(candidate: str) -> str:
    candidate = normalize_text(candidate).strip('" .:-–—')
    stop_patterns = (
        r"\bOpening Government\b",
        r"\bOpening Opposition\b",
        r"\bClosing Government\b",
        r"\bClosing Opposition\b",
        r"\bOG\b",
        r"\bOO\b",
        r"\bCG\b",
        r"\bCO\b",
        r"\bAff(?:irmative)?:\b",
        r"\bNeg(?:ative)?:\b",
        r"\bTeams?:\b",
        r"\bAdjudicators?:\b",
        r"\bWinner:\b",
        r"\bhttps?://",
        r"\bWelcome\b",
        r"\bwithout further ado\b",
    )
    for pattern in stop_patterns:
        match = re.search(pattern, candidate, flags=re.IGNORECASE)
        if match and match.start() >= 20:
            candidate = candidate[: match.start()].strip(" .:-–—")

    sentence = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)[0].strip()
    if len(sentence) >= 20:
        candidate = sentence
    return candidate[:260].strip(" .:-–—")


def infer_motion_from_recent(row: dict, raw_text: str) -> tuple[str, bool]:
    raw_motion = clean_motion(str(row.get("motion") or ""))
    needs_review = not raw_motion or "motion needs review" in raw_motion.lower() or raw_motion.endswith("...")
    if raw_motion and not needs_review:
        return truncate_motion_candidate(raw_motion), False

    haystacks = [
        str(row.get("description") or ""),
        raw_text[:4000],
        str(row.get("title") or ""),
    ]
    patterns = (
        r"(?:motion(?:\s+for\s+[^:.]{0,80})?\s+(?:reads|is|was|follows)|motion\s*:)\s*[\"']?(.{20,320})",
        r"(?:the\s+topic\s+of\s+(?:this\s+)?debate\s+is|the\s+motion\s+of\s+which\s+is\s+follows)\s*[\"']?(.{20,320})",
        r"((?:This\s+House|This\s+house|TH[BRWSOP]?\b|That\s+we|Assuming\s+the|Under\s+the\s+veil|For\s+the\s+purposes).{20,320})",
    )
    for haystack in haystacks:
        normalized = normalize_text(haystack)
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                candidate = truncate_motion_candidate(match.group(1))
                if len(candidate.split()) >= 5:
                    return candidate, needs_review

    fallback = clean_motion(str(row.get("title") or row.get("id") or "Untitled debate"))
    return fallback, True


def classify_themes(motion: str) -> list[str]:
    haystack = motion.lower()
    scores: Counter[str] = Counter()
    for theme, keywords in THEME_KEYWORDS.items():
        for keyword in keywords:
            if keyword_matches(haystack, keyword):
                scores[theme] += 2 if " " in keyword else 1
    if not scores:
        return ["general_debate"]
    top_score = max(scores.values())
    themes = [theme for theme, score in scores.most_common() if score >= max(1, top_score - 1)]
    return themes[:3]


def keyword_matches(haystack: str, keyword: str) -> bool:
    keyword = keyword.lower()
    if " " in keyword or "-" in keyword:
        return keyword in haystack
    return re.search(rf"\b{re.escape(keyword)}\b", haystack) is not None


def iter_chunks(
    text: str,
    *,
    chunk_words: int,
    overlap_words: int,
) -> Iterable[tuple[int, int, str]]:
    words = text.split()
    if not words:
        return
    start = 0
    total = len(words)
    while start < total:
        end = min(total, start + chunk_words)
        yield start, end, " ".join(words[start:end])
        if end >= total:
            break
        start = max(end - overlap_words, start + 1)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reset_generated_output(output_dir: Path) -> None:
    """Clear generated corpus files while preserving user-managed incoming files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("documents", "indexes", "chunks"):
        path = output_dir / directory
        if path.exists():
            shutil.rmtree(path)
    for filename in ("README.md", "manifest.csv", "manifest.jsonl", "recommended_queries.md"):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    (output_dir / "documents").mkdir(parents=True)
    (output_dir / "indexes" / "by_theme").mkdir(parents=True)
    (output_dir / "chunks").mkdir(parents=True)
    (output_dir / "incoming").mkdir(parents=True, exist_ok=True)


def render_document(metadata: dict, body: str) -> str:
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
    wrapped_body = "\n\n".join(textwrap.wrap(body, width=100, break_long_words=False))
    return (
        "---\n"
        f"transcript_id: {metadata['transcript_id']}\n"
        f"title_code: {metadata['title_code']}\n"
        f"source_index: {metadata['source_index']}\n"
        f"themes: {', '.join(metadata['themes'])}\n"
        "---\n\n"
        f"# {metadata['motion']}\n\n"
        "## Metadata\n\n"
        "```json\n"
        f"{metadata_json}\n"
        "```\n\n"
        "## Transcript\n\n"
        f"{wrapped_body}\n"
    )


def add_document(
    *,
    output_dir: Path,
    document_rows: list[dict],
    chunk_rows: list[dict],
    theme_docs: dict[str, list[dict]],
    source_index: int,
    doc_id: str,
    title_code: str,
    source_title: str,
    motion: str,
    body: str,
    source_raw_text_path: Path,
    source_markdown_path: Path | None,
    corpus_source: str,
    chunk_words: int,
    overlap_words: int,
    views_from_title: int | None = None,
    extra_metadata: dict | None = None,
) -> None:
    extra_metadata = extra_metadata or {}
    motion = clean_motion(motion) or source_title
    body = clean_transcript(body, motion)
    themes = classify_themes(motion)
    canonical_name = f"{source_index:03d}_{title_code}_{slugify(motion)}.md"
    canonical_path = output_dir / "documents" / canonical_name
    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

    metadata = {
        "transcript_id": doc_id,
        "source_index": source_index,
        "source_title": source_title,
        "title_code": title_code,
        "views_from_title": views_from_title,
        "motion": motion,
        "themes": themes,
        "source_raw_text_path": str(source_raw_text_path),
        "source_markdown_path": str(source_markdown_path or source_raw_text_path),
        "canonical_path": str(canonical_path),
        "word_count": len(body.split()),
        "character_count": len(body),
        "content_hash": content_hash,
        "corpus_source": corpus_source,
        **extra_metadata,
    }
    canonical_path.write_text(render_document(metadata, body), encoding="utf-8")

    document_row = {
        "transcript_id": doc_id,
        "source_index": source_index,
        "title_code": title_code,
        "views_from_title": views_from_title,
        "themes": ";".join(themes),
        "motion": motion,
        "word_count": metadata["word_count"],
        "character_count": metadata["character_count"],
        "canonical_path": str(canonical_path),
        "source_raw_text_path": str(source_raw_text_path),
        "content_hash": content_hash,
        "corpus_source": corpus_source,
        "youtube_url": extra_metadata.get("youtube_url", ""),
        "competition": extra_metadata.get("competition", ""),
        "year": extra_metadata.get("year", ""),
        "round": extra_metadata.get("round", ""),
        "motion_needs_review": extra_metadata.get("motion_needs_review", False),
    }
    document_rows.append(document_row)
    for theme in themes:
        theme_docs.setdefault(theme, []).append(document_row)

    for chunk_index, (word_start, word_end, chunk_text) in enumerate(
        iter_chunks(body, chunk_words=chunk_words, overlap_words=overlap_words)
    ):
        chunk_id = f"{doc_id}::chunk-{chunk_index:03d}"
        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "transcript_id": doc_id,
                "chunk_index": chunk_index,
                "word_start": word_start,
                "word_end": word_end,
                "source_index": source_index,
                "title_code": title_code,
                "themes": themes,
                "motion": motion,
                "canonical_path": str(canonical_path),
                "source_raw_text_path": str(source_raw_text_path),
                "corpus_source": corpus_source,
                "source_title": source_title,
                "youtube_url": extra_metadata.get("youtube_url", ""),
                "competition": extra_metadata.get("competition", ""),
                "year": extra_metadata.get("year", ""),
                "round": extra_metadata.get("round", ""),
                "motion_needs_review": extra_metadata.get("motion_needs_review", False),
                "text": chunk_text,
            }
        )


def prepare_corpus(
    source_dir: Path,
    output_dir: Path,
    chunk_words: int,
    overlap_words: int,
    recent_source_dir: Path | None = DEFAULT_RECENT_SOURCE_DIR,
) -> dict:
    manifest_path = source_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    reset_generated_output(output_dir)

    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    document_rows: list[dict] = []
    chunk_rows: list[dict] = []
    theme_docs: dict[str, list[dict]] = {}
    source_counts: Counter[str] = Counter()

    for row in rows:
        source_index = len(document_rows) + 1
        motion = clean_motion(row["motion"])
        raw_path = source_dir / row["raw_text_path"]
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        body = clean_transcript(raw_text, motion)
        doc_id = transcript_id(row)
        add_document(
            output_dir=output_dir,
            document_rows=document_rows,
            chunk_rows=chunk_rows,
            theme_docs=theme_docs,
            source_index=source_index,
            doc_id=doc_id,
            title_code=row["title_code"],
            source_title=row["source_title"],
            motion=motion,
            body=body,
            source_raw_text_path=raw_path,
            source_markdown_path=source_dir / row["markdown_path"],
            corpus_source="notebooklm_export",
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            views_from_title=int(row["views_from_title"]) if row["views_from_title"] else None,
            extra_metadata={
                "exported_at": row["exported_at"],
                "original_source_index": int(row["source_index"]),
            },
        )
        source_counts["notebooklm_export"] += 1

    if recent_source_dir and recent_source_dir.exists():
        curated_manifest = recent_source_dir / "curated_major_manifest.json"
        if curated_manifest.exists():
            recent_rows = json.loads(curated_manifest.read_text(encoding="utf-8"))
            for row in recent_rows:
                plain_text_path = row.get("plain_text_path")
                if not plain_text_path:
                    continue
                raw_path = recent_source_dir / plain_text_path
                if not raw_path.exists():
                    continue
                raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
                motion, motion_needs_review = infer_motion_from_recent(row, raw_text)
                competition = normalize_text(str(row.get("inferred_competition") or "YouTube"))
                year = str(row.get("inferred_year") or "")
                title_code = slugify(f"{year}-{competition}", max_len=28).upper().replace("-", "_")
                add_document(
                    output_dir=output_dir,
                    document_rows=document_rows,
                    chunk_rows=chunk_rows,
                    theme_docs=theme_docs,
                    source_index=len(document_rows) + 1,
                    doc_id=f"yt_{row.get('id')}",
                    title_code=title_code,
                    source_title=str(row.get("title") or raw_path.name),
                    motion=motion,
                    body=raw_text,
                    source_raw_text_path=raw_path,
                    source_markdown_path=None,
                    corpus_source="youtube_recent_2022_2026",
                    chunk_words=chunk_words,
                    overlap_words=overlap_words,
                    views_from_title=int(row["view_count"]) if row.get("view_count") else None,
                    extra_metadata={
                        "youtube_url": row.get("url") or "",
                        "competition": competition,
                        "year": year,
                        "round": row.get("inferred_round") or "",
                        "duration_string": row.get("duration_string") or "",
                        "motion_needs_review": motion_needs_review,
                        "curation_status": row.get("curation_status") or "",
                        "exported_at": row.get("exported_at") or "",
                    },
                )
                source_counts["youtube_recent_2022_2026"] += 1

    next_source_index = len(document_rows) + 1
    incoming_files = sorted(
        path
        for path in (output_dir / "incoming").glob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".md"}
        and path.name.lower() != "readme.md"
        and not path.name.startswith(".")
    )
    for incoming_offset, source_path in enumerate(incoming_files):
        source_index = next_source_index + incoming_offset
        raw_text = source_path.read_text(encoding="utf-8", errors="replace")
        motion, body = extract_incoming_motion(raw_text, source_path)
        doc_id = f"incoming_{source_path.stem}"
        add_document(
            output_dir=output_dir,
            document_rows=document_rows,
            chunk_rows=chunk_rows,
            theme_docs=theme_docs,
            source_index=source_index,
            doc_id=doc_id,
            title_code="INCOMING",
            source_title=source_path.name,
            motion=motion,
            body=body,
            source_raw_text_path=source_path,
            source_markdown_path=source_path,
            corpus_source="incoming",
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            extra_metadata={"exported_at": datetime.now(UTC).isoformat()},
        )
        source_counts["incoming"] += 1

    manifest_fields = [
        "transcript_id",
        "source_index",
        "title_code",
        "views_from_title",
        "themes",
        "motion",
        "word_count",
        "character_count",
        "canonical_path",
        "source_raw_text_path",
        "content_hash",
        "corpus_source",
        "youtube_url",
        "competition",
        "year",
        "round",
        "motion_needs_review",
    ]
    write_csv(output_dir / "manifest.csv", document_rows, manifest_fields)
    write_jsonl(output_dir / "manifest.jsonl", document_rows)
    write_jsonl(output_dir / "chunks" / "search_chunks.jsonl", chunk_rows)

    theme_summary_rows = []
    for theme, docs in sorted(theme_docs.items()):
        theme_summary_rows.append(
            {
                "theme": theme,
                "document_count": len(docs),
                "word_count": sum(int(doc["word_count"]) for doc in docs),
            }
        )
        lines = [f"# {theme.replace('_', ' ').title()}", ""]
        for doc in sorted(docs, key=lambda item: int(item["source_index"])):
            rel_path = Path(doc["canonical_path"]).relative_to(output_dir)
            lines.append(f"- [{doc['transcript_id']}]({rel_path}) — {doc['motion']}")
        (output_dir / "indexes" / "by_theme" / f"{theme}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    write_csv(
        output_dir / "indexes" / "theme_summary.csv", theme_summary_rows, ["theme", "document_count", "word_count"]
    )
    source_summary_rows = []
    for source_name, count in sorted(source_counts.items()):
        source_summary_rows.append(
            {
                "corpus_source": source_name,
                "document_count": count,
                "word_count": sum(
                    int(doc["word_count"]) for doc in document_rows if doc["corpus_source"] == source_name
                ),
            }
        )
    write_csv(
        output_dir / "indexes" / "source_summary.csv",
        source_summary_rows,
        ["corpus_source", "document_count", "word_count"],
    )

    write_readme(
        output_dir,
        source_dir,
        recent_source_dir,
        source_summary_rows,
        len(document_rows),
        len(chunk_rows),
        chunk_words,
        overlap_words,
    )
    write_query_guide(output_dir)
    write_incoming_readme(output_dir)

    return {
        "documents": len(document_rows),
        "chunks": len(chunk_rows),
        "themes": len(theme_summary_rows),
        "sources": dict(source_counts),
        "output_dir": str(output_dir),
    }


def write_readme(
    output_dir: Path,
    source_dir: Path,
    recent_source_dir: Path | None,
    source_summary_rows: list[dict],
    document_count: int,
    chunk_count: int,
    chunk_words: int,
    overlap_words: int,
) -> None:
    script_path = Path(__file__).resolve()
    source_summary = "\n".join(
        f"- {row['corpus_source']}: {row['document_count']} documents, {row['word_count']} words"
        for row in source_summary_rows
    )
    readme = f"""# Prepared Debate Transcript Corpus

This folder is prepared for local research/search ingestion.

- Documents: {document_count}
- Retrieval chunks: {chunk_count}
- Chunk size: {chunk_words} words
- Chunk overlap: {overlap_words} words
- Corpus root: `{output_dir}`
- NotebookLM export root: `{source_dir}`
- Recent YouTube source root: `{recent_source_dir or ""}`
- Prep script: `{script_path}`

## Source Summary

{source_summary}

## Rebuild Command

```bash
python3 {script_path} --source-dir {source_dir} --recent-source-dir {recent_source_dir or ""} --output-dir {output_dir}
```

## Folder Layout

- `{output_dir / "documents"}`: cleaned canonical transcript documents with metadata.
- `{output_dir / "chunks" / "search_chunks.jsonl"}`: retrieval-ready chunks for vector/BM25 indexing.
- `{output_dir / "manifest.csv"}`: one row per transcript, with clean motion text and themes.
- `{output_dir / "manifest.jsonl"}`: JSONL equivalent of the manifest.
- `{output_dir / "indexes" / "by_theme"}`: lightweight theme lists for browsing.
- `{output_dir / "indexes" / "theme_summary.csv"}`: theme counts.
- `{output_dir / "indexes" / "source_summary.csv"}`: source collection counts.
- `{output_dir / "recommended_queries.md"}`: query patterns that work well with this corpus.
- `{output_dir / "incoming"}`: drop future transcript files or exports here before rerunning the prep step.

## Research Handling Recommendation

Use this corpus as a separate local data source called something like `Debate Transcript Corpus`.
For best results, retrieve 8-20 chunks from local transcripts, then ask the research agent to
synthesize argument patterns and cite `transcript_id`, motion, and chunk id. Keep web search enabled
only when the task asks for factual/current-world evidence.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_query_guide(output_dir: Path) -> None:
    guide = """# Recommended Queries for Debate Transcript Research

## Best Query Shape

Use local transcript search when you want debate technique, argument interactions, framing,
comparative burdens, or examples of how motions were actually debated.

Good pattern:

> Search the local debate transcript corpus for motions and speeches relevant to [topic]. Extract
> reusable argument frames, the strongest clash points, and examples of rebuttal interaction.
> Cite transcript id, chunk id, and the returned source URL or canonical path for each example.

## Strong Examples

- Using local transcripts only, find debate examples about university admissions, meritocracy,
  and standardized testing. Extract the main government framing, opposition framing, and core clash.
- Search the local debate corpus for arguments about authoritarian regimes and opposition movements.
  Return recurring frames, comparative mechanisms, and the best rebuttal patterns.
- Find transcript examples where teams debate technology governance, surveillance, or AI-like systems.
  Extract argument interactions, not just isolated claims.
- For a lesson on criminal justice rehabilitation, search the transcript corpus for motions about punishment,
  convicts, cultural defenses, and courts. Build a debate brief with reusable framing.
- Compare how teams handle "access vs excellence" across education-related transcripts. Cite transcript IDs
  and chunk IDs.

## When to Combine Local + Web

Use local transcripts for:

- argument structure
- framing
- clash/rebuttal patterns
- examples of how debaters model motions

Use web search for:

- current facts
- statistics
- legal/policy updates
- recent case studies

Combined query:

> First search the local debate transcript corpus for argument frames on [topic]. Then use web
> search only to update factual claims. Keep transcript-derived framing separate from external
> evidence and cite both.

## Query Details That Help

Include any of these when useful:

- `local transcripts only`
- `return transcript_id, chunk_id, and source URL/canonical path`
- `focus on argument interactions`
- `extract government model and opposition response`
- `separate principled framing from practical mechanisms`
- `find analogous motions, even if the topic wording differs`
"""
    (output_dir / "recommended_queries.md").write_text(guide, encoding="utf-8")


def write_incoming_readme(output_dir: Path) -> None:
    script_path = Path(__file__).resolve()
    readme = (
        """# Incoming Transcripts

Drop future transcript exports here before preparing the next corpus revision.

Preferred format:

- `.txt` or `.md`
- Include a first line like: `Motion: ...`
- If known, include source/title/date above the transcript.

The prep script preserves this folder when it regenerates the corpus. Any `.txt` or `.md` files
placed directly in this folder are included in the next corpus build. If no `Motion:` line is found,
the filename is used as a fallback motion/title.
"""
        + f"""

Rebuild from any terminal with:

```bash
python3 {script_path} --output-dir {output_dir}
```
"""
    )
    (output_dir / "incoming" / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--recent-source-dir", type=Path, default=DEFAULT_RECENT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-words", type=int, default=900)
    parser.add_argument("--overlap-words", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_corpus(
        args.source_dir, args.output_dir, args.chunk_words, args.overlap_words, args.recent_source_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

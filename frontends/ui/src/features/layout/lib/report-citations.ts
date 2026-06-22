// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Report citation utilities (pure functions)
 *
 * The backend rebuilds report references deterministically as:
 *
 *   ## References
 *   [1] Some Article Title [academic]: https://example.org/paper
 *   [2] example.com: https://example.com/post
 *
 * with inline `[n]` markers in the report body. These helpers:
 *  - parse the References section into a number -> entry map
 *  - rewrite inline `[n]` markers into markdown links (`#citation-n`) so the
 *    renderer can attach interactive popovers
 *  - merge reference entries with live citation events (title, source class,
 *    published date, supporting extract) into popover-ready details
 */

import type { CitationSource } from '@/features/chat/types'

/** Source classes emitted by the backend source classifier */
const KNOWN_SOURCE_CLASSES = new Set([
  'first_party',
  'primary_issuer',
  'academic',
  'authoritative_third_party',
  'trade_press',
  'vendor_marketing',
  'content_marketing',
  'forum',
  'unknown',
])

/** A single entry parsed from the report's References section */
export interface ReferenceEntry {
  number: number
  title?: string
  url?: string
  sourceClass?: string
}

/** Popover-ready citation detail (reference entry enriched with event metadata) */
export interface ReportCitationDetail {
  number: number
  title: string
  url?: string
  sourceClass?: string
  publishedDate?: string
  /** Supporting extract/snippet from the source, when artifact data carried one */
  quote?: string
}

/** Matches the start of a references/sources section (mirrors backend heading detection) */
const REFERENCE_HEADING_PATTERN =
  /^(?:#{2,6}\s+(?:(?:\d+|[A-Z])[).:-]?\s+)?(?:Sources|References)\b.*|\*\*References:?\*\*|Reference\s+List)\s*$/im

const URL_PATTERN = /https?:\/\/[^\s<>")\]]+/

/** Normalize a URL for matching reference entries to citation events */
export const normalizeCitationUrl = (url: string): string => {
  try {
    const parsed = new URL(url.trim())
    const host = parsed.hostname.toLowerCase().replace(/^www\./, '')
    const path = parsed.pathname.replace(/\/+$/, '')
    return `${host}${path}${parsed.search}`
  } catch {
    return url.trim().replace(/\/+$/, '').toLowerCase()
  }
}

/** Get hostname for display fallback ("example.com") */
export const getCitationDomain = (url: string): string => {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url.slice(0, 40)
  }
}

/**
 * Parse the References/Sources section of a report into numbered entries.
 * Returns an empty map when the report has no recognizable references section.
 */
export const extractReferenceEntries = (markdown: string): Map<number, ReferenceEntry> => {
  const entries = new Map<number, ReferenceEntry>()
  if (!markdown) return entries

  const headingMatch = REFERENCE_HEADING_PATTERN.exec(markdown)
  if (!headingMatch) return entries

  const section = markdown.slice(headingMatch.index + headingMatch[0].length)

  for (const rawLine of section.split('\n')) {
    const line = rawLine.trim()
    if (!line) continue
    // Stop at the next heading — references are expected to be the last section,
    // but defensively avoid swallowing appendix content.
    if (/^#{1,6}\s/.test(line)) break

    // Accept "[1] ...", "1. ...", "1) ..." reference line shapes
    const numberMatch = /^(?:\[(\d{1,3})\]|(\d{1,3})[.)])\s+(.+)$/.exec(line)
    if (!numberMatch) continue

    const number = Number(numberMatch[1] ?? numberMatch[2])
    if (!Number.isFinite(number) || number <= 0) continue

    let rest = numberMatch[3].trim()
    const urlMatch = URL_PATTERN.exec(rest)
    const url = urlMatch ? urlMatch[0].replace(/[.,;]+$/, '') : undefined
    if (urlMatch) {
      rest = rest.slice(0, urlMatch.index).trim()
    }

    // Trailing "[source_class]" badge before the URL separator
    let sourceClass: string | undefined
    const classMatch = /\[([a-z_]+)\]\s*:?\s*$/.exec(rest)
    if (classMatch && KNOWN_SOURCE_CLASSES.has(classMatch[1])) {
      sourceClass = classMatch[1]
      rest = rest.slice(0, classMatch.index).trim()
    }

    const title = rest.replace(/[-–—:]\s*$/, '').trim() || undefined

    entries.set(number, { number, title, url, sourceClass })
  }

  return entries
}

/**
 * Rewrite inline `[n]` citation markers into markdown links to `#citation-n`
 * so the markdown renderer can attach interactive popovers.
 *
 * Only numbers present in `referenceNumbers` are rewritten, only in the body
 * (the References section itself is left untouched), and fenced code blocks
 * are skipped. Existing markdown links (`[1](...)`) are not touched.
 */
export const linkifyCitationMarkers = (
  markdown: string,
  referenceNumbers: ReadonlySet<number>
): string => {
  if (!markdown || referenceNumbers.size === 0) return markdown

  const headingMatch = REFERENCE_HEADING_PATTERN.exec(markdown)
  const body = headingMatch ? markdown.slice(0, headingMatch.index) : markdown
  const tail = headingMatch ? markdown.slice(headingMatch.index) : ''

  // Skip fenced code blocks: only rewrite even-indexed segments.
  const segments = body.split(/(```[\s\S]*?(?:```|$))/)
  const rewritten = segments
    .map((segment, index) => {
      if (index % 2 === 1) return segment
      return segment.replace(/\[(\d{1,3})\](?!\(|:)/g, (match, num: string) => {
        const number = Number(num)
        if (!referenceNumbers.has(number)) return match
        return `[[${number}]](#citation-${number})`
      })
    })
    .join('')

  return rewritten + tail
}

/**
 * Build popover-ready citation details by merging parsed reference entries
 * with citation events captured in the store. Degrades gracefully: any field
 * other than the number may be missing.
 */
export const buildCitationDetails = (
  markdown: string,
  citations: readonly CitationSource[] | undefined
): Map<number, ReportCitationDetail> => {
  const references = extractReferenceEntries(markdown)
  const details = new Map<number, ReportCitationDetail>()
  if (references.size === 0) return details

  const citationsByUrl = new Map<string, CitationSource>()
  for (const citation of citations ?? []) {
    if (!citation?.url) continue
    const key = normalizeCitationUrl(citation.url)
    const existing = citationsByUrl.get(key)
    // Prefer entries that carry metadata
    if (!existing || (!existing.title && citation.title)) {
      citationsByUrl.set(key, citation)
    }
  }

  for (const [number, entry] of references) {
    const matched = entry.url ? citationsByUrl.get(normalizeCitationUrl(entry.url)) : undefined

    const title =
      entry.title ||
      matched?.title ||
      (entry.url ? getCitationDomain(entry.url) : `Source ${number}`)

    // A citation event's content is a supporting extract only when it differs
    // from the bare URL (citation events often carry the URL as content).
    const rawQuote = matched?.content?.trim()
    const quote =
      rawQuote && rawQuote !== matched?.url && rawQuote !== entry.url && rawQuote.length >= 24
        ? rawQuote
        : undefined

    details.set(number, {
      number,
      title,
      url: entry.url || matched?.url,
      sourceClass: entry.sourceClass || matched?.sourceClass,
      publishedDate: matched?.publishedDate,
      quote,
    })
  }

  return details
}

/** Human-readable label for a source class badge ("trade_press" -> "Trade press") */
export const formatSourceClassLabel = (sourceClass: string): string => {
  const cleaned = sourceClass.replace(/_/g, ' ').trim()
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

/** Format a published date for the popover; returns undefined when unparseable */
export const formatPublishedDate = (value?: string): string | undefined => {
  if (!value?.trim()) return undefined
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value.trim()
  return parsed.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

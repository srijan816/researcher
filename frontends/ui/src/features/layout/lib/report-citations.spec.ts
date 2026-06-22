// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect } from 'vitest'
import {
  buildCitationDetails,
  extractReferenceEntries,
  formatSourceClassLabel,
  linkifyCitationMarkers,
  normalizeCitationUrl,
} from './report-citations'
import type { CitationSource } from '@/features/chat/types'

const REPORT = `# AI Chips Market

NVIDIA leads the market [1]. Some analysts disagree [2], and others
have no view [3][1].

\`\`\`python
data = [1]  # not a citation
\`\`\`

## References
[1] NVIDIA Q3 Earnings [first_party]: https://investor.nvidia.com/q3
[2] Market analysis [trade_press]: https://www.example.com/analysis/
[3] forum.example.org: https://forum.example.org/thread
`

describe('extractReferenceEntries', () => {
  test('parses canonical backend reference lines', () => {
    const entries = extractReferenceEntries(REPORT)

    expect(entries.size).toBe(3)
    expect(entries.get(1)).toEqual({
      number: 1,
      title: 'NVIDIA Q3 Earnings',
      url: 'https://investor.nvidia.com/q3',
      sourceClass: 'first_party',
    })
    expect(entries.get(2)?.sourceClass).toBe('trade_press')
    expect(entries.get(3)?.url).toBe('https://forum.example.org/thread')
  })

  test('returns empty map when no references section exists', () => {
    expect(extractReferenceEntries('# Report\n\nNo refs here [1].').size).toBe(0)
    expect(extractReferenceEntries('').size).toBe(0)
  })

  test('supports numbered list style references and stops at next heading', () => {
    const md = `Body [1].

## Sources
1. Title One: https://one.example.com
2) Title Two: https://two.example.com

## Appendix
3. Not a reference: https://nope.example.com
`
    const entries = extractReferenceEntries(md)
    expect(entries.size).toBe(2)
    expect(entries.get(1)?.url).toBe('https://one.example.com')
    expect(entries.get(2)?.title).toBe('Title Two')
  })
})

describe('linkifyCitationMarkers', () => {
  test('rewrites inline markers to #citation-n links in the body only', () => {
    const numbers = new Set([1, 2, 3])
    const result = linkifyCitationMarkers(REPORT, numbers)

    expect(result).toContain('NVIDIA leads the market [[1]](#citation-1).')
    expect(result).toContain('[[3]](#citation-3)[[1]](#citation-1)')
    // References section is untouched
    expect(result).toContain('[1] NVIDIA Q3 Earnings [first_party]: https://investor.nvidia.com/q3')
    // Fenced code blocks are untouched
    expect(result).toContain('data = [1]  # not a citation')
  })

  test('ignores numbers without a matching reference and existing links', () => {
    const result = linkifyCitationMarkers('See [1] and [9]. Also [2](https://a.b).', new Set([1]))
    expect(result).toContain('[[1]](#citation-1)')
    expect(result).toContain('[9]')
    expect(result).not.toContain('#citation-9')
    expect(result).toContain('[2](https://a.b)')
  })

  test('returns input unchanged when there are no reference numbers', () => {
    expect(linkifyCitationMarkers(REPORT, new Set())).toBe(REPORT)
  })
})

describe('buildCitationDetails', () => {
  const citations: CitationSource[] = [
    {
      id: 'c1',
      url: 'https://investor.nvidia.com/q3/',
      content: 'NVIDIA reported record data center revenue of $14.5B in Q3, up 41% YoY.',
      timestamp: new Date('2026-01-01T00:00:00Z'),
      isCited: true,
      title: 'NVIDIA Q3 FY26 Earnings Release',
      sourceClass: 'first_party',
      publishedDate: '2025-11-20',
    },
    {
      id: 'c2',
      url: 'https://example.com/analysis',
      // content equal to URL — must NOT become a quote
      content: 'https://example.com/analysis',
      timestamp: new Date('2026-01-01T00:00:00Z'),
      isCited: true,
    },
  ]

  test('merges reference entries with citation event metadata', () => {
    const details = buildCitationDetails(REPORT, citations)

    const first = details.get(1)
    expect(first?.title).toBe('NVIDIA Q3 Earnings')
    expect(first?.url).toBe('https://investor.nvidia.com/q3')
    expect(first?.sourceClass).toBe('first_party')
    expect(first?.publishedDate).toBe('2025-11-20')
    expect(first?.quote).toContain('record data center revenue')
  })

  test('degrades to title and URL only when no event metadata matches', () => {
    const details = buildCitationDetails(REPORT, [])

    const third = details.get(3)
    expect(third?.title).toBe('forum.example.org')
    expect(third?.url).toBe('https://forum.example.org/thread')
    expect(third?.publishedDate).toBeUndefined()
    expect(third?.quote).toBeUndefined()
  })

  test('does not treat a bare URL content as a supporting quote', () => {
    const details = buildCitationDetails(REPORT, citations)
    expect(details.get(2)?.quote).toBeUndefined()
  })

  test('returns empty map for reports without references', () => {
    expect(buildCitationDetails('no refs', citations).size).toBe(0)
  })
})

describe('normalizeCitationUrl', () => {
  test('treats www/trailing-slash variants as the same source', () => {
    expect(normalizeCitationUrl('https://www.example.com/a/')).toBe(
      normalizeCitationUrl('https://example.com/a')
    )
  })
})

describe('formatSourceClassLabel', () => {
  test('humanizes snake_case classes', () => {
    expect(formatSourceClassLabel('trade_press')).toBe('Trade press')
    expect(formatSourceClassLabel('first_party')).toBe('First party')
  })
})

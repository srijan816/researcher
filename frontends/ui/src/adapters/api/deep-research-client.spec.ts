// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import { coerceSSEText, extractCitationMeta } from './deep-research-client'

describe('coerceSSEText', () => {
  test('extracts MiniMax thinking blocks as text', () => {
    expect(coerceSSEText([{ type: 'thinking', thinking: 'checking current sources', index: 0 }])).toBe(
      'checking current sources'
    )
  })

  test('drops signature-only thinking blocks', () => {
    expect(coerceSSEText([{ type: 'thinking', signature: 'abc123', index: 0 }])).toBe('')
  })

  test('drops tool protocol deltas that are rendered elsewhere', () => {
    expect(
      coerceSSEText([
        { id: 'call_1', type: 'tool_use', name: 'read_file', input: {}, index: 1 },
        { type: 'input_json_delta', partial_json: '{"path": "/shared/file.txt"}', index: 1 },
      ])
    ).toBe('')
  })

  test('keeps normal text blocks render-safe', () => {
    expect(coerceSSEText([{ type: 'text', text: 'Final answer text' }])).toBe('Final answer text')
  })
})

describe('extractCitationMeta', () => {
  test('extracts flat backend citation fields', () => {
    expect(
      extractCitationMeta({
        type: 'citation_source',
        content: 'https://example.org/post',
        url: 'https://example.org/post',
        title: '  Example Post  ',
        source_class: 'trade_press',
        citation_key: 'example-post',
        published_date: '2026-02-01',
      })
    ).toEqual({
      title: 'Example Post',
      sourceClass: 'trade_press',
      citationKey: 'example-post',
      publishedDate: '2026-02-01',
    })
  })

  test('returns undefined when no metadata is present (older events)', () => {
    expect(
      extractCitationMeta({ type: 'citation_use', content: 'https://a.b', url: 'https://a.b' })
    ).toBeUndefined()
    expect(extractCitationMeta({ title: '   ', source_class: '' })).toBeUndefined()
  })

  test('ignores non-string metadata values', () => {
    expect(extractCitationMeta({ title: 42, source_class: null, published: 'yesterday' })).toEqual({
      title: undefined,
      sourceClass: undefined,
      citationKey: undefined,
      publishedDate: 'yesterday',
    })
  })
})

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect } from 'vitest'
import { findVerificationSummary, parseVerificationSummary } from './report-verification'

const VALID = JSON.stringify({
  summary: { supported: 12, partially_supported: 3, contradicted: 1, not_addressed: 2 },
  claims: [],
})

describe('parseVerificationSummary', () => {
  test('parses a valid verification report', () => {
    expect(parseVerificationSummary(VALID)).toEqual({
      supported: 12,
      partiallySupported: 3,
      contradicted: 1,
      notAddressed: 2,
      total: 18,
    })
  })

  test('returns null for missing, malformed, or empty content', () => {
    expect(parseVerificationSummary(undefined)).toBeNull()
    expect(parseVerificationSummary('')).toBeNull()
    expect(parseVerificationSummary('not json {')).toBeNull()
    expect(parseVerificationSummary('{"claims": []}')).toBeNull()
    expect(parseVerificationSummary('{"summary": "oops"}')).toBeNull()
  })

  test('returns null when all counts are zero', () => {
    expect(parseVerificationSummary('{"summary": {"supported": 0}}')).toBeNull()
  })

  test('coerces invalid counts to zero', () => {
    expect(
      parseVerificationSummary('{"summary": {"supported": "4", "contradicted": -2, "not_addressed": null}}')
    ).toEqual({ supported: 4, partiallySupported: 0, contradicted: 0, notAddressed: 0, total: 4 })
  })
})

describe('findVerificationSummary', () => {
  test('finds the verification report among file artifacts', () => {
    const files = [
      { filename: '/shared/plan.json', content: '{}' },
      { filename: '/shared/verification_report.json', content: VALID },
    ]
    expect(findVerificationSummary(files)?.supported).toBe(12)
  })

  test('prefers the most recent matching artifact', () => {
    const files = [
      { filename: '/shared/verification_report.json', content: '{"summary": {"supported": 1}}' },
      { filename: '/shared/verification_report.json', content: VALID },
    ]
    expect(findVerificationSummary(files)?.total).toBe(18)
  })

  test('returns null when absent or unparseable', () => {
    expect(findVerificationSummary([])).toBeNull()
    expect(findVerificationSummary(undefined)).toBeNull()
    expect(
      findVerificationSummary([{ filename: '/shared/verification_report.json', content: 'nope' }])
    ).toBeNull()
  })
})

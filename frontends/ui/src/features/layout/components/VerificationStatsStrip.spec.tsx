// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { vi, describe, test, expect } from 'vitest'
import { VerificationStatsStrip } from './VerificationStatsStrip'

const mockState: { deepResearchFiles?: Array<{ filename: string; content: string }> } = {}

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (s: typeof mockState) => unknown) =>
    selector ? selector(mockState) : mockState
  ),
}))

const VALID_REPORT = JSON.stringify({
  summary: { supported: 12, partially_supported: 3, contradicted: 1, not_addressed: 2 },
  claims: [],
})

describe('VerificationStatsStrip', () => {
  test('renders nothing when no verification report file exists', () => {
    mockState.deepResearchFiles = [{ filename: '/shared/plan.json', content: '{}' }]
    render(<VerificationStatsStrip />)
    expect(screen.queryByTestId('verification-stats-strip')).not.toBeInTheDocument()
  })

  test('renders nothing when files are undefined (store not hydrated)', () => {
    mockState.deepResearchFiles = undefined
    render(<VerificationStatsStrip />)
    expect(screen.queryByTestId('verification-stats-strip')).not.toBeInTheDocument()
  })

  test('renders nothing for malformed verification JSON', () => {
    mockState.deepResearchFiles = [
      { filename: '/shared/verification_report.json', content: 'not json' },
    ]
    render(<VerificationStatsStrip />)
    expect(screen.queryByTestId('verification-stats-strip')).not.toBeInTheDocument()
  })

  test('renders the claims-checked summary when the report is present', () => {
    mockState.deepResearchFiles = [
      { filename: '/shared/verification_report.json', content: VALID_REPORT },
    ]
    render(<VerificationStatsStrip />)

    const strip = screen.getByTestId('verification-stats-strip')
    expect(strip).toHaveTextContent('Claims checked:')
    expect(strip).toHaveTextContent('12 supported')
    expect(strip).toHaveTextContent('3 partial')
    expect(strip).toHaveTextContent('1 contradicted')
    expect(strip).toHaveTextContent('2 unverified')
  })
})

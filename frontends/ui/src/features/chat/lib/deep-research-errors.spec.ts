// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import { isAuthRequiredDeepResearchError } from './deep-research-errors'
import { isUnavailableDeepResearchJobError } from './deep-research-errors'

describe('isUnavailableDeepResearchJobError', () => {
  it('returns true for 404', () => {
    expect(isUnavailableDeepResearchJobError(new Error('Failed to get job status: 404'))).toBe(true)
  })

  it('returns true for 410', () => {
    expect(isUnavailableDeepResearchJobError(new Error('Failed to get job status: 410'))).toBe(true)
  })

  it('returns true for "expired"', () => {
    expect(isUnavailableDeepResearchJobError(new Error('Job expired'))).toBe(true)
  })

  it('returns true for "not found"', () => {
    expect(isUnavailableDeepResearchJobError(new Error('Job not found: abc123'))).toBe(true)
  })

  it('returns false for 401', () => {
    expect(isUnavailableDeepResearchJobError(new Error('Failed to get job status: 401'))).toBe(false)
  })

  it('returns false for non-Error inputs', () => {
    expect(isUnavailableDeepResearchJobError('Failed to get job status: 404')).toBe(false)
    expect(isUnavailableDeepResearchJobError(null)).toBe(false)
  })
})

describe('isAuthRequiredDeepResearchError', () => {
  it('returns true for 401', () => {
    expect(isAuthRequiredDeepResearchError(new Error('Failed to get job status: 401'))).toBe(true)
  })

  it('returns true for 403', () => {
    expect(isAuthRequiredDeepResearchError(new Error('Failed to get job status: 403'))).toBe(true)
  })

  it('returns false for 404', () => {
    expect(isAuthRequiredDeepResearchError(new Error('Failed to get job status: 404'))).toBe(false)
  })

  it('returns false for 500', () => {
    expect(isAuthRequiredDeepResearchError(new Error('Failed to get job status: 500'))).toBe(false)
  })

  it('returns false for non-Error inputs', () => {
    expect(isAuthRequiredDeepResearchError('Failed to get job status: 401')).toBe(false)
    expect(isAuthRequiredDeepResearchError({ message: '401' })).toBe(false)
  })
})

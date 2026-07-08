// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import {
  hasValidSessionToken,
  resolveCallbackUrl,
  shouldRedirectToSignIn,
} from './proxy-guard'

describe('proxy-guard', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-02T12:00:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('shouldRedirectToSignIn', () => {
    test('redirects app pages', () => {
      expect(shouldRedirectToSignIn('/')).toBe(true)
      expect(shouldRedirectToSignIn('/batch')).toBe(true)
    })

    test('skips API and auth routes', () => {
      expect(shouldRedirectToSignIn('/api/jobs/async/jobs')).toBe(false)
      expect(shouldRedirectToSignIn('/api/auth/signin')).toBe(false)
      expect(shouldRedirectToSignIn('/auth/signin')).toBe(false)
      expect(shouldRedirectToSignIn('/auth/error')).toBe(false)
    })
  })

  describe('hasValidSessionToken', () => {
    test('accepts a valid token', () => {
      const expiresAt = Math.floor(Date.now() / 1000) + 3600
      expect(
        hasValidSessionToken({
          idToken: 'valid-token',
          expiresAt,
        } as never)
      ).toBe(true)
    })

    test('rejects missing or expired tokens', () => {
      expect(hasValidSessionToken(null)).toBe(false)
      expect(
        hasValidSessionToken({
          idToken: 'expired-token',
          expiresAt: Math.floor(Date.now() / 1000) - 10,
        } as never)
      ).toBe(false)
    })
  })

  describe('resolveCallbackUrl', () => {
    test('keeps safe relative paths', () => {
      expect(resolveCallbackUrl('/batch?tab=queue')).toBe('/batch?tab=queue')
    })

    test('rejects open redirects', () => {
      expect(resolveCallbackUrl('https://evil.example')).toBe('/')
      expect(resolveCallbackUrl('//evil.example')).toBe('/')
      expect(resolveCallbackUrl(undefined)).toBe('/')
    })
  })
})
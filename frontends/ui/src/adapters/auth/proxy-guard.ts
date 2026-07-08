// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { JWT } from 'next-auth/jwt'
import { getIdTokenCookieDecision } from './id-token-cookie'

/** API and auth pages manage their own responses; never redirect them to sign-in. */
export const shouldRedirectToSignIn = (pathname: string): boolean => {
  if (pathname.startsWith('/api/')) return false
  if (pathname.startsWith('/auth/')) return false
  return true
}

export const hasValidSessionToken = (token: JWT | null): boolean => {
  if (!token) return false

  return (
    getIdTokenCookieDecision({
      tokenError: token.error,
      idToken: token.idToken,
      expiresAt: token.expiresAt as number | undefined,
    }) === 'set'
  )
}

/** Restrict callback URLs to same-origin relative paths. */
export const resolveCallbackUrl = (raw: string | null | undefined): string => {
  if (!raw || !raw.startsWith('/') || raw.startsWith('//')) return '/'
  return raw
}
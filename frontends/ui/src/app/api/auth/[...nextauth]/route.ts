// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NextAuth API Route Handler
 *
 * Handles all authentication requests:
 * - GET /api/auth/signin
 * - GET /api/auth/signout
 * - GET /api/auth/session
 * - POST /api/auth/callback/oauth
 *
 * After successful OAuth callback, sets the idToken as a cookie for backend auth.
 * This is necessary because middleware skips /api/auth/ routes.
 */

import { NextRequest, NextResponse } from 'next/server'
import NextAuth from 'next-auth'
import { decode, getToken } from 'next-auth/jwt'
import {
  authOptions,
  isAuthRequired,
  SESSION_MAX_AGE_SECONDS,
  shouldUseSecureCookies,
} from '@/adapters/auth/config'
import {
  getIdTokenCookieDecision,
  idTokenCookieMaxAgeSeconds,
} from '@/adapters/auth/id-token-cookie'

const nextAuthHandler = NextAuth(authOptions)

const clearAuthCookies = (response: NextResponse): void => {
  response.cookies.delete('idToken')
  response.cookies.delete('next-auth.session-token')
  response.cookies.delete('__Secure-next-auth.session-token')
  response.cookies.delete('next-auth.csrf-token')
  response.cookies.delete('__Host-next-auth.csrf-token')
  response.cookies.delete('next-auth.callback-url')
  response.cookies.delete('__Secure-next-auth.callback-url')
}

const cloneResponse = (response: Response): NextResponse =>
  new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: new Headers(response.headers),
  })

interface SessionCookieSource {
  idToken?: string
  expiresAt?: number
  error?: string
}

const readSessionCookieSource = async (
  response: Response
): Promise<SessionCookieSource | undefined> => {
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    return undefined
  }

  try {
    const session = (await response.clone().json()) as {
      idToken?: unknown
      idTokenExpiresAt?: unknown
      error?: unknown
    }

    return {
      idToken: typeof session.idToken === 'string' ? session.idToken : undefined,
      expiresAt: typeof session.idTokenExpiresAt === 'number' ? session.idTokenExpiresAt : undefined,
      error: typeof session.error === 'string' ? session.error : undefined,
    }
  } catch {
    return undefined
  }
}

/**
 * NextAuth wraps the session JWT in a JWE and sets it on the response as
 * ``__Secure-next-auth.session-token`` (HTTPS deployments) or
 * ``next-auth.session-token`` (plain HTTP).  After a successful sign-in, this
 * is the canonical source of truth for the freshly-minted session — the
 * request may not have any session cookie yet, and the response body for the
 * credentials callback is just ``{"url": "..."}`` with no token fields.
 *
 * We extract the raw JWE from the response's ``Set-Cookie`` headers and
 * decrypt it with the NextAuth secret.  The decrypted payload carries the
 * ``idToken`` / ``expiresAt`` that the backend needs as a forwardable
 * transport cookie.
 *
 * We try every plausible cookie name rather than trusting a single value of
 * ``shouldUseSecureCookies()`` — the helper and NextAuth's own cookie-name
 * derivation can disagree when an env override is set to a non-boolean
 * value (e.g. ``SECURE_COOKIES=`` empty).  Reading the response is the
 * single source of truth.
 */
const readResponseSessionTokenSource = async (
  response: Response
): Promise<SessionCookieSource | undefined> => {
  const candidateCookieNames = [
    '__Secure-next-auth.session-token',
    'next-auth.session-token',
  ]

  // ``getSetCookie()`` is the only spec-compliant way to read multiple
  // ``Set-Cookie`` headers — ``headers.get('set-cookie')`` collapses them.
  const setCookieHeaders =
    typeof response.headers.getSetCookie === 'function'
      ? response.headers.getSetCookie()
      : [response.headers.get('set-cookie') ?? '']

  let raw: string | undefined
  for (const name of candidateCookieNames) {
    const found = setCookieHeaders
      .map((header) => parseSetCookieHeader(header, name))
      .find((value): value is string => typeof value === 'string' && value.length > 0)
    if (found) {
      raw = found
      break
    }
  }

  if (!raw) return undefined

  try {
    const secret = process.env.NEXTAUTH_SECRET
    if (!secret) return undefined

    const payload = (await decode({ token: raw, secret })) as {
      idToken?: unknown
      expiresAt?: unknown
      error?: unknown
    } | null

    if (!payload) return undefined

    return {
      idToken: typeof payload.idToken === 'string' ? payload.idToken : undefined,
      expiresAt: typeof payload.expiresAt === 'number' ? payload.expiresAt : undefined,
      error: typeof payload.error === 'string' ? payload.error : undefined,
    }
  } catch {
    return undefined
  }
}

const parseSetCookieHeader = (header: string, name: string): string | undefined => {
  if (!header) return undefined
  // Headers may be a single string or joined — split on the standard
  // cookie separator and find the entry whose key matches ``name``.
  const entries = header.split(/,(?=\s*[!#$%&'*+\-.^_`|~0-9A-Za-z]+=)/)
  for (const entry of entries) {
    const trimmed = entry.trim()
    const eq = trimmed.indexOf('=')
    if (eq < 0) continue
    const key = trimmed.slice(0, eq).trim()
    if (key !== name) continue
    // Drop attributes (``Path``, ``HttpOnly``, …) — anything after the first
    // ``;`` is metadata, not the cookie value.
    const value = trimmed.slice(eq + 1).split(';', 1)[0].trim()
    // ``Set-Cookie`` values are URL-encoded by NextAuth — leave the
    // JWE untouched (``decode`` handles JWE inputs directly).
    return value
  }
  return undefined
}

const syncIdTokenCookie = async (
  req: NextRequest,
  response: Response,
  {
    preserveExpiredRequestToken = false,
    preferResponseSessionToken = false,
    decryptResponseSessionToken = false,
  }: {
    preserveExpiredRequestToken?: boolean
    preferResponseSessionToken?: boolean
    decryptResponseSessionToken?: boolean
  } = {}
): Promise<NextResponse> => {
  const responseSession = preferResponseSessionToken
    ? await readSessionCookieSource(response)
    : undefined
  // On the credentials callback path, NextAuth returns ``{"url": "..."}`` in
  // the response body — no token fields.  The freshly-minted session JWT is
  // only present as a ``Set-Cookie`` header on the response.  We decrypt that
  // cookie here so the next downstream request can forward the idToken.
  const responseSessionToken = decryptResponseSessionToken
    ? await readResponseSessionTokenSource(response)
    : undefined
  const newResponse = cloneResponse(response)

  try {
    const token = await getToken({
      req,
      secret: process.env.NEXTAUTH_SECRET,
      secureCookie: shouldUseSecureCookies(),
    })

    // Priority of sources, highest to lowest:
    //   1. ``/api/auth/session`` response body (refreshed-session path)
    //   2. ``Set-Cookie`` JWE decrypted from this response (callback path)
    //   3. ``getToken(req)`` — request-side session cookie
    // The callback path's decrypted JWE is fresher than the request-side
    // cookie (which is absent on first sign-in), so it wins when present.
    const hasResponseSessionCookieSource =
      responseSession?.idToken !== undefined ||
      responseSession?.expiresAt !== undefined ||
      responseSession?.error !== undefined
    const expiresAt =
      responseSession?.expiresAt ??
      responseSessionToken?.expiresAt ??
      (token?.expiresAt as number | undefined)
    const idToken =
      responseSession?.idToken ??
      responseSessionToken?.idToken ??
      (token?.idToken as string | undefined)
    const tokenError =
      (hasResponseSessionCookieSource ? responseSession?.error : token?.error) ??
      responseSessionToken?.error
    const cookieDecision = getIdTokenCookieDecision({
      // If the session response includes refreshed token fields, it is the
      // authoritative state. Do not let a stale request-side refresh error
      // shadow a successful recovery from the same /api/auth/session response.
      tokenError,
      idToken,
      expiresAt,
      preserveExpiredRequestToken,
    })

    if (cookieDecision === 'delete') {
      newResponse.cookies.delete('idToken')
      return newResponse
    }

    if (cookieDecision === 'preserve') {
      return newResponse
    }

    newResponse.cookies.set('idToken', idToken!, {
      httpOnly: true,
      sameSite: 'lax',
      path: '/',
      secure: shouldUseSecureCookies(),
      maxAge: idTokenCookieMaxAgeSeconds(expiresAt!, SESSION_MAX_AGE_SECONDS),
    })
  } catch (error) {
    console.error('[NextAuth] Error syncing idToken cookie:', error)
  }

  return newResponse
}

/**
 * Wrapper that sets idToken cookie after successful auth callback.
 * The middleware skips /api/auth/ routes, so we need to set the cookie here.
 *
 * Handles both:
 * - OAuth callbacks (GET /api/auth/callback/oauth)
 * - Credentials callbacks (POST /api/auth/callback/dev-bypass)
 */
const withIdTokenCookie = async (
  req: NextRequest,
  context: { params: Promise<{ nextauth: string[] }> }
): Promise<Response> => {
  const params = await context.params

  if (!isAuthRequired()) {
    const action = params.nextauth?.[0]

    if (action === 'session') {
      const response = NextResponse.json({}, { status: 200 })
      clearAuthCookies(response)
      return response
    }

    const response = NextResponse.json({ ok: true }, { status: 200 })
    clearAuthCookies(response)
    return response
  }

  // Run NextAuth handler first
  const response = await nextAuthHandler(req, context)
  const action = params.nextauth?.[0]

  // Check if this is a callback (OAuth GET or Credentials POST)
  const isCallback = params.nextauth?.includes('callback')
  if (action === 'session') {
    // On session requests, NextAuth may refresh the JWT and write it only to
    // the response. Prefer the session payload so idToken gets the refreshed
    // token and TTL; fall back to preserving when only the old request JWT is
    // visible.
    return syncIdTokenCookie(req, response, {
      preserveExpiredRequestToken: true,
      preferResponseSessionToken: true,
    })
  }

  if (isCallback) {
    console.log('[NextAuth] Syncing idToken cookie after callback')
    // On the credentials callback (e.g. ``/api/auth/callback/local-users``),
    // NextAuth returns ``{"url": "..."}`` in the response body and writes the
    // freshly-minted session JWT only to a ``Set-Cookie`` header.  We must
    // decrypt that JWE to populate the ``idToken`` transport cookie — the
    // request has no session cookie yet, so ``getToken(req)`` returns null
    // and we'd otherwise ``delete`` the idToken on a successful sign-in.
    return syncIdTokenCookie(req, response, {
      decryptResponseSessionToken: true,
    })
  }

  return response
}

export const GET = withIdTokenCookie
export const POST = withIdTokenCookie

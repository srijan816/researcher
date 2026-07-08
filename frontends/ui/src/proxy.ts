// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Next.js Proxy for Authentication Cookie Management
 *
 * This proxy extracts the idToken from the NextAuth JWT session and sets it
 * as a cookie on every request. The backend expects this cookie to identify users.
 *
 * When REQUIRE_AUTH=true, unauthenticated page requests are redirected to
 * /auth/signin before the app shell renders.
 *
 * IMPORTANT: Token refresh happens ONLY in NextAuth's JWT callback (config.ts),
 * not here. The proxy cannot update the NextAuth JWT session, and many OAuth
 * providers use rotating refresh tokens (each refresh invalidates the previous token).
 * If we refresh here, the new refresh_token would be lost and subsequent refreshes
 * would fail with "invalid_grant".
 *
 * The backend looks for user auth in priority order:
 * 1. Cookie: idToken (this proxy provides this)
 * 2. Env: Backend auth token
 * 3. Cached token from interactive login
 *
 * Note: In Next.js 16+, proxy.ts replaces middleware.ts and runs in Node.js
 * runtime by default, which provides full access to Node.js APIs.
 */

import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { getToken } from 'next-auth/jwt'
import type { JWT } from 'next-auth/jwt'
import {
  SESSION_MAX_AGE_SECONDS,
  isAuthRequired,
  shouldUseSecureCookies,
} from '@/adapters/auth/config'
import {
  getIdTokenCookieDecision,
  idTokenCookieMaxAgeSeconds,
} from '@/adapters/auth/id-token-cookie'
import {
  hasValidSessionToken,
  resolveCallbackUrl,
  shouldRedirectToSignIn,
} from '@/adapters/auth/proxy-guard'

const applyIdTokenCookie = (response: NextResponse, token: JWT | null): void => {
  if (!token) {
    response.cookies.delete('idToken')
    return
  }

  const expiresAt = token.expiresAt as number | undefined
  const cookieDecision = getIdTokenCookieDecision({
    tokenError: token.error,
    idToken: token.idToken,
    expiresAt,
  })

  if (cookieDecision === 'set') {
    response.cookies.set('idToken', token.idToken as string, {
      httpOnly: true,
      sameSite: 'lax',
      path: '/',
      secure: shouldUseSecureCookies(),
      maxAge: idTokenCookieMaxAgeSeconds(expiresAt!, SESSION_MAX_AGE_SECONDS),
    })
    return
  }

  response.cookies.delete('idToken')
}

export default async function proxy(req: NextRequest) {
  if (!isAuthRequired()) {
    const response = NextResponse.next()
    response.cookies.delete('idToken')
    response.cookies.delete('next-auth.session-token')
    response.cookies.delete('__Secure-next-auth.session-token')
    response.cookies.delete('next-auth.csrf-token')
    response.cookies.delete('__Host-next-auth.csrf-token')
    response.cookies.delete('next-auth.callback-url')
    response.cookies.delete('__Secure-next-auth.callback-url')
    return response
  }

  const pathname = req.nextUrl.pathname

  // Skip proxy for static files, public API routes, and NextAuth API routes
  if (
    pathname.startsWith('/_next/') ||
    pathname.startsWith('/api/public/') ||
    pathname.startsWith('/api/auth/') ||
    pathname.startsWith('/favicon.ico') ||
    pathname.startsWith('/icon.svg') ||
    // Brand logo assets must render on the signed-out sign-in page.
    pathname.startsWith('/brand/') ||
    pathname.startsWith('/public/')
  ) {
    return NextResponse.next()
  }

  try {
    const rawToken = await getToken({
      req,
      secret: process.env.NEXTAUTH_SECRET,
      secureCookie: shouldUseSecureCookies(),
    })
    const token = rawToken && typeof rawToken !== 'string' ? rawToken : null

    if (shouldRedirectToSignIn(pathname) && !hasValidSessionToken(token)) {
      const signInUrl = new URL('/auth/signin', req.url)
      const callbackPath = `${pathname}${req.nextUrl.search}`
      signInUrl.searchParams.set('callbackUrl', resolveCallbackUrl(callbackPath))
      const response = NextResponse.redirect(signInUrl)
      response.cookies.delete('idToken')
      return response
    }

    const response = NextResponse.next()
    applyIdTokenCookie(response, token)
    return response
  } catch (error) {
    console.error('[Proxy] Error processing token:', error)
    return NextResponse.next()
  }
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     *
     * Note: API auth routes are filtered dynamically in the proxy
     * function to allow NextAuth to handle them without interference
     */
    '/((?!_next/static|_next/image|favicon.ico|public).*)',
  ],
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Sign In Page
 *
 * Custom sign-in page for OAuth authentication.
 * Redirects to home when REQUIRE_AUTH=false since auth is not needed.
 */

'use client'

import { type ReactNode, Suspense, useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { signIn } from 'next-auth/react'
import { Flex, Text, Button, Card, Stack, Spinner } from '@/adapters/ui'
import { LoadingSpinner } from '@/adapters/ui/icons'
import { useAppConfig } from '@/shared/context'

const DISCLAIMER_TEXT =
  'Disclaimer: AI models generate responses and outputs based on complex algorithms and machine learning techniques, and these responses or outputs may be inaccurate, harmful, biased, or indecent. By testing this model, you assume the risk of any harm caused by any response or output of the model. We may capture interaction analytics to improve the application experience. We will not retain your content, documents or output for analysis for training, but may retain it to enable session history. For more information visit the Docs page.'

/**
 * Sign-in form content (uses useSearchParams which requires Suspense)
 */
const SignInContent = (): ReactNode => {
  const router = useRouter()
  const { authRequired, authProviderId } = useAppConfig()
  const searchParams = useSearchParams()
  const error = searchParams?.get('error') ?? null
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [rememberMe, setRememberMe] = useState(true)
  const [localError, setLocalError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Redirect to home if auth is disabled - this page is not needed
  useEffect(() => {
    if (!authRequired) {
      router.replace('/')
    }
  }, [authRequired, router])

  // Show loading while redirecting
  if (!authRequired) {
    return (
      <Flex align="center" justify="center" className="py-8">
        <Spinner size="medium" aria-label="Redirecting..." />
      </Flex>
    )
  }

  const handleSignIn = async (): Promise<void> => {
    if (authProviderId !== 'local-users') {
      await signIn(authProviderId, { callbackUrl: '/' })
      return
    }

    setIsSubmitting(true)
    setLocalError(null)
    const result = await signIn(authProviderId, {
      username,
      password,
      remember_me: rememberMe ? 'true' : 'false',
      redirect: false,
      callbackUrl: '/',
    })
    setIsSubmitting(false)
    if (result?.ok) {
      router.replace('/')
      return
    }
    setLocalError('Invalid username or password.')
  }

  // Map NextAuth error codes to user-friendly messages
  const getErrorMessage = (errorCode: string | null): string | null => {
    if (!errorCode) return null

    const errorMessages: Record<string, string> = {
      OAuthSignin:
        'OAuth configuration error. Check that OAUTH_CLIENT_ID is set in .env.local',
      OAuthCallback: 'OAuth callback error. The authentication response was invalid.',
      OAuthCreateAccount: 'Could not create user account.',
      EmailCreateAccount: 'Could not create user account.',
      Callback: 'Authentication callback error.',
      OAuthAccountNotLinked: 'This account is linked to a different sign-in method.',
      SessionRequired: 'Please sign in to access this page.',
      Default: 'An authentication error occurred. Please try again.',
      CredentialsSignin: 'Invalid username or password.',
    }

    return errorMessages[errorCode] || errorMessages.Default
  }

  const errorMessage = localError || getErrorMessage(error)

  return (
    <Stack gap="6" align="center">
      <span className="flex h-10 w-10 items-center justify-center rounded-md bg-[#20808d] text-lg font-semibold text-white">
        D
      </span>

      <Flex direction="col" gap="2" align="center">
        <Text kind="title/lg">Sign in to Deep Research</Text>
        <Text kind="body/regular/md" className="text-secondary text-center">
          Sign in to continue
        </Text>
      </Flex>

      {errorMessage && (
        <Flex
          direction="col"
          gap="2"
          className="bg-status-error-muted border-status-error w-full rounded-md border p-4"
        >
          <Text kind="label/semibold/sm" className="text-status-error">
            Authentication Error
          </Text>
          <Text kind="body/regular/sm">{errorMessage}</Text>
        </Flex>
      )}

      {authProviderId === 'local-users' && (
        <form
          className="flex w-full flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault()
            void handleSignIn()
          }}
        >
          <label className="flex flex-col gap-1">
            <Text kind="label/regular/sm" className="text-subtle">Username</Text>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              className="border-base bg-surface-raised text-primary h-11 rounded-md border px-3 text-base outline-none focus:border-accent"
            />
          </label>
          <label className="flex flex-col gap-1">
            <Text kind="label/regular/sm" className="text-subtle">Password</Text>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              autoComplete="current-password"
              className="border-base bg-surface-raised text-primary h-11 rounded-md border px-3 text-base outline-none focus:border-accent"
            />
          </label>
          <label className="flex items-center gap-2">
            <input
              checked={rememberMe}
              onChange={(event) => setRememberMe(event.target.checked)}
              type="checkbox"
              className="border-base text-accent h-4 w-4 rounded"
            />
            <Text kind="label/regular/sm" className="text-subtle">
              Remember me on this device
            </Text>
          </label>
          <Button
            kind="primary"
            size="large"
            type="submit"
            disabled={isSubmitting || !username.trim() || !password}
            className="w-full"
          >
            Sign in
          </Button>
        </form>
      )}

      {authProviderId !== 'local-users' && (
        <Button kind="primary" size="large" onClick={() => void handleSignIn()} className="w-full">
          Sign in
        </Button>
      )}

      <Text kind="body/regular/sm" className="text-subtle text-center">
        By signing in, you agree to the terms of service and privacy policy.
      </Text>
    </Stack>
  )
}

/**
 * Sign In Page
 */
const SignInPage = (): ReactNode => {
  return (
    <Flex
      direction="col"
      align="center"
      justify="center"
      className="bg-surface-sunken relative min-h-screen p-8"
    >
      {/* Card content */}
      <Card className="relative z-10 w-full max-w-md p-6">
        <Suspense
          fallback={
            <Flex align="center" justify="center" className="py-8">
              <LoadingSpinner size="medium" aria-label="Loading" />
            </Flex>
          }
        >
          <SignInContent />
        </Suspense>
      </Card>

      {/* Disclaimer */}
      <Text
        kind="body/regular/sm"
        className="text-subtle absolute bottom-4 left-1/2 z-10 w-full max-w-3xl -translate-x-1/2 px-4 text-center opacity-70"
      >
        {DISCLAIMER_TEXT}
      </Text>
    </Flex>
  )
}

export default SignInPage

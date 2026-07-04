// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Batch Queue Page
 *
 * Route alias for the main app with the Batch queue panel opened.
 * This prevents stale/direct /batch links from falling through to Next's 404.
 */

'use client'

import { type ReactNode, Suspense, useEffect } from 'react'
import { useAuth } from '@/adapters/auth'
import { MainLayout, useLayoutStore } from '@/features/layout'

const BatchRouteOpener = (): null => {
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)

  useEffect(() => {
    openRightPanel('batch-queue')
  }, [openRightPanel])

  return null
}

const BatchContent = (): ReactNode => {
  const { user, isAuthenticated, authRequired, signIn, signOut } = useAuth()

  return (
    <>
      <BatchRouteOpener />
      <MainLayout
        isAuthenticated={isAuthenticated}
        authRequired={authRequired}
        user={
          isAuthenticated
            ? {
                name: user?.name || undefined,
                email: user?.email || undefined,
                image: user?.image || undefined,
              }
            : undefined
        }
        onSignIn={signIn}
        onSignOut={signOut}
      />
    </>
  )
}

const BatchPage = (): ReactNode => {
  return (
    <Suspense fallback={null}>
      <BatchContent />
    </Suspense>
  )
}

export default BatchPage

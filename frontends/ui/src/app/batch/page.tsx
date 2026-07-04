// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Batch Queue Page
 *
 * Thin redirect: /batch opens the batch-queue panel in the main app.
 * The panel state lives in the (in-memory) layout store, so we set it
 * before navigating — the single batch surface stays inside MainLayout
 * rather than duplicating the layout on a standalone route.
 */

'use client'

import { type ReactNode, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useLayoutStore } from '@/features/layout'

const BatchPage = (): ReactNode => {
  const router = useRouter()
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)

  useEffect(() => {
    openRightPanel('batch-queue')
    router.replace('/')
  }, [openRightPanel, router])

  return null
}

export default BatchPage

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Application Providers
 *
 * Wraps the application with necessary providers:
 * - AppConfigProvider (runtime server-side config)
 * - ThemeProvider (KUI dark/light mode)
 * - SessionProvider (NextAuth)
 * - DeepResearchRestorer (checks for active deep research jobs on mount)
 */

'use client'

import { type ReactNode, useEffect, useRef, useState } from 'react'
import { SessionProvider } from 'next-auth/react'
import { ThemeProvider } from '@/adapters/ui'
import { listConversationSnapshots, listJobs, syncConversationSnapshots } from '@/adapters/api'
import { useAuth } from '@/adapters/auth'
import { AppConfigProvider, type AppConfig } from '@/shared/context'
import { useLayoutStore } from '@/features/layout'
import { useChatStore } from '@/features/chat/store'

interface ProvidersProps {
  children: ReactNode
  /** Runtime configuration from server-side environment variables */
  config: AppConfig
}

/**
 * Applies theme classes directly to the document element.
 * This ensures theme changes happen without remounting the component tree.
 * Defers application until after hydration to prevent SSR mismatches.
 */
const useThemeEffect = (): void => {
  const [mounted, setMounted] = useState(false)

  // Mark as mounted after first render (client-side only)
  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    // Skip during SSR and initial hydration
    if (!mounted) return

    const root = document.documentElement

    // Remove existing theme classes
    root.classList.remove('nv-light', 'nv-dark')
    root.classList.add('nv-dark')
  }, [mounted])
}

/**
 * Hook to fetch data sources on app initialization.
 * Loads available data sources from the API and updates the layout store.
 * Only web_search is enabled by default - users must manually enable other sources.
 */
const useDataSourcesInit = (): void => {
  const fetchDataSources = useLayoutStore((state) => state.fetchDataSources)
  const availableDataSources = useLayoutStore((state) => state.availableDataSources)

  useEffect(() => {
    // Only fetch if not already loaded
    if (availableDataSources === null) {
      fetchDataSources()
    }
  }, [fetchDataSources, availableDataSources])
}

/**
 * Restores per-session data source toggles after the initial API fetch.
 * On page refresh, fetchDataSources sets enabledDataSourceIds to [web_search].
 * This hook overrides that default with the stored per-session selection.
 * Waits for both availableDataSources and a hydrated conversation before restoring.
 */
const useDataSourceSessionRestore = (): void => {
  const availableDataSources = useLayoutStore((state) => state.availableDataSources)
  const setEnabledDataSources = useLayoutStore((state) => state.setEnabledDataSources)
  const conversationId = useChatStore((state) => state.currentConversation?.id)
  const restoredRef = useRef(false)

  useEffect(() => {
    if (restoredRef.current || !availableDataSources) return

    const conversation = useChatStore.getState().currentConversation
    if (!conversation) return

    const savedIds = conversation.enabledDataSourceIds
    if (savedIds && savedIds.length > 0) {
      const availableIds = new Set(availableDataSources.map((s) => s.id))
      const validIds = savedIds.filter((id) => availableIds.has(id))
      if (validIds.length > 0) {
        setEnabledDataSources(validIds)
      }
    }

    restoredRef.current = true
  }, [availableDataSources, conversationId, setEnabledDataSources])
}

/**
 * Theme wrapper that syncs with layout store.
 * Applies theme classes directly to document for instant updates.
 * Uses defer prop to prevent hydration mismatches.
 */
const ThemeWrapper = ({ children }: { children: ReactNode }): ReactNode => {
  // Apply the fixed dark theme directly to document
  useThemeEffect()

  // Initialize data sources
  useDataSourcesInit()

  // Restore per-session data source toggles after initial fetch
  useDataSourceSessionRestore()

  return (
    <ThemeProvider theme="dark" global defer>
      {children}
    </ThemeProvider>
  )
}

/**
 * Restores deep research state on conversation load.
 * - Reconnects to running/submitted jobs for page refresh recovery.
 * - Cleans up orphaned 'starting' banners by polling job status via REST.
 * Completed jobs are loaded on-demand via "View Report" click.
 */
const DeepResearchRestorer = ({ children }: { children: ReactNode }): ReactNode => {
  const [mounted, setMounted] = useState(false)
  const { user, isAuthenticated, isLoading: isAuthLoading, authRequired, idToken } = useAuth()
  const setCurrentUser = useChatStore((state) => state.setCurrentUser)
  const syncResearchHistory = useChatStore((state) => state.syncResearchHistory)
  const mergeServerConversations = useChatStore((state) => state.mergeServerConversations)
  const reconnectToActiveJob = useChatStore((state) => state.reconnectToActiveJob)
  const cleanupOrphanedStartingBanners = useChatStore((state) => state.cleanupOrphanedStartingBanners)
  const currentConversationId = useChatStore((state) => state.currentConversation?.id)
  const isDeepResearchStreaming = useChatStore((state) => state.isDeepResearchStreaming)
  const conversations = useChatStore((state) => state.conversations)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || isAuthLoading || (authRequired && !isAuthenticated)) return

    const userId = user?.id ?? 'default-user'
    let cancelled = false
    let inFlight = false

    const syncLimit = 50
    const syncIntervalMs = 45_000

    const isPageVisible = (): boolean =>
      typeof document === 'undefined' ? true : document.visibilityState !== 'hidden'

    const sync = async () => {
      if (cancelled || inFlight || !isPageVisible()) return
      inFlight = true
      try {
        if (useChatStore.getState().currentUserId !== userId) {
          setCurrentUser(userId)
        }
        const [conversationResponse, response] = await Promise.all([
          listConversationSnapshots(),
          listJobs(idToken || undefined, syncLimit),
        ])
        if (!cancelled) {
          mergeServerConversations(conversationResponse.conversations)
          syncResearchHistory(response.jobs)
        }
      } catch (error) {
        if (!cancelled) {
          console.warn('Failed to sync research history:', error)
        }
      } finally {
        inFlight = false
      }
    }

    sync()
    const intervalId = setInterval(() => {
      void sync()
    }, syncIntervalMs)
    const visibilityHandler = () => {
      if (document.visibilityState === 'visible') {
        void sync()
      }
    }
    document.addEventListener('visibilitychange', visibilityHandler)

    return () => {
      cancelled = true
      clearInterval(intervalId)
      document.removeEventListener('visibilitychange', visibilityHandler)
    }
  }, [
    mounted,
    isAuthLoading,
    authRequired,
    isAuthenticated,
    user?.id,
    idToken,
    setCurrentUser,
    syncResearchHistory,
    mergeServerConversations,
  ])

  useEffect(() => {
    if (!mounted || isAuthLoading || (authRequired && !isAuthenticated)) return
    if (!useChatStore.getState().currentUserId) return

    const timeoutId = window.setTimeout(() => {
      const state = useChatStore.getState()
      const userConversations = state.currentUserId
        ? state.conversations.filter((conversation) => conversation.userId === state.currentUserId)
        : []
      if (userConversations.length === 0) return
      syncConversationSnapshots(userConversations).catch((error) => {
        console.warn('Failed to persist conversation snapshots:', error)
      })
    }, 1200)

    return () => window.clearTimeout(timeoutId)
  }, [mounted, isAuthLoading, authRequired, isAuthenticated, conversations])

  useEffect(() => {
    if (!mounted || !currentConversationId || isDeepResearchStreaming) return

    const restore = async () => {
      await reconnectToActiveJob()
      await cleanupOrphanedStartingBanners()
    }
    restore()
  }, [mounted, currentConversationId, isDeepResearchStreaming, reconnectToActiveJob, cleanupOrphanedStartingBanners])

  return <>{children}</>
}

export const Providers = ({ children, config }: ProvidersProps): ReactNode => {
  const content = (
    <ThemeWrapper>
      <DeepResearchRestorer>{children}</DeepResearchRestorer>
    </ThemeWrapper>
  )

  return (
    <AppConfigProvider config={config}>
      <SessionProvider
        refetchInterval={config.authRequired ? config.sessionRefreshIntervalSeconds : 0}
        refetchOnWindowFocus={config.authRequired}
        refetchWhenOffline={false}
      >
        {content}
      </SessionProvider>
    </AppConfigProvider>
  )
}

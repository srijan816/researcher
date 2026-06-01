// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react'
import { render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import type { AppConfig } from '@/shared/context'
import { Providers } from './providers'

const sessionProviderProps: Array<Record<string, unknown>> = []

const layoutState = {
  theme: 'dark',
  fetchDataSources: vi.fn(),
  availableDataSources: [] as Array<{ id: string }>,
  setEnabledDataSources: vi.fn(),
}

const chatState = {
  currentUserId: 'default-user',
  conversations: [],
  currentConversation: null as { id: string; enabledDataSourceIds?: string[] } | null,
  setCurrentUser: vi.fn(),
  syncResearchHistory: vi.fn(),
  mergeServerConversations: vi.fn(),
  reconnectToActiveJob: vi.fn(),
  cleanupOrphanedStartingBanners: vi.fn(),
  isDeepResearchStreaming: false,
}

vi.mock('next-auth/react', () => ({
  useSession: () => ({ data: null, status: 'unauthenticated' }),
  signIn: vi.fn(),
  signOut: vi.fn(),
  SessionProvider: ({ children, ...props }: { children: ReactNode }) => {
    sessionProviderProps.push(props as Record<string, unknown>)
    return <>{children}</>
  },
}))

vi.mock('@/adapters/api', () => ({
  listConversationSnapshots: vi.fn().mockResolvedValue({ conversations: [] }),
  listJobs: vi.fn().mockResolvedValue({ jobs: [] }),
  syncConversationSnapshots: vi.fn().mockResolvedValue({ conversations: [] }),
}))

vi.mock('@/adapters/ui', () => ({
  ThemeProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
}))

vi.mock('@/features/layout', () => ({
  useLayoutStore: (selector: (state: typeof layoutState) => unknown) => selector(layoutState),
}))

vi.mock('@/features/chat/store', () => ({
  useChatStore: Object.assign(
    (selector: (state: typeof chatState) => unknown) => selector(chatState),
    {
      getState: () => chatState,
    }
  ),
}))

const baseConfig: AppConfig = {
  authRequired: true,
  authProviderId: 'test-provider',
  sessionRefreshIntervalSeconds: 240,
  fileUpload: {
    acceptedTypes: '.pdf',
    acceptedMimeTypes: ['application/pdf'],
    maxTotalSizeMB: 100,
    maxFileSize: 100 * 1024 * 1024,
    maxTotalSize: 100 * 1024 * 1024,
    maxFileCount: 10,
    fileExpirationCheckIntervalHours: 0,
  },
}

describe('Providers', () => {
  beforeEach(() => {
    sessionProviderProps.length = 0
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  test('uses config-driven SessionProvider polling when auth is required', () => {
    render(
      <Providers config={baseConfig}>
        <div>content</div>
      </Providers>
    )

    const latest = sessionProviderProps.at(-1)
    expect(latest).toEqual(
      expect.objectContaining({
        refetchInterval: baseConfig.sessionRefreshIntervalSeconds,
        refetchOnWindowFocus: true,
        refetchWhenOffline: false,
      })
    )
  })

  test('disables SessionProvider polling when auth is not required', () => {
    render(
      <Providers config={{ ...baseConfig, authRequired: false }}>
        <div>content</div>
      </Providers>
    )

    const latest = sessionProviderProps.at(-1)
    expect(latest).toEqual(
      expect.objectContaining({
        refetchInterval: 0,
        refetchOnWindowFocus: false,
        refetchWhenOffline: false,
      })
    )
  })

  test('does not create duplicate interval timers in providers tree', () => {
    const setIntervalSpy = vi.spyOn(globalThis, 'setInterval')

    render(
      <Providers config={baseConfig}>
        <div>content</div>
      </Providers>
    )

    expect(setIntervalSpy).not.toHaveBeenCalled()
  })
})

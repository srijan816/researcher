// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook, act } from '@testing-library/react'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { useLoadJobData } from './use-load-job-data'

const mockGetJobStatus = vi.fn()
const mockGetJobReport = vi.fn()
const mockGetJobState = vi.fn()
const mockCreateDeepResearchClient = vi.fn()
const mockSetReportContent = vi.fn()
const mockAddDeepResearchToolCall = vi.fn()
const mockCompleteDeepResearchToolCall = vi.fn()
const mockClearDeepResearch = vi.fn()
const mockSetCurrentStatus = vi.fn()
const mockSetLoadedJobId = vi.fn()
const mockSetStreamLoaded = vi.fn()
const mockStopAllDeepResearchSpinners = vi.fn()
const mockAddErrorCard = vi.fn()
const mockCompleteDeepResearch = vi.fn()
const mockSetStreaming = vi.fn()
const mockPatchConversationMessage = vi.fn()
const mockAddDeepResearchBanner = vi.fn()
const mockOpenRightPanel = vi.fn()
const mockSetResearchPanelTab = vi.fn()
const mockSetState = vi.fn()

let mockStoreState = {
  currentConversation: {
    id: 'conv-1',
    messages: [
      {
        id: 'tracking-msg',
        role: 'assistant' as const,
        content: '',
        timestamp: new Date(),
        messageType: 'agent_response' as const,
        deepResearchJobId: 'job-404',
        deepResearchJobStatus: 'running' as const,
        isDeepResearchActive: true,
      },
      {
        id: 'starting-banner',
        role: 'assistant' as const,
        content: '',
        timestamp: new Date(),
        messageType: 'deep_research_banner' as const,
        deepResearchBannerData: { bannerType: 'starting' as const, jobId: 'job-404' },
      },
    ],
  },
  deepResearchJobId: null as string | null,
  deepResearchStreamLoaded: false,
}

vi.mock('@/adapters/api', () => ({
  getJobStatus: (...args: unknown[]) => mockGetJobStatus(...args),
  getJobReport: (...args: unknown[]) => mockGetJobReport(...args),
  getJobState: (...args: unknown[]) => mockGetJobState(...args),
  createDeepResearchClient: (...args: unknown[]) => mockCreateDeepResearchClient(...args),
}))

vi.mock('../store', () => ({
  useChatStore: Object.assign(
    vi.fn((selector?: (s: any) => any) => {
      const state = {
        setReportContent: mockSetReportContent,
        addDeepResearchToolCall: mockAddDeepResearchToolCall,
        completeDeepResearchToolCall: mockCompleteDeepResearchToolCall,
        clearDeepResearch: mockClearDeepResearch,
        setCurrentStatus: mockSetCurrentStatus,
        setLoadedJobId: mockSetLoadedJobId,
        setStreamLoaded: mockSetStreamLoaded,
        stopAllDeepResearchSpinners: mockStopAllDeepResearchSpinners,
        addErrorCard: mockAddErrorCard,
        completeDeepResearch: mockCompleteDeepResearch,
        setStreaming: mockSetStreaming,
        patchConversationMessage: mockPatchConversationMessage,
        addDeepResearchBanner: mockAddDeepResearchBanner,
      }
      return selector ? selector(state) : state
    }),
    {
      getState: vi.fn(() => mockStoreState),
      setState: (...args: unknown[]) => mockSetState(...args),
    }
  ),
}))

vi.mock('@/adapters/auth', () => ({
  useAuth: vi.fn(() => ({
    idToken: 'token-123',
  })),
}))

vi.mock('@/features/layout/store', () => ({
  useLayoutStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      openRightPanel: mockOpenRightPanel,
      setResearchPanelTab: mockSetResearchPanelTab,
    }
    return selector ? selector(state) : state
  }),
}))

describe('useLoadJobData', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockStoreState = {
      currentConversation: {
        id: 'conv-1',
        messages: [
          {
            id: 'tracking-msg',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'agent_response',
            deepResearchJobId: 'job-404',
            deepResearchJobStatus: 'running',
            isDeepResearchActive: true,
          },
          {
            id: 'starting-banner',
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            messageType: 'deep_research_banner',
            deepResearchBannerData: { bannerType: 'starting', jobId: 'job-404' },
          },
        ],
      },
      deepResearchJobId: null,
      deepResearchStreamLoaded: false,
    }
    mockSetState.mockImplementation((updater: unknown) => {
      if (typeof updater === 'function') {
        updater(mockStoreState)
      }
    })
  })

  test('marks unavailable job as failed when report load hits 404', async () => {
    mockGetJobStatus.mockRejectedValue(new Error('Failed to get job status: 404'))

    const { result } = renderHook(() => useLoadJobData())

    await act(async () => {
      await result.current.loadReport('job-404')
    })

    expect(mockPatchConversationMessage).toHaveBeenCalledWith(
      'conv-1',
      'tracking-msg',
      expect.objectContaining({
        deepResearchJobStatus: 'failure',
        isDeepResearchActive: false,
      })
    )
    expect(mockAddDeepResearchBanner).toHaveBeenCalledWith('failure', 'job-404', 'conv-1')
    expect(mockAddErrorCard).toHaveBeenCalledWith(
      'agent.deep_research_load_failed',
      'Failed to get job status: 404'
    )
  })

  test('opens a partial report when replay ends with a model failure after report.md', async () => {
    mockGetJobStatus.mockResolvedValue({ job_id: 'job-partial', status: 'failure', error: 'Connection error' })
    mockCreateDeepResearchClient.mockImplementation(({ callbacks }) => ({
      connect: () => {
        callbacks.onFileUpdate('report.md', '# Finished report')
        callbacks.onJobStatus('failure', 'Model call failed after 2 attempts with APIConnectionError')
      },
      disconnect: vi.fn(),
    }))

    const { result } = renderHook(() => useLoadJobData())

    await act(async () => {
      await result.current.importJobStream('job-partial')
    })

    expect(mockSetState).toHaveBeenCalled()
    expect(mockSetLoadedJobId).toHaveBeenCalledWith('job-partial')
    expect(mockSetResearchPanelTab).toHaveBeenCalledWith('report')
    expect(mockOpenRightPanel).toHaveBeenCalledWith('research')
    expect(mockAddErrorCard).not.toHaveBeenCalled()
  })

  test('preserves replayed event timestamps instead of stamping everything with refresh time', async () => {
    mockGetJobStatus.mockResolvedValue({ job_id: 'job-history', status: 'success' })
    mockCreateDeepResearchClient.mockImplementation(({ callbacks }) => ({
      connect: () => {
        callbacks.onWorkflowStart('researcher-agent', 'input', 'workflow-event', 'agent-1', '2026-05-28T10:00:00.000Z')
        callbacks.onToolStart(
          'advanced_web_search_tool',
          { query: 'test query' },
          'researcher-agent',
          'tool-event',
          'agent-1',
          '2026-05-28T10:01:00.000Z'
        )
        callbacks.onFileUpdate('report.md', '# Finished report', '2026-05-28T10:02:00.000Z')
        callbacks.onJobStatus('success')
      },
      disconnect: vi.fn(),
    }))

    const { result } = renderHook(() => useLoadJobData())

    await act(async () => {
      await result.current.importJobStream('job-history')
    })

    const setStateUpdater = mockSetState.mock.calls.find(([arg]) => typeof arg === 'function')?.[0]
    expect(setStateUpdater).toBeInstanceOf(Function)

    const committed = (setStateUpdater as (state: typeof mockStoreState) => Record<string, unknown>)(mockStoreState)
    expect((committed.deepResearchAgents as Array<{ startedAt: Date }>)[0].startedAt.toISOString()).toBe('2026-05-28T10:00:00.000Z')
    expect((committed.deepResearchToolCalls as Array<{ timestamp: Date }>)[0].timestamp.toISOString()).toBe('2026-05-28T10:01:00.000Z')
    expect((committed.deepResearchFiles as Array<{ timestamp: Date }>)[0].timestamp.toISOString()).toBe('2026-05-28T10:02:00.000Z')
  })
})

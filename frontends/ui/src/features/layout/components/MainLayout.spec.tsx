// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { MainLayout } from './MainLayout'

const mockUpdateSessionUrl = vi.fn()
const mockClearSessionUrl = vi.fn()
const mockSelectConversation = vi.fn()
const mockStartNewSessionDraft = vi.fn()
const mockDeleteConversation = vi.fn()
const mockDeleteAllConversations = vi.fn()
const mockUpdateConversationTitle = vi.fn()
const mockCloseRightPanel = vi.fn()

// Mock the useSessionUrl hook (uses Next.js App Router hooks)
vi.mock('@/hooks/use-session-url', () => ({
  useSessionUrl: vi.fn(() => ({
    updateSessionUrl: mockUpdateSessionUrl,
    clearSessionUrl: mockClearSessionUrl,
  })),
}))

// Mock the chat store
const defaultChatState = () => ({
  currentConversation: { id: 'session-1', title: 'Test Session' },
  conversations: [],
  currentUserId: 'default-user',
  getUserConversations: vi.fn(() => []),
  selectConversation: mockSelectConversation,
  startNewSessionDraft: mockStartNewSessionDraft,
  deleteConversation: mockDeleteConversation,
  deleteAllConversations: mockDeleteAllConversations,
  updateConversationTitle: mockUpdateConversationTitle,
  isStreaming: false,
  pendingInteraction: null,
  isDeepResearchStreaming: false,
  deepResearchOwnerConversationId: null,
})

vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (state: any) => any) => {
    const state = defaultChatState()
    return selector ? selector(state) : state
  }),
  useDeepResearch: vi.fn(() => ({
    isResearching: false,
    connect: vi.fn(),
    disconnect: vi.fn(),
    cancel: vi.fn(),
  })),
  useBatchResearchQueue: vi.fn(),
  NoSourcesBanner: () => <div data-testid="no-sources-banner">No Sources Banner</div>,
}))

// Mock the layout store
vi.mock('../store', () => ({
  useLayoutStore: vi.fn((selector?: (state: any) => any) => {
    const state = {
      rightPanel: null,
      isSessionsPanelOpen: false,
      setSessionsPanelOpen: vi.fn(),
      enabledDataSourceIds: ['source-1', 'source-2'],
      closeRightPanel: mockCloseRightPanel,
    }
    return selector ? selector(state) : state
  }),
}))

vi.mock('@/adapters/api', () => ({
  deleteAllConversationSnapshots: vi.fn().mockResolvedValue(undefined),
  deleteConversationSnapshot: vi.fn().mockResolvedValue(undefined),
}))

// Mock child components
vi.mock('./AppBar', () => ({
  AppBar: ({
    sessionTitle,
    onNewSession,
  }: {
    sessionTitle: string
    onNewSession?: () => void
  }) => (
    <>
      <div data-testid="app-bar">{sessionTitle}</div>
      <button type="button" onClick={onNewSession}>
        Header New Session
      </button>
    </>
  ),
}))

vi.mock('./SessionsPanel', () => ({
  SessionsPanel: () => <div data-testid="sessions-panel">Sessions Panel</div>,
}))

vi.mock('./ChatArea', () => ({
  ChatArea: () => <div data-testid="chat-area">Chat Area</div>,
}))

vi.mock('./InputArea', () => ({
  InputArea: () => <div data-testid="input-area">Input Area</div>,
}))

vi.mock('./ResearchPanel', () => ({
  ResearchPanel: () => <div data-testid="research-panel">Research Panel</div>,
}))

vi.mock('./DataSourcesPanel', () => ({
  DataSourcesPanel: () => <div data-testid="data-sources-panel">Data Sources Panel</div>,
}))

vi.mock('./SettingsPanel', () => ({
  SettingsPanel: () => <div data-testid="settings-panel">Settings Panel</div>,
}))

vi.mock('./DocsPanel', () => ({
  DocsPanel: () => <div data-testid="docs-panel">Docs Panel</div>,
}))

import { useChatStore } from '@/features/chat'
import { useLayoutStore } from '../store'

describe('MainLayout', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useChatStore).mockImplementation((selector?: (state: any) => any) => {
      const state = defaultChatState()
      return selector ? selector(state) : state
    })
    vi.mocked(useLayoutStore).mockImplementation((selector?: (state: any) => any) => {
      const state = {
        rightPanel: null,
        isSessionsPanelOpen: false,
        setSessionsPanelOpen: vi.fn(),
        enabledDataSourceIds: ['source-1', 'source-2'],
        closeRightPanel: mockCloseRightPanel,
      }
      return selector ? selector(state) : state
    })
  })

  test('renders all main sections', () => {
    render(<MainLayout />)

    expect(screen.getByTestId('app-bar')).toBeInTheDocument()
    expect(screen.getByTestId('sessions-panel')).toBeInTheDocument()
    expect(screen.getByTestId('chat-area')).toBeInTheDocument()
    expect(screen.getByTestId('input-area')).toBeInTheDocument()
    expect(screen.getByTestId('research-panel')).toBeInTheDocument()
    expect(screen.getByTestId('data-sources-panel')).toBeInTheDocument()
    expect(screen.getByTestId('settings-panel')).toBeInTheDocument()
  })

  test('passes session title to AppBar', () => {
    render(<MainLayout />)

    expect(screen.getByTestId('app-bar')).toHaveTextContent('Test Session')
  })

  test('shows "New Session" when no current conversation', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (state: any) => any) => {
      const state = { ...defaultChatState(), currentConversation: null }
      return selector ? selector(state) : state
    })

    render(<MainLayout />)

    expect(screen.getByTestId('app-bar')).toHaveTextContent('New Session')
  })

  test('passes auth state to components', () => {
    const onSignIn = vi.fn()
    const onSignOut = vi.fn()
    const user = { name: 'Test User', email: 'test@example.com' }

    render(
      <MainLayout isAuthenticated={true} user={user} onSignIn={onSignIn} onSignOut={onSignOut} />
    )

    // Components render - props are passed to mocked child components
    expect(screen.getByTestId('app-bar')).toBeInTheDocument()
    expect(screen.getByTestId('chat-area')).toBeInTheDocument()
    expect(screen.getByTestId('input-area')).toBeInTheDocument()
  })

  test('wires the AppBar new session action to draft session flow', async () => {
    const user = userEvent.setup()

    render(<MainLayout />)

    await user.click(screen.getByRole('button', { name: /header new session/i }))

    expect(mockStartNewSessionDraft).toHaveBeenCalledOnce()
    expect(mockClearSessionUrl).toHaveBeenCalledOnce()
    expect(mockCloseRightPanel).toHaveBeenCalledOnce()
  })

  test('keeps new session action available while shallow streaming is active', async () => {
    const user = userEvent.setup()
    const startNewSessionDraft = vi.fn()

    vi.mocked(useChatStore).mockImplementation((selector?: (state: any) => any) => {
      const state = {
        ...defaultChatState(),
        startNewSessionDraft,
        isStreaming: true,
      }
      return selector ? selector(state) : state
    })

    render(<MainLayout />)

    const newSessionButton = screen.getByRole('button', { name: /header new session/i })
    expect(newSessionButton).not.toBeDisabled()
    await user.click(newSessionButton)
    expect(startNewSessionDraft).toHaveBeenCalledOnce()
  })

  test('adjusts chat width when details panel is open', () => {
    vi.mocked(useLayoutStore).mockImplementation((selector?: (state: any) => any) => {
      const state = {
        rightPanel: 'research',
        isSessionsPanelOpen: false,
        setSessionsPanelOpen: vi.fn(),
        enabledDataSourceIds: ['source-1', 'source-2'],
        closeRightPanel: mockCloseRightPanel,
      }
      return selector ? selector(state) : state
    })

    const { container } = render(<MainLayout />)

    // The chat container should have 40% width when details panel is open
    const chatContainer = container.querySelector('[style*="width"]')
    expect(chatContainer).toHaveStyle({ width: '40%' })
  })

  test('shows full width when details panel is closed', () => {
    vi.mocked(useLayoutStore).mockImplementation((selector?: (state: any) => any) => {
      const state = {
        rightPanel: null,
        isSessionsPanelOpen: false,
        setSessionsPanelOpen: vi.fn(),
        enabledDataSourceIds: ['source-1', 'source-2'],
        closeRightPanel: mockCloseRightPanel,
      }
      return selector ? selector(state) : state
    })

    const { container } = render(<MainLayout />)

    const chatContainer = container.querySelector('[style*="width"]')
    expect(chatContainer).toHaveStyle({ width: '100%' })
  })
})

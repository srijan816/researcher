// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * MainLayout Component
 *
 * The main application layout container that orchestrates:
 * - NavRail (left, full-height primary chrome)
 * - Contextual strip (top, only inside an active session)
 * - SessionsPanel "Library" (left, slides next to the rail)
 * - ChatArea + InputArea (center, responsive width)
 * - ResearchPanel (right, pushes content - takes 60% when open)
 * - DataSourcesPanel / SettingsPanel / DocsPanel (right, overlay)
 *
 * Handles auth state to show different UI for logged-in vs logged-out users.
 */

'use client'

import { type FC, useCallback, useEffect, useMemo, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { Flex, Text } from '@/adapters/ui'
import { useReducedMotion } from '@/hooks/use-reduced-motion'
import { NavRail } from './NavRail'
import { SessionsPanel } from './SessionsPanel'
import { ChatArea } from './ChatArea'
import { InputArea } from './InputArea'
import { ResearchPanel } from './ResearchPanel'
import { DataSourcesPanel } from './DataSourcesPanel'
import { SettingsPanel } from './SettingsPanel'
import { DocsPanel } from './DocsPanel'
import { BatchQueuePanel } from './BatchQueuePanel'
import { AccountPanel } from './AccountPanel'
import { useChatStore, useDeepResearch, useBatchResearchQueue, NoSourcesBanner } from '@/features/chat'
import { hasActiveDeepResearchJob } from '@/features/chat/lib/session-activity'
import { deleteAllConversationSnapshots, deleteConversationSnapshot } from '@/adapters/api'
import { useLayoutStore } from '../store'
import { useSessionUrl } from '@/hooks/use-session-url'
import { Chat, Generate, Menu, Settings, Book, Logout } from '@/adapters/ui/icons'
import { Avatar } from '@/adapters/ui'
import { ThemeToggleButton } from './ThemeToggleButton'

const DISPLAYABLE_MESSAGE_TYPES = new Set([
  'user',
  'status',
  'prompt',
  'agent_response',
  'file',
  'file_upload_status',
  'error',
  'deep_research_banner',
])

interface MainLayoutProps {
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
  /** Whether authentication is required (false = using default user) */
  authRequired?: boolean
  /** User information for the nav rail */
  user?: {
    name?: string
    email?: string
    image?: string
  }
  /** Callback when sign in is clicked */
  onSignIn?: () => void
  /** Callback when sign out is clicked */
  onSignOut?: () => void
}

/**
 * Main application layout with all panels and regions.
 * Manages the overall structure and panel states.
 * Chat state is managed via the useChatStore.
 */
export const MainLayout: FC<MainLayoutProps> = ({
  isAuthenticated = false,
  authRequired = false,
  user,
  onSignIn,
  onSignOut,
}) => {
  const {
    currentConversation,
    conversations,
    isDeepResearchStreaming,
    deepResearchOwnerConversationId,
    currentUserId,
  } = useChatStore(useShallow((s) => ({
    currentConversation: s.currentConversation,
    conversations: s.conversations,
    isDeepResearchStreaming: s.isDeepResearchStreaming,
    deepResearchOwnerConversationId: s.deepResearchOwnerConversationId,
    currentUserId: s.currentUserId,
  })))

  const selectConversation = useChatStore((s) => s.selectConversation)
  const startNewSessionDraft = useChatStore((s) => s.startNewSessionDraft)
  const deleteConversation = useChatStore((s) => s.deleteConversation)
  const deleteAllConversations = useChatStore((s) => s.deleteAllConversations)
  const updateConversationTitle = useChatStore((s) => s.updateConversationTitle)

  const {
    rightPanel,
    isSessionsPanelOpen,
    setSessionsPanelOpen,
    openRightPanel,
    closeRightPanel,
  } = useLayoutStore(useShallow((s) => ({
    rightPanel: s.rightPanel,
    isSessionsPanelOpen: s.isSessionsPanelOpen,
    setSessionsPanelOpen: s.setSessionsPanelOpen,
    openRightPanel: s.openRightPanel,
    closeRightPanel: s.closeRightPanel,
  })))
  const isResearchPanelOpen = rightPanel === 'research'
  const prefersReducedMotion = useReducedMotion()
  const [isMobileViewport, setIsMobileViewport] = useState(false)
  // Mobile overflow "More" sheet (Settings / Account / Docs / user)
  const [isMobileMoreOpen, setIsMobileMoreOpen] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 767px)')
    const handleChange = () => setIsMobileViewport(mediaQuery.matches)
    handleChange()
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  // Deep research SSE hook - manages connection when deep research starts
  useDeepResearch()
  // Batch research queue runner - submits approved queue items one by one
  useBatchResearchQueue()

  // Sync session state with URL query parameters
  const { updateSessionUrl, clearSessionUrl } = useSessionUrl({ isAuthenticated })

  // Wrap selectConversation to also update URL
  const handleSelectSession = useCallback(
    (sessionId: string) => {
      selectConversation(sessionId)
      updateSessionUrl(sessionId)
    },
    [selectConversation, updateSessionUrl]
  )

  // Start a new unsaved draft session and clear URL until first interaction.
  const handleNewSession = useCallback(() => {
    startNewSessionDraft()
    clearSessionUrl()
    closeRightPanel()
  }, [startNewSessionDraft, clearSessionUrl, closeRightPanel])

  // Wrap deleteConversation to clear URL if deleting current session
  const handleDeleteSession = useCallback(
    (sessionId: string) => {
      const wasCurrentSession = currentConversation?.id === sessionId
      deleteConversation(sessionId)
      void deleteConversationSnapshot(sessionId).catch((error) => {
        console.warn('Failed to delete conversation snapshot:', error)
      })
      if (wasCurrentSession) {
        clearSessionUrl()
      }
    },
    [deleteConversation, currentConversation?.id, clearSessionUrl]
  )

  // Delete all sessions for the current user
  const handleDeleteAllSessions = useCallback(() => {
    deleteAllConversations()
    void deleteAllConversationSnapshots().catch((error) => {
      console.warn('Failed to delete conversation snapshots:', error)
    })
    clearSessionUrl()
  }, [deleteAllConversations, clearSessionUrl])

  const isAdmin = currentUserId === 'srijan'
  const userConversations = useMemo(
    () => currentUserId ? conversations.filter((c) => isAdmin || c.userId === currentUserId) : [],
    [conversations, currentUserId, isAdmin]
  )

  const sessions = useMemo(
    () => userConversations.map((conv) => ({
      id: conv.id,
      title: conv.title,
      userId: conv.userId,
      ownerDisplayName: conv.ownerDisplayName,
      date: conv.updatedAt,
      hasActiveDeepResearch:
        hasActiveDeepResearchJob(conv.messages) ||
        (isDeepResearchStreaming && deepResearchOwnerConversationId === conv.id),
    })),
    [userConversations, isDeepResearchStreaming, deepResearchOwnerConversationId]
  )

  const hasDisplayableMessages = useMemo(
    () =>
      (currentConversation?.messages ?? []).some((msg) => {
        const messageType = msg.messageType || (msg.role === 'user' ? 'user' : 'assistant')
        return DISPLAYABLE_MESSAGE_TYPES.has(messageType)
      }),
    [currentConversation?.messages]
  )

  const showHomeExperience =
    isAuthenticated && currentUserId === 'srijan' && !hasDisplayableMessages && !isResearchPanelOpen
  const showContextStrip = isAuthenticated && !showHomeExperience && hasDisplayableMessages

  return (
    <Flex
      className={`deep-app-shell h-[100dvh] w-full overflow-hidden ${
        showHomeExperience ? 'deep-app-shell--home' : ''
      }`}
    >
      {/* Left navigation rail - primary chrome */}
      <NavRail
        isAuthenticated={isAuthenticated}
        authRequired={authRequired}
        user={user}
        onNewSession={handleNewSession}
        onSignIn={onSignIn}
        onSignOut={onSignOut}
      />

      <Flex direction="col" className="min-w-0 flex-1 overflow-hidden">
      {/* Contextual strip - only inside an active session */}
      {showContextStrip && (
        <div
          className="deep-context-strip flex h-[var(--header-height)] shrink-0 items-center border-b border-base px-4"
          data-testid="context-strip"
        >
          <span className="gx-mono mr-3 hidden shrink-0 sm:inline">SESSION</span>
          <Text kind="body/regular/md" className="min-w-0 flex-1 truncate text-secondary">
            {currentConversation?.title || 'New Session'}
          </Text>
        </div>
      )}

      {/* Main Content Area - using explicit widths instead of flex for smoother animation */}
      <div className="relative flex min-w-0 flex-1 overflow-hidden">
        {/* Center Content: Chat + Input - Responsive to research panel */}
        <div
          className={`relative flex min-w-0 flex-col ${
            showHomeExperience ? 'deep-home-stage scrollbar-hide overflow-y-auto' : 'overflow-hidden'
          }`}
          style={{
            width: isResearchPanelOpen && !isMobileViewport ? '40%' : '100%',
            transition: prefersReducedMotion ? 'none' : 'width 600ms ease-in-out',
          }}
        >
          {/* Chat Area - Scrollable */}
          <ChatArea
            isAuthenticated={isAuthenticated}
            onSignIn={onSignIn}
            homeExperience={showHomeExperience}
          />

          {/* No sources warning - shown when no data sources or files available */}
          {!showHomeExperience && <NoSourcesBanner isAuthenticated={isAuthenticated} />}

          {/* Input Area - Fixed at bottom of chat */}
          {/* Using WebSocket mode for full HITL (human-in-the-loop) support */}
          <InputArea
            isAuthenticated={isAuthenticated}
            connectionMode="websocket"
            variant={showHomeExperience ? 'hero' : 'dock'}
          />
        </div>

        {/* Research Panel (Right) - Pushes content, takes 60% width */}
        <ResearchPanel isAuthenticated={isAuthenticated} showToggle={!showHomeExperience} />
      </div>

      {/* Overlay Panels - These slide over the content */}

      {/* Sessions Panel (Left) - Only functional when authenticated */}
      <SessionsPanel
        sessions={sessions}
        selectedSessionId={currentConversation?.id}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        onDeleteAllSessions={handleDeleteAllSessions}
        onRenameSession={updateConversationTitle}
      />

      {/* Data Sources Panel (Right) - Overlay */}
      <DataSourcesPanel />

      {/* Settings Panel (Right) - Overlay */}
      <SettingsPanel />

      {/* Docs Panel (Right) - Overlay */}
      <DocsPanel />

      {/* Batch Research Queue Panel (Right) - Overlay */}
      <BatchQueuePanel />

      {/* Account Panel (Right) - Overlay */}
      <AccountPanel />

      {/* Mobile Bottom Navigation Bar */}
      {isMobileViewport && isAuthenticated && (
        <div className="border-base bg-surface-base flex h-[var(--mobile-nav-height)] w-full shrink-0 border-t backdrop-blur-md">
          {/* Tab 1: History */}
          <button
            type="button"
            onClick={() => {
              setSessionsPanelOpen(true)
              setIsMobileMoreOpen(false)
              closeRightPanel()
            }}
            className={`flex flex-1 flex-col items-center justify-center gap-1 transition-colors ${
              isSessionsPanelOpen ? 'text-accent-primary' : 'text-subtle hover:text-primary'
            }`}
          >
            <Menu className="h-5 w-5" />
            <span className="text-[10px] font-semibold uppercase tracking-wider">History</span>
          </button>

          {/* Tab 2: Chat */}
          <button
            type="button"
            onClick={() => {
              setSessionsPanelOpen(false)
              setIsMobileMoreOpen(false)
              closeRightPanel()
            }}
            className={`flex flex-1 flex-col items-center justify-center gap-1 transition-colors ${
              !isSessionsPanelOpen && !isResearchPanelOpen
                ? 'text-accent-primary'
                : 'text-subtle hover:text-primary'
            }`}
          >
            <Chat className="h-5 w-5" />
            <span className="text-[10px] font-semibold uppercase tracking-wider">Chat</span>
          </button>

          {/* Tab 3: Research */}
          <button
            type="button"
            onClick={() => {
              setSessionsPanelOpen(false)
              setIsMobileMoreOpen(false)
              openRightPanel('research')
            }}
            className={`flex flex-1 flex-col items-center justify-center gap-1 transition-colors ${
              isResearchPanelOpen ? 'text-accent-primary' : 'text-subtle hover:text-primary'
            }`}
          >
            <Generate className="h-5 w-5" />
            <span className="text-[10px] font-semibold uppercase tracking-wider">Research</span>
          </button>

          {/* Tab 4: More (Settings / Account / Docs / user) */}
          <button
            type="button"
            onClick={() => setIsMobileMoreOpen((open) => !open)}
            aria-expanded={isMobileMoreOpen}
            aria-label="More options"
            data-testid="mobile-more-button"
            className={`flex flex-1 flex-col items-center justify-center gap-1 transition-colors ${
              isMobileMoreOpen ? 'text-accent-primary' : 'text-subtle hover:text-primary'
            }`}
          >
            <Settings className="h-5 w-5" />
            <span className="text-[10px] font-semibold uppercase tracking-wider">More</span>
          </button>
        </div>
      )}

      {/* Mobile "More" sheet */}
      {isMobileViewport && isAuthenticated && isMobileMoreOpen && (
        <div
          className="fixed inset-0 z-50"
          role="dialog"
          aria-label="More options"
          data-testid="mobile-more-sheet"
        >
          {/* Backdrop */}
          <button
            type="button"
            aria-label="Close more options"
            className="absolute inset-0 bg-black/40"
            onClick={() => setIsMobileMoreOpen(false)}
          />
          <div className="border-base bg-surface-raised absolute inset-x-0 bottom-0 max-h-[70dvh] overflow-y-auto rounded-t-2xl border-t p-4 pb-6">
            {/* Signed-in user */}
            <div className="border-base mb-3 flex items-center gap-3 border-b pb-3">
              <Avatar
                size="medium"
                src={user?.image}
                fallback={(user?.name || user?.email || 'U').charAt(0).toUpperCase()}
              />
              <div className="min-w-0 flex-1">
                <Text kind="label/bold/md" className="text-primary block truncate">
                  {user?.name || 'User'}
                </Text>
                {user?.email && (
                  <Text kind="body/regular/sm" className="text-subtle block truncate">
                    {user.email}
                  </Text>
                )}
              </div>
              <ThemeToggleButton compact />
            </div>

            {(
              [
                { label: 'Settings', panel: 'settings' as const, icon: <Settings className="h-5 w-5" /> },
                { label: 'Account', panel: 'account' as const, icon: <Avatar size="small" fallback={(user?.name || 'U').charAt(0).toUpperCase()} /> },
                { label: 'Docs', panel: 'docs' as const, icon: <Book className="h-5 w-5" /> },
              ]
            ).map((item) => (
              <button
                key={item.panel}
                type="button"
                onClick={() => {
                  setIsMobileMoreOpen(false)
                  setSessionsPanelOpen(false)
                  openRightPanel(item.panel)
                }}
                className="text-primary hover:bg-interaction-base flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm font-medium"
              >
                <span className="text-subtle flex h-6 w-6 items-center justify-center">{item.icon}</span>
                {item.label}
              </button>
            ))}

            {onSignOut && (
              <button
                type="button"
                onClick={() => {
                  setIsMobileMoreOpen(false)
                  onSignOut()
                }}
                className="text-error hover:bg-interaction-base flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm font-medium"
              >
                <span className="flex h-6 w-6 items-center justify-center">
                  <Logout className="h-5 w-5" />
                </span>
                Sign Out
              </button>
            )}
          </div>
        </div>
      )}
      </Flex>
    </Flex>
  )
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * MainLayout Component
 *
 * The main application layout container that orchestrates:
 * - AppBar (top)
 * - SessionsPanel (left, overlay)
 * - ChatArea + InputArea (center, responsive width)
 * - ResearchPanel (right, pushes content - takes 60% when open)
 * - DataSourcesPanel / SettingsPanel / DocsPanel (right, overlay)
 *
 * Handles auth state to show different UI for logged-in vs logged-out users.
 */

'use client'

import { type FC, useCallback, useEffect, useMemo, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { Flex } from '@/adapters/ui'
import { useReducedMotion } from '@/hooks/use-reduced-motion'
import { AppBar } from './AppBar'
import { SessionsPanel } from './SessionsPanel'
import { ChatArea } from './ChatArea'
import { InputArea } from './InputArea'
import { ResearchPanel } from './ResearchPanel'
import { DataSourcesPanel } from './DataSourcesPanel'
import { SettingsPanel } from './SettingsPanel'
import { DocsPanel } from './DocsPanel'
import { useChatStore, useDeepResearch, NoSourcesBanner } from '@/features/chat'
import { hasActiveDeepResearchJob } from '@/features/chat/lib/session-activity'
import { deleteAllConversationSnapshots, deleteConversationSnapshot } from '@/adapters/api'
import { useLayoutStore } from '../store'
import { useSessionUrl } from '@/hooks/use-session-url'

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
  /** User information for AppBar */
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

  const isResearchPanelOpen = useLayoutStore((s) => s.rightPanel === 'research')
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const prefersReducedMotion = useReducedMotion()
  const [isMobileViewport, setIsMobileViewport] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 767px)')
    const handleChange = () => setIsMobileViewport(mediaQuery.matches)
    handleChange()
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  // Deep research SSE hook - manages connection when deep research starts
  useDeepResearch()

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
  return (
    <Flex
      direction="col"
      className={`deep-app-shell h-[100dvh] w-full overflow-hidden ${
        showHomeExperience ? 'deep-app-shell--home' : ''
      }`}
    >
      {/* AppBar - Fixed at top */}
      <AppBar
        sessionTitle={currentConversation?.title || 'New Session'}
        isAuthenticated={isAuthenticated}
        authRequired={authRequired}
        user={user}
        onNewSession={handleNewSession}
        onSignIn={onSignIn}
        onSignOut={onSignOut}
      />

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
    </Flex>
  )
}

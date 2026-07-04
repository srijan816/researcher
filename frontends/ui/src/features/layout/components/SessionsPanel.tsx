// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SessionsPanel Component
 *
 * Left panel displaying session history with new session and delete all buttons.
 * Slides in from the left and overlays content.
 */

'use client'

import { type FC, type KeyboardEvent, memo, useCallback, useMemo, useState, useRef, useEffect } from 'react'
import { Flex, Text, Button, SidePanel } from '@/adapters/ui'
import { useShallow } from 'zustand/react/shallow'
import { Chat, Edit, Trash, Plus, Search, LoadingSpinner } from '@/adapters/ui/icons'
import { useLayoutStore } from '../store'
import { useChatStore } from '@/features/chat'
import { checkStorageHealth } from '@/features/chat/lib/storage-manager'
import { DeleteSessionConfirmationModal } from './DeleteSessionConfirmationModal'
import { DeleteAllSessionsConfirmationModal } from './DeleteAllSessionsConfirmationModal'

interface Session {
  id: string
  title: string
  userId?: string
  ownerDisplayName?: string
  date: Date
  hasActiveDeepResearch?: boolean
}

interface SessionsPanelProps {
  /** List of sessions to display */
  sessions?: Session[]
  /** Currently selected session ID */
  selectedSessionId?: string
  /** Callback when a session is selected */
  onSelectSession?: (sessionId: string) => void
  /** Callback when new session is clicked */
  onNewSession?: () => void
  /** Callback when a session is deleted */
  onDeleteSession?: (sessionId: string) => void
  /** Callback when all sessions are deleted */
  onDeleteAllSessions?: () => void
  /** Callback when a session is renamed */
  onRenameSession?: (sessionId: string, newTitle: string) => void
}

/**
 * Sessions panel with history grouped by date.
 * Opens from the left side of the screen.
 */
export const SessionsPanel: FC<SessionsPanelProps> = memo(function SessionsPanel({
  sessions = [],
  selectedSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onDeleteAllSessions,
  onRenameSession,
}) {
  const isSessionsPanelOpen = useLayoutStore((s) => s.isSessionsPanelOpen)
  const setSessionsPanelOpen = useLayoutStore((s) => s.setSessionsPanelOpen)

  const isSessionBusy = useChatStore((s) => s.isSessionBusy)
  const pruneExpiredSessions = useChatStore((s) => s.pruneExpiredSessions)
  // Navigation-specific busy check: only shallow thinking (WebSocket) and HITL prompts
  // block session switching. Deep research runs server-side and can be reconnected,
  // so it should NOT prevent navigation.
  const { isStreaming, hasPendingInteraction } = useChatStore(useShallow((s) => ({
    isStreaming: s.isStreaming,
    hasPendingInteraction: s.pendingInteraction !== null,
  })))
  const isNavigationBlocked = isStreaming || hasPendingInteraction
  const [searchQuery, setSearchQuery] = useState('')
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [deleteAllModalOpen, setDeleteAllModalOpen] = useState(false)
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null)
  const [collapsedOwnerIds, setCollapsedOwnerIds] = useState<Set<string>>(() => new Set())

  // Storage usage percentage — refreshes only when the panel opens
  const [storagePercent, setStoragePercent] = useState<number>(0)
  useEffect(() => {
    if (isSessionsPanelOpen) {
      // Drop sessions older than the backend job TTL (24h) so the list
      // never shows ghost entries whose reports are gone server-side.
      pruneExpiredSessions()
      const { percentUsed } = checkStorageHealth()
      setStoragePercent(Math.round(percentUsed))
    }
  }, [isSessionsPanelOpen, pruneExpiredSessions])

  // Deep research runs server-side, so local deletion should remain available.
  // Shallow streaming and HITL still block destructive actions in the active UI.
  const isDeleteBlocked = isNavigationBlocked

  const handleDeleteClick = useCallback((sessionId: string) => {
    setSessionToDelete(sessionId)
    setDeleteModalOpen(true)
  }, [])

  const handleConfirmDelete = useCallback(() => {
    if (sessionToDelete) {
      onDeleteSession?.(sessionToDelete)
      setSessionToDelete(null)
    }
  }, [sessionToDelete, onDeleteSession])

  const handleDeleteAllClick = useCallback(() => {
    setDeleteAllModalOpen(true)
  }, [])

  const handleConfirmDeleteAll = useCallback(() => {
    onDeleteAllSessions?.()
  }, [onDeleteAllSessions])

  const handleOpenChange = useCallback(
    (open: boolean) => {
      setSessionsPanelOpen(open)
    },
    [setSessionsPanelOpen]
  )

  const handleClose = useCallback(() => {
    setSessionsPanelOpen(false)
  }, [setSessionsPanelOpen])

  const handleNewSession = useCallback(() => {
    onNewSession?.()
    handleClose()
  }, [onNewSession, handleClose])

  const handleSessionClick = useCallback(
    (sessionId: string) => {
      onSelectSession?.(sessionId)
      handleClose()
    },
    [onSelectSession, handleClose]
  )

  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions
    const query = searchQuery.toLowerCase()
    return sessions.filter((s) => s.title.toLowerCase().includes(query))
  }, [sessions, searchQuery])

  const ownerGroups = useMemo(() => groupSessionsByOwner(filteredSessions), [filteredSessions])
  const showOwnerGroups = Object.keys(ownerGroups).length > 1
  const toggleOwner = useCallback((ownerId: string) => {
    setCollapsedOwnerIds((current) => {
      const next = new Set(current)
      if (next.has(ownerId)) {
        next.delete(ownerId)
      } else {
        next.add(ownerId)
      }
      return next
    })
  }, [])

  return (
    <SidePanel
      className="z-[70] bg-surface-sunken top-0 h-[calc(100dvh-var(--mobile-nav-height))] sm:h-[100dvh] w-screen max-w-none rounded-none border-r border-base sm:max-w-[300px] sm:rounded-none md:left-[var(--nav-rail-width)]"
      open={isSessionsPanelOpen}
      onOpenChange={handleOpenChange}
      side="left"
      bordered
      closeOnClickOutside={false}
      forceMount
      slotHeading={
        <Flex align="center" gap="2">
          <Chat />
          Library
        </Flex>
      }
      slotFooter={
        <Flex direction="col" gap="1">
          <Text kind="body/regular/xs" className="text-subtle">
            Synced to the local app database. Browser cache: {storagePercent}% used.
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            Note: Sessions and files are saved for a limited time before automatic deletion.
          </Text>
        </Flex>
      }
    >
      {/* Delete All + New Session */}
      <Flex align="center" justify="between" gap="2" className="mb-4">
        <Button
          kind="tertiary"
          size="small"
          color="danger"
          onClick={handleDeleteAllClick}
          disabled={isDeleteBlocked}
          aria-label={isDeleteBlocked ? "Delete all sessions (disabled)" : "Delete all sessions"}
          title={isDeleteBlocked ? "Cannot delete while operations are in progress" : "Delete all sessions"}
          className="min-h-10"
        >
          <Flex align="center" gap="1">
            <Trash className="h-4 w-4" />
            <Text kind="label/regular/sm">Delete All</Text>
          </Flex>
        </Button>
        <Button
          kind="tertiary"
          size="small"
          onClick={handleNewSession}
          aria-label="Start new session"
          title="Start new session"
          className="min-h-10"
        >
          <Flex align="center" gap="1">
            <Plus className="h-4 w-4" />
            <Text kind="label/regular/sm">New Session</Text>
          </Flex>
        </Button>
      </Flex>

      {/* Search */}
      <div className="relative mb-4">
        <Search className="text-subtle pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search sessions..."
          className="bg-surface-base border-base text-primary placeholder:text-subtle h-11 w-full rounded-md border pl-8 pr-3 text-base outline-none focus:border-accent-primary sm:h-9 sm:text-sm"
          aria-label="Search sessions"
        />
      </div>

      {/* Session List */}
      <Flex direction="col" className="flex-1 overflow-y-auto">
        {showOwnerGroups
          ? Object.entries(ownerGroups).map(([ownerId, ownerSessions]) => {
              const collapsed = collapsedOwnerIds.has(ownerId)
              const ownerLabel = ownerSessions[0]?.ownerDisplayName || ownerId
              return (
                <Flex key={ownerId} direction="col" gap="2" className="mb-4">
                  <button
                    type="button"
                    onClick={() => toggleOwner(ownerId)}
                    className="text-subtle hover:text-primary flex w-full items-center justify-between rounded-md px-1 py-1 text-left"
                    aria-expanded={!collapsed}
                  >
                    <Text kind="label/semibold/xs" className="uppercase">
                      {ownerLabel}
                    </Text>
                    <Text kind="body/regular/xs">
                      {collapsed ? 'Show' : 'Hide'} {ownerSessions.length}
                    </Text>
                  </button>
                  {!collapsed && (
                    <SessionDateGroups
                      sessions={ownerSessions}
                      selectedSessionId={selectedSessionId}
                      isNavigationBlocked={isNavigationBlocked}
                      isSessionBusy={isSessionBusy}
                      onSelect={handleSessionClick}
                      onDelete={handleDeleteClick}
                      onRename={onRenameSession}
                    />
                  )}
                </Flex>
              )
            })
          : <SessionDateGroups
              sessions={filteredSessions}
              selectedSessionId={selectedSessionId}
              isNavigationBlocked={isNavigationBlocked}
              isSessionBusy={isSessionBusy}
              onSelect={handleSessionClick}
              onDelete={handleDeleteClick}
              onRename={onRenameSession}
            />}

        {filteredSessions.length === 0 && (
          <Flex direction="col" align="center" justify="center" className="flex-1 py-8">
            <Text kind="body/regular/sm" className="text-subtle">
              {searchQuery.trim() ? 'No matching sessions' : 'No sessions yet'}
            </Text>
            {!searchQuery.trim() && (
              <Button kind="secondary" size="small" onClick={handleNewSession} className="mt-4">
                Start a new session
              </Button>
            )}
          </Flex>
        )}
      </Flex>

      <DeleteSessionConfirmationModal
        open={deleteModalOpen}
        onOpenChange={setDeleteModalOpen}
        onConfirm={handleConfirmDelete}
      />

      <DeleteAllSessionsConfirmationModal
        open={deleteAllModalOpen}
        onOpenChange={setDeleteAllModalOpen}
        onConfirm={handleConfirmDeleteAll}
      />
    </SidePanel>
  )
})

interface SessionDateGroupsProps {
  sessions: Session[]
  selectedSessionId?: string
  isNavigationBlocked: boolean
  isSessionBusy: (sessionId: string) => boolean
  onSelect: (sessionId: string) => void
  onDelete: (sessionId: string) => void
  onRename?: (sessionId: string, newTitle: string) => void
}

const SessionDateGroups: FC<SessionDateGroupsProps> = ({
  sessions,
  selectedSessionId,
  isNavigationBlocked,
  isSessionBusy,
  onSelect,
  onDelete,
  onRename,
}) => {
  const groupedSessions = useMemo(() => groupSessionsByDate(sessions), [sessions])
  return (
    <>
      {Object.entries(groupedSessions).map(([dateLabel, dateSessions]) => (
        <Flex key={dateLabel} direction="col" gap="2" className="mb-4">
          <span className="gx-mono">{dateLabel.toUpperCase()}</span>
          {dateSessions.map((session) => (
            <SessionItem
              key={session.id}
              session={session}
              isSelected={selectedSessionId === session.id}
              isBusy={isNavigationBlocked}
              isSessionActive={isSessionBusy(session.id)}
              onSelect={onSelect}
              onDelete={onDelete}
              onRename={onRename}
            />
          ))}
        </Flex>
      ))}
    </>
  )
}

/**
 * SessionItem Component
 *
 * Individual session item with hover-reveal edit/delete icons and inline rename.
 */
interface SessionItemProps {
  session: Session
  isSelected: boolean
  /** Navigation block: true when shallow thinking (WS) or HITL prompt is pending.
   *  Deep research does NOT block navigation since it runs server-side. */
  isBusy?: boolean
  /** Per-session block: true when this specific session has active deep research */
  isSessionActive?: boolean
  onSelect?: (sessionId: string) => void
  onDelete?: (sessionId: string) => void
  onRename?: (sessionId: string, newTitle: string) => void
}

const SessionItem: FC<SessionItemProps> = ({
  session,
  isSelected,
  isBusy = false,
  isSessionActive = false,
  onSelect,
  onDelete,
  onRename,
}) => {
  const [isHovered, setIsHovered] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState(session.title)
  const inputRef = useRef<HTMLInputElement>(null)

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  const handleClick = useCallback(() => {
    if (!isEditing && !isBusy) {
      onSelect?.(session.id)
    }
  }, [isEditing, isBusy, onSelect, session.id])

  const handleEditClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      setEditValue(session.title)
      setIsEditing(true)
    },
    [session.title]
  )

  const handleDeleteClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onDelete?.(session.id)
    },
    [onDelete, session.id]
  )

  const handleSaveRename = useCallback(() => {
    const trimmedValue = editValue.trim()
    if (trimmedValue && trimmedValue !== session.title) {
      onRename?.(session.id, trimmedValue)
    }
    setIsEditing(false)
  }, [editValue, session.id, session.title, onRename])

  const handleCancelRename = useCallback(() => {
    setEditValue(session.title)
    setIsEditing(false)
  }, [session.title])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        handleSaveRename()
      } else if (e.key === 'Escape') {
        e.preventDefault()
        handleCancelRename()
      }
    },
    [handleSaveRename, handleCancelRename]
  )

  const handleInputBlur = useCallback(() => {
    handleSaveRename()
  }, [handleSaveRename])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setEditValue(e.target.value)
  }, [])

  return (
    <div
      role="button"
      tabIndex={isBusy ? -1 : 0}
      onClick={handleClick}
      onKeyDown={(e) => e.key === 'Enter' && !isEditing && !isBusy && handleClick()}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`
        group flex min-h-12 w-full items-center gap-2 rounded-lg
        border-l-2 p-2 pl-3 text-left transition-colors
        outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand
        sm:min-h-10
        ${isBusy ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}
        ${
          isSelected
            ? 'bg-surface-raised border-l-[#5AA7FF]'
            : 'border-l-transparent hover:bg-surface-raised-50 bg-transparent'
        }
      `}
      aria-label={isBusy ? `Session: ${session.title} (processing in progress)` : `Session: ${session.title}`}
      aria-disabled={isBusy}
    >
      {isEditing ? (
        <input
          ref={inputRef}
          type="text"
          value={editValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onBlur={handleInputBlur}
          onClick={(e) => e.stopPropagation()}
          className="
            bg-surface-base border-accent-primary text-primary h-8 min-w-0 flex-1 rounded border
            px-2 py-1 text-sm outline-none
          "
          aria-label="Edit session title"
        />
      ) : (
        <>
          {/* Loading indicator for active deep research */}
          {session.hasActiveDeepResearch && (
            <LoadingSpinner
              className="shrink-0 text-accent-primary"
              aria-label="Deep research in progress"
            />
          )}

          <Text kind="body/regular/sm" className="text-primary min-w-0 flex-1 truncate">
            {session.title}
          </Text>

          {/* Mono timestamp - hidden when hover actions show */}
          {!(isHovered || isSelected) && (
            <span className="gx-mono shrink-0">
              {new Date(session.date).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false,
              })}
            </span>
          )}

          {/* Action icons - shown on hover */}
          {(isHovered || isSelected) && (
            <Flex align="center" gap="1" className="shrink-0">
              <Button
                kind="tertiary"
                size="tiny"
                onClick={handleEditClick}
                disabled={isBusy || isSessionActive}
                aria-label={isBusy || isSessionActive ? "Rename session (disabled)" : "Rename session"}
                title={isBusy || isSessionActive ? "Cannot rename while operations are in progress" : "Rename session"}
                className="min-h-8 min-w-8"
              >
                <Edit height={16} width={16} />
              </Button>
              <Button
                kind="tertiary"
                size="tiny"
                color="danger"
                onClick={handleDeleteClick}
                disabled={isBusy}
                aria-label={isBusy ? "Delete session (disabled)" : "Delete session"}
                title={isBusy ? "Cannot delete while operations are in progress" : "Delete session"}
                className="min-h-8 min-w-8"
              >
                <Trash height={16} width={16} />
              </Button>
            </Flex>
          )}
        </>
      )}
    </div>
  )
}

/**
 * Groups sessions by relative date labels (Today, Yesterday, or date string)
 */
const groupSessionsByDate = (sessions: Session[]): Record<string, Session[]> => {
  const groups: Record<string, Session[]> = {}
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)

  for (const session of sessions) {
    const sessionDate = new Date(session.date)
    let label: string

    if (isSameDay(sessionDate, today)) {
      label = 'Today'
    } else if (isSameDay(sessionDate, yesterday)) {
      label = 'Yesterday'
    } else {
      label = sessionDate.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      })
    }

    if (!groups[label]) {
      groups[label] = []
    }
    groups[label].push(session)
  }

  return groups
}

const groupSessionsByOwner = (sessions: Session[]): Record<string, Session[]> => {
  const groups: Record<string, Session[]> = {}
  for (const session of sessions) {
    const ownerId = session.userId || 'unknown'
    if (!groups[ownerId]) {
      groups[ownerId] = []
    }
    groups[ownerId].push(session)
  }
  return groups
}

const isSameDay = (d1: Date, d2: Date): boolean => {
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
  )
}

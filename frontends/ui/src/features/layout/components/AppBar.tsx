// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AppBar Component
 *
 * Top navigation bar with menu toggle, logo, session title,
 * and action buttons (Add Sources, Settings, Docs, User Avatar).
 *
 * Shows different states based on authentication:
 * - Auth disabled: Default User avatar with info tooltip (no sign in/out)
 * - Logged out: Sign In button, disabled action buttons
 * - Logged in: User avatar with dropdown menu
 */

'use client'

import { type FC, memo, useCallback, useState } from 'react'
import { Flex, Text, Button, Avatar, Popover, Divider } from '@/adapters/ui'
import { Menu, Globe, Settings, Book, Lock, Logout, ChevronRight, Info } from '@/adapters/ui/icons'
import { useLayoutStore } from '../store'

interface AppBarProps {
  /** Current session title to display */
  sessionTitle?: string
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
  /** Whether authentication is required (false = using default user) */
  authRequired?: boolean
  /** User info for avatar */
  user?: {
    name?: string
    email?: string
    image?: string
  }
  /** Callback when a new session is requested */
  onNewSession?: () => void
  /** Callback when sign in is clicked */
  onSignIn?: () => void
  /** Callback when sign out is clicked */
  onSignOut?: () => void
}

/**
 * Main navigation bar at the top of the application.
 * Controls sidebar toggles and navigation actions.
 */
export const AppBar: FC<AppBarProps> = memo(function AppBar({
  sessionTitle = 'New Session',
  isAuthenticated = false,
  authRequired = false,
  user,
  onNewSession,
  onSignIn,
  onSignOut,
}) {
  const toggleSessionsPanel = useLayoutStore((s) => s.toggleSessionsPanel)
  const isSessionsPanelOpen = useLayoutStore((s) => s.isSessionsPanelOpen)
  const setSessionsPanelOpen = useLayoutStore((s) => s.setSessionsPanelOpen)
  const rightPanel = useLayoutStore((s) => s.rightPanel)
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)

  const handleMenuClick = useCallback(() => {
    if (!isAuthenticated) return
    if (!isSessionsPanelOpen) {
      closeRightPanel()
    }
    toggleSessionsPanel()
  }, [closeRightPanel, isAuthenticated, isSessionsPanelOpen, toggleSessionsPanel])

  const handleAddSourcesClick = useCallback(() => {
    if (!isAuthenticated) return
    const { rightPanel, closeRightPanel, openRightPanel } = useLayoutStore.getState()
    setSessionsPanelOpen(false)
    if (rightPanel === 'data-sources') {
      closeRightPanel()
    } else {
      openRightPanel('data-sources')
    }
  }, [isAuthenticated, setSessionsPanelOpen])

  const handleSettingsClick = useCallback(() => {
    if (!isAuthenticated) return
    const { rightPanel, closeRightPanel, openRightPanel } = useLayoutStore.getState()
    setSessionsPanelOpen(false)
    if (rightPanel === 'settings') {
      closeRightPanel()
    } else {
      openRightPanel('settings')
    }
  }, [isAuthenticated, setSessionsPanelOpen])

  const handleDocsClick = useCallback(() => {
    setSessionsPanelOpen(false)
    if (rightPanel === 'docs') {
      closeRightPanel()
    } else {
      openRightPanel('docs')
    }
  }, [rightPanel, openRightPanel, closeRightPanel, setSessionsPanelOpen])

  const handleNewSessionClick = useCallback(() => {
    if (!isAuthenticated) return
    onNewSession?.()
  }, [isAuthenticated, onNewSession])

  const handleSignOut = useCallback(() => {
    setIsUserMenuOpen(false)
    onSignOut?.()
  }, [onSignOut])

  return (
    <header className="deep-appbar shrink-0 border-b border-base bg-surface-base">
      <Flex align="center" justify="between" className="h-[var(--header-height)] min-w-0 gap-1 px-2 sm:gap-4 sm:px-4">
        {/* Left section: New session button + Sessions toggle */}
        <Flex align="center" gap="2" className="min-w-0 flex-1">
          <Button
            kind="tertiary"
            size="small"
            onClick={handleNewSessionClick}
            disabled={!isAuthenticated}
            aria-label="Create new session"
            title="Create new session"
            className="shrink-0"
          >
            <Flex align="center" gap="density-lg">
              <span className="flex h-7 w-7 items-center justify-center rounded-[8px] bg-[#C7FF3D] text-sm font-semibold italic text-[#0B0C0E] transition-shadow duration-200 hover:shadow-[0_0_12px_rgba(199,255,61,0.35)] motion-reduce:transition-none">
                α
              </span>

              <Text kind="label/semibold/lg" className="hidden whitespace-nowrap text-primary sm:block">
                Alpha Research
              </Text>
            </Flex>
          </Button>

          <div className="shrink-0">
            <Button
              kind="tertiary"
              size="small"
              onClick={handleMenuClick}
              disabled={!isAuthenticated}
              aria-label="Toggle sessions sidebar"
              title="Toggle sessions sidebar"
            >
              <Flex align="center" gap="1">
                <Menu className="h-4 w-4" />
                <Text kind="label/regular/md" className="hidden sm:inline">History</Text>
              </Flex>
            </Button>
          </div>

          {isAuthenticated && (
            <div className="ml-4 hidden min-w-0 flex-1 items-center md:flex">
              <Text
                kind="body/regular/md"
                className="block w-full max-w-[360px] truncate text-subtle lg:max-w-[480px] xl:max-w-[560px]"
              >
                {sessionTitle}
              </Text>
            </div>
          )}
        </Flex>

        {/* Right section: Actions + User */}
        <Flex align="center" gap="1" className="shrink-0 sm:gap-2">
          <Button
            kind="tertiary"
            size="small"
            onClick={handleAddSourcesClick}
            disabled={!isAuthenticated}
            aria-label="Add data sources"
            title="Add data sources"
          >
            <Flex align="center" gap="1">
              <Globe className="h-4 w-4" />
              <Text kind="label/regular/md" className="hidden sm:inline">Data Sources</Text>
            </Flex>
          </Button>

          <Button
            kind="tertiary"
            size="small"
            onClick={handleSettingsClick}
            disabled={!isAuthenticated}
            aria-label="Open settings"
            title="Open settings"
          >
            <Flex align="center" gap="1">
              <Settings className="h-4 w-4" />
              <Text kind="label/regular/md" className="hidden sm:inline">Settings</Text>
            </Flex>
          </Button>

          <Button
            kind="tertiary"
            size="small"
            onClick={handleDocsClick}
            aria-label="Open documentation"
            title="Open documentation"
          >
            <Flex align="center" gap="1">
              <Book className="h-4 w-4" />
              <Text kind="label/regular/md" className="hidden sm:inline">Docs</Text>
              <ChevronRight className="hidden h-3 w-3 -rotate-45 sm:block" />
            </Flex>
          </Button>

          {/* User section: Auth not required notice, Avatar with dropdown, or Sign In button */}
          {!authRequired ? (
            <Popover
              open={isUserMenuOpen}
              onOpenChange={setIsUserMenuOpen}
              side="bottom"
              align="end"
              slotContent={<AuthDisabledContent />}
            >
              <Button
                kind="tertiary"
                size="small"
                aria-label="Default User - Authentication Not Configured"
                title="Default User set. Authentication Not Configured."
                className="sm:ml-2"
              >
                <Avatar size="small" fallback="D" />
              </Button>
            </Popover>
          ) : isAuthenticated ? (
            <Popover
              open={isUserMenuOpen}
              onOpenChange={setIsUserMenuOpen}
              side="bottom"
              align="end"
              slotContent={<UserDropdownContent user={user} onSignOut={handleSignOut} />}
            >
              <Button
                kind="tertiary"
                size="small"
                aria-label={`User menu for ${user?.name || user?.email || 'User'}`}
                title="User menu"
                className="sm:ml-2"
              >
                <Avatar
                  size="small"
                  src={user?.image}
                  fallback={(user?.name || user?.email || 'U').charAt(0).toUpperCase()}
                />
              </Button>
            </Popover>
          ) : (
            <Button
              kind="primary"
              size="small"
              onClick={onSignIn}
              aria-label="Sign in"
              title="Sign in"
              className="sm:ml-2 bg-[#C7FF3D] text-[#0B0C0E] hover:bg-[#A6D633]"
            >
              <Flex align="center" gap="1">
                <Lock className="h-4 w-4" />
                <Text kind="label/semibold/sm">Sign In</Text>
              </Flex>
            </Button>
          )}
        </Flex>
      </Flex>
    </header>
  )
})

/**
 * User dropdown content with profile info and sign out button
 */
interface UserDropdownContentProps {
  user?: {
    name?: string
    email?: string
    image?: string
  }
  onSignOut?: () => void
}

const UserDropdownContent: FC<UserDropdownContentProps> = ({ user, onSignOut }) => {
  return (
    <Flex direction="col" gap="3" className="min-w-[240px] p-4">
      {/* User info section */}
      <Flex align="center" gap="3">
        <Avatar
          size="medium"
          src={user?.image}
          fallback={(user?.name || user?.email || 'U').charAt(0).toUpperCase()}
        />
        <Flex direction="col" gap="1">
          <Text kind="label/bold/md" className="text-primary">
            {user?.name || 'User'}
          </Text>
          {user?.email && (
            <Text kind="body/regular/sm" className="text-subtle">
              {user.email}
            </Text>
          )}
        </Flex>
      </Flex>

      <Divider />

      {/* Sign out button */}
      <Button
        kind="secondary"
        size="small"
        onClick={onSignOut}
        className="w-full"
        aria-label="Sign out"
        title="Sign out"
      >
        <Flex align="center" justify="center" gap="2">
          <Logout className="h-4 w-4" />
          <Text kind="label/regular/sm">Sign Out</Text>
        </Flex>
      </Button>
    </Flex>
  )
}

/**
 * Content shown when authentication is disabled
 * Displays info message instead of sign out option
 */
const AuthDisabledContent: FC = () => {
  return (
    <Flex direction="col" gap="3" className="min-w-[240px] p-4">
      {/* User info section */}
      <Flex align="center" gap="3">
        <Avatar size="medium" fallback="D" />
        <Flex direction="col" gap="1">
          <Text kind="label/bold/md" className="text-primary">
            Default User
          </Text>
        </Flex>
      </Flex>

      <Divider />

      {/* Info message */}
      <Flex align="center" gap="2" className="rounded bg-[var(--background-color-surface-raised)] p-3">
        <Info className="h-4 w-4 shrink-0 text-[var(--text-color-subtle)]" />
        <Text kind="body/regular/sm" className="text-subtle">
          Authentication Not Configured
        </Text>
      </Flex>
    </Flex>
  )
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NavRail Component
 *
 * The primary chrome of GenAlphAI Research: a 72px full-height left rail with
 * the GenAlphAI mark on top, icon buttons with tiny labels (New, Library,
 * Sources, Settings, Docs) and the user avatar / sign-in at the bottom.
 *
 * Replaces the old top AppBar as primary navigation.
 */

'use client'

import { type FC, type ReactNode, memo, useCallback, useState } from 'react'
import { Flex, Text, Button, Avatar, Popover, Divider } from '@/adapters/ui'
import { Menu, Globe, Settings, Book, Lock, Logout, Plus, Info, Clock } from '@/adapters/ui/icons'
import { AnimatedWordmark } from '@/shared/components/AnimatedWordmark'
import { useChatStore } from '@/features/chat'
import { useLayoutStore } from '../store'
import { ThemeToggleButton } from './ThemeToggleButton'

/** Particle cap for the small rail-sized wordmark (keep it cheap) */
const RAIL_WORDMARK_MAX_PARTICLES = 180

interface NavRailProps {
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

interface RailButtonProps {
  label: string
  ariaLabel: string
  title?: string
  active?: boolean
  disabled?: boolean
  onClick?: () => void
  children: ReactNode
}

const RailButton: FC<RailButtonProps> = ({
  label,
  ariaLabel,
  title,
  active = false,
  disabled = false,
  onClick,
  children,
}) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    aria-label={ariaLabel}
    aria-pressed={active}
    title={title ?? ariaLabel}
    className={`gx-rail-button flex w-14 flex-col items-center justify-center gap-1 py-2 outline-none disabled:cursor-not-allowed disabled:opacity-40 ${
      active ? 'gx-rail-button--active' : ''
    }`}
  >
    {children}
    <span className="text-[10px] font-medium leading-none">{label}</span>
  </button>
)

/**
 * Left navigation rail — primary application chrome.
 */
export const NavRail: FC<NavRailProps> = memo(function NavRail({
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
  // Ambient status: the rail wordmark's α particles only animate while
  // deep research is running (same signal as the research status pill).
  const isDeepResearchStreaming = useChatStore((s) => s.isDeepResearchStreaming)
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false)

  const handleLibraryClick = useCallback(() => {
    if (!isAuthenticated) return
    if (!isSessionsPanelOpen) {
      useLayoutStore.getState().closeRightPanel()
    }
    toggleSessionsPanel()
  }, [isAuthenticated, isSessionsPanelOpen, toggleSessionsPanel])

  const togglePanel = useCallback(
    (panel: 'data-sources' | 'settings' | 'docs' | 'batch-queue' | 'account') => {
      const { rightPanel, closeRightPanel, openRightPanel } = useLayoutStore.getState()
      setSessionsPanelOpen(false)
      if (rightPanel === panel) {
        closeRightPanel()
      } else {
        openRightPanel(panel)
      }
    },
    [setSessionsPanelOpen]
  )

  const handleSourcesClick = useCallback(() => {
    if (!isAuthenticated) return
    togglePanel('data-sources')
  }, [isAuthenticated, togglePanel])

  const handleSettingsClick = useCallback(() => {
    if (!isAuthenticated) return
    togglePanel('settings')
  }, [isAuthenticated, togglePanel])

  const handleDocsClick = useCallback(() => {
    togglePanel('docs')
  }, [togglePanel])

  const handleBatchClick = useCallback(() => {
    if (!isAuthenticated) return
    togglePanel('batch-queue')
  }, [isAuthenticated, togglePanel])

  const handleAccountClick = useCallback(() => {
    setIsUserMenuOpen(false)
    togglePanel('account')
  }, [togglePanel])

  const handleNewSessionClick = useCallback(() => {
    if (!isAuthenticated) return
    onNewSession?.()
  }, [isAuthenticated, onNewSession])

  const handleSignOut = useCallback(() => {
    setIsUserMenuOpen(false)
    onSignOut?.()
  }, [onSignOut])

  return (
    <nav
      className="gx-nav-rail hidden h-full w-[var(--nav-rail-width)] shrink-0 flex-col items-center justify-between py-4 md:flex"
      aria-label="Primary"
      data-testid="nav-rail"
    >
      <Flex direction="col" align="center" gap="4" className="w-full">
        {/* GenAlphAI mark — returns home / new session */}
        <button
          type="button"
          onClick={handleNewSessionClick}
          disabled={!isAuthenticated}
          aria-label="GenAlphAI Research home"
          title="GenAlphAI Research"
          className="gx-rail-logo flex h-12 w-14 items-center justify-center outline-none disabled:cursor-not-allowed disabled:opacity-40"
        >
          <AnimatedWordmark
            className="w-[50px]"
            active={isDeepResearchStreaming}
            maxParticles={RAIL_WORDMARK_MAX_PARTICLES}
          />
        </button>

        <Flex direction="col" align="center" gap="1" className="w-full">
          <RailButton
            label="New"
            ariaLabel="Create new session"
            disabled={!isAuthenticated}
            onClick={handleNewSessionClick}
          >
            <Plus className="h-5 w-5" />
          </RailButton>

          <RailButton
            label="Library"
            ariaLabel="Toggle sessions library"
            active={isSessionsPanelOpen}
            disabled={!isAuthenticated}
            onClick={handleLibraryClick}
          >
            <Menu className="h-5 w-5" />
          </RailButton>

          <RailButton
            label="Sources"
            ariaLabel="Add data sources"
            active={rightPanel === 'data-sources'}
            disabled={!isAuthenticated}
            onClick={handleSourcesClick}
          >
            <Globe className="h-5 w-5" />
          </RailButton>

          <RailButton
            label="Batch"
            ariaLabel="Open batch research queue"
            active={rightPanel === 'batch-queue'}
            disabled={!isAuthenticated}
            onClick={handleBatchClick}
          >
            <Clock className="h-5 w-5" />
          </RailButton>

          <RailButton
            label="Settings"
            ariaLabel="Open settings"
            active={rightPanel === 'settings'}
            disabled={!isAuthenticated}
            onClick={handleSettingsClick}
          >
            <Settings className="h-5 w-5" />
          </RailButton>

          <RailButton
            label="Docs"
            ariaLabel="Open documentation"
            active={rightPanel === 'docs'}
            onClick={handleDocsClick}
          >
            <Book className="h-5 w-5" />
          </RailButton>
        </Flex>
      </Flex>

      {/* Bottom: theme toggle + user avatar / sign in */}
      <Flex direction="col" align="center" gap="2" className="w-full">
        <ThemeToggleButton compact />
        {!authRequired ? (
          <Popover
            open={isUserMenuOpen}
            onOpenChange={setIsUserMenuOpen}
            side="right"
            align="end"
            slotContent={<AuthDisabledContent />}
          >
            <button
              type="button"
              aria-label="Default User - Authentication Not Configured"
              title="Default User set. Authentication Not Configured."
              className="gx-rail-button flex h-12 w-14 items-center justify-center"
            >
              <Avatar size="small" fallback="D" />
            </button>
          </Popover>
        ) : isAuthenticated ? (
          <Popover
            open={isUserMenuOpen}
            onOpenChange={setIsUserMenuOpen}
            side="right"
            align="end"
            slotContent={<UserDropdownContent user={user} onSignOut={handleSignOut} onAccount={handleAccountClick} />}
          >
            <button
              type="button"
              aria-label={`User menu for ${user?.name || user?.email || 'User'}`}
              title="User menu"
              className="gx-rail-button flex h-12 w-14 items-center justify-center"
            >
              <Avatar
                size="small"
                src={user?.image}
                fallback={(user?.name || user?.email || 'U').charAt(0).toUpperCase()}
              />
            </button>
          </Popover>
        ) : (
          <RailButton label="Sign in" ariaLabel="Sign in" onClick={onSignIn}>
            <Lock className="h-5 w-5" />
          </RailButton>
        )}
      </Flex>
    </nav>
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
  onAccount?: () => void
}

const UserDropdownContent: FC<UserDropdownContentProps> = ({ user, onSignOut, onAccount }) => {
  return (
    <Flex direction="col" gap="3" className="min-w-[240px] p-4">
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

      <Button
        kind="secondary"
        size="small"
        onClick={onAccount}
        className="w-full"
        aria-label="Open account settings"
        title="Account"
      >
        <Flex align="center" justify="center" gap="2">
          <Settings className="h-4 w-4" />
          <Text kind="label/regular/sm">Account</Text>
        </Flex>
      </Button>

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
 */
const AuthDisabledContent: FC = () => {
  return (
    <Flex direction="col" gap="3" className="min-w-[240px] p-4">
      <Flex align="center" gap="3">
        <Avatar size="medium" fallback="D" />
        <Flex direction="col" gap="1">
          <Text kind="label/bold/md" className="text-primary">
            Default User
          </Text>
        </Flex>
      </Flex>

      <Divider />

      <Flex align="center" gap="2" className="rounded bg-[var(--background-color-surface-raised)] p-3">
        <Info className="h-4 w-4 shrink-0 text-[var(--text-color-subtle)]" />
        <Text kind="body/regular/sm" className="text-subtle">
          Authentication Not Configured
        </Text>
      </Flex>
    </Flex>
  )
}

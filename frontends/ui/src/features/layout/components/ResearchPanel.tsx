// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ResearchPanel Component
 *
 * Right-side panel showing Thinking, Citations, or Report content.
 * Includes top action bar with tabs.
 *
 * This panel PUSHES the chat area (takes 60% width) rather than overlaying it.
 */

'use client'

import { type FC, type ReactNode, memo, useCallback, useEffect, useMemo, useState } from 'react'
import { Flex, Button, SegmentedControl, Spinner, Text } from '@/adapters/ui'
import { Close, Generate, StopCircle } from '@/adapters/ui/icons'
import { useCancelDeepResearchJob, useChatStore, useLoadJobData } from '@/features/chat'
import { useReducedMotion } from '@/hooks/use-reduced-motion'
import { useLayoutStore } from '../store'
import { BatchResearchQueue } from './BatchResearchQueue'
import { PlanTab } from './PlanTab'
import { TasksTab } from './TasksTab'
import { ThinkingTab } from './ThinkingTab'
import { CitationsTab } from './CitationsTab'
import { ReportTab } from './ReportTab'
import type { ResearchPanelTab } from '../types'

const TABS_REQUIRING_STREAM: ResearchPanelTab[] = ['tasks', 'thinking', 'citations']

const formatElapsed = (timestamp?: Date): string => {
  if (!timestamp) return ''
  const elapsedSeconds = Math.max(0, Math.floor((Date.now() - timestamp.getTime()) / 1000))
  if (elapsedSeconds < 5) return 'just now'
  if (elapsedSeconds < 60) return `${elapsedSeconds}s ago`
  const minutes = Math.floor(elapsedSeconds / 60)
  return `${minutes}m ago`
}

interface ResearchPanelProps {
  /** Content to display in the panel */
  children?: ReactNode
  /** Whether the user is authenticated */
  isAuthenticated?: boolean
  /** Whether to show the protruding toggle when the panel is closed */
  showToggle?: boolean
}

/**
 * Research panel with tabbed content (Thinking, Citations, Report).
 * Opens from the right side of the screen, pushing the chat area.
 * Takes 60% of the screen width when open.
 */
export const ResearchPanel: FC<ResearchPanelProps> = memo(function ResearchPanel({
  children,
  isAuthenticated = false,
  showToggle = true,
}) {
  const isOpen = useLayoutStore((s) => s.rightPanel === 'research')
  const researchPanelTab = useLayoutStore((s) => s.researchPanelTab)
  const setResearchPanelTab = useLayoutStore((s) => s.setResearchPanelTab)
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const isDeepResearchStreaming = useChatStore((state) => state.isDeepResearchStreaming)
  const deepResearchJobId = useChatStore((state) => state.deepResearchJobId)
  const deepResearchStreamLoaded = useChatStore((state) => state.deepResearchStreamLoaded)
  const deepResearchStatus = useChatStore((state) => state.deepResearchStatus)
  const deepResearchEngine = useChatStore((state) => state.deepResearchEngine)
  const deepResearchActivity = useChatStore((state) => state.deepResearchActivity)
  const deepResearchAgents = useChatStore((state) => state.deepResearchAgents)
  const deepResearchToolCalls = useChatStore((state) => state.deepResearchToolCalls)
  const deepResearchFiles = useChatStore((state) => state.deepResearchFiles)
  const { importStreamOnly, isLoading: isStreamLoading } = useLoadJobData()
  const { cancelDeepResearchJob, isCancelling } = useCancelDeepResearchJob()

  const prefersReducedMotion = useReducedMotion()
  const [isMobileViewport, setIsMobileViewport] = useState(false)
  const [, forceActivityTick] = useState(0)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 767px)')
    const handleChange = () => setIsMobileViewport(mediaQuery.matches)
    handleChange()
    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  useEffect(() => {
    if (!isDeepResearchStreaming) return
    const timer = window.setInterval(() => forceActivityTick((value) => value + 1), 5000)
    return () => window.clearInterval(timer)
  }, [isDeepResearchStreaming])

  const activityStats = useMemo(() => {
    const runningAgents = deepResearchAgents.filter((agent) => agent.status === 'running').length
    const completedAgents = deepResearchAgents.filter((agent) => agent.status === 'complete').length
    const runningTools = deepResearchToolCalls.filter((tool) => tool.status === 'running').length
    const completedTools = deepResearchToolCalls.filter((tool) => tool.status === 'complete').length
    return { runningAgents, completedAgents, runningTools, completedTools, files: deepResearchFiles.length }
  }, [deepResearchAgents, deepResearchToolCalls, deepResearchFiles])
  const researchEngineLabel = deepResearchEngine === 'claude_code' ? 'Claude Code' : 'AIQ'

  const handleClose = useCallback(() => {
    closeRightPanel()
  }, [closeRightPanel])

  const handleStopResearch = useCallback(async () => {
    await cancelDeepResearchJob(deepResearchJobId)
  }, [cancelDeepResearchJob, deepResearchJobId])

  const handleToggle = useCallback(() => {
    if (!isAuthenticated) return

    if (isOpen) {
      closeRightPanel()
    } else {
      openRightPanel('research')

      // Trigger stream import when opening panel if current tab requires it and data not loaded
      if (
        TABS_REQUIRING_STREAM.includes(researchPanelTab) &&
        deepResearchJobId &&
        !deepResearchStreamLoaded &&
        !isDeepResearchStreaming &&
        !isStreamLoading
      ) {
        void importStreamOnly(deepResearchJobId)
      }
    }
  }, [isAuthenticated, isOpen, closeRightPanel, openRightPanel, researchPanelTab, deepResearchJobId, deepResearchStreamLoaded, isDeepResearchStreaming, isStreamLoading, importStreamOnly])

  const handleTabChange = useCallback(
    (value: string) => {
      const tab = value as ResearchPanelTab
      setResearchPanelTab(tab)

      // Trigger stream import when clicking Tasks/Thinking/Citations for a loaded (non-streaming) job
      if (
        TABS_REQUIRING_STREAM.includes(tab) &&
        deepResearchJobId &&
        !deepResearchStreamLoaded &&
        !isDeepResearchStreaming &&
        !isStreamLoading
      ) {
        // Fire and forget - don't block the tab change
        void importStreamOnly(deepResearchJobId)
      }
    },
    [setResearchPanelTab, deepResearchJobId, deepResearchStreamLoaded, isDeepResearchStreaming, isStreamLoading, importStreamOnly]
  )

  if (!showToggle && !isOpen) {
    return null
  }

  if (isMobileViewport && !isOpen) {
    return null
  }

  return (
    // Wrapper: uses flex to keep button visible while panel animates
    <div
      className={
        isMobileViewport
          ? isOpen
            ? 'fixed left-0 right-0 z-40 flex'
            : 'hidden'
          : 'relative h-full flex'
      }
      style={{
        top: isMobileViewport && isOpen ? 'var(--header-height)' : undefined,
        height: isMobileViewport && isOpen ? 'calc(100dvh - var(--header-height) - var(--mobile-nav-height))' : undefined,
        width: isMobileViewport ? '100vw' : isOpen ? 'calc(60% + 40px)' : '40px',
        minWidth: isMobileViewport ? '100vw' : isOpen ? 'calc(60% + 40px)' : '40px',
        transition: prefersReducedMotion
          ? 'none'
          : 'width 600ms ease-in-out, min-width 600ms ease-in-out',
      }}
    >
      {/* Toggle Tag Button - protruding from left side, always visible */}
      <button
        type="button"
        onClick={handleToggle}
        disabled={!isAuthenticated}
        className={`research-panel-toggle border-base bg-surface-base relative z-10 w-10 shrink-0 items-center justify-center self-start overflow-hidden mt-[calc(var(--spacing)*3)] rounded-l-lg border-b border-l border-r border-t transition-colors ${
          isMobileViewport ? 'hidden' : 'flex'
        } ${
          !showToggle && !isOpen ? 'hidden' : ''
        } ${
          isAuthenticated ? 'cursor-pointer hover:border-[#5AA7FF]' : 'cursor-not-allowed opacity-50'
        }`}
        style={{ height: 'calc(var(--spacing) * 38)' }}
        aria-label={isOpen ? 'Close research panel' : 'Open research panel'}
        aria-expanded={isOpen}
        title={isAuthenticated ? (isOpen ? 'Close research panel' : 'Open research panel') : 'Sign in to access research panel'}
        data-testid="research-panel-toggle"
      >
        <span
          className="absolute left-1/2 -translate-x-1/2 flex items-center justify-center"
          style={{ top: 'calc(var(--spacing) * 3)', width: 'calc(var(--spacing) * 6)', height: 'calc(var(--spacing) * 6)' }}
        >
          {isDeepResearchStreaming ? (
            <Spinner size="small" aria-label="Researching" />
          ) : (
            <Generate className="h-[calc(var(--spacing)*6)] w-[calc(var(--spacing)*6)]" />
          )}
        </span>
        <Text
          kind="label/semibold/sm"
          className="absolute left-1/2 -translate-x-1/2 -rotate-90 whitespace-nowrap text-primary"
          style={{ top: 'calc(var(--spacing) * 21)' }}
        >
          Research
        </Text>
      </button>

      {/* Outer container: clips content, fills remaining space */}
      <div
        className={`border-base bg-surface-base h-full flex-1 overflow-hidden border-l border-t -ml-px ${
          isMobileViewport && isOpen ? 'rounded-none' : 'rounded-tl-xl'
        }`}
        aria-hidden={!isOpen}
      >
        {/* Inner container: fixed width so content stays stable */}
        <Flex
          direction="col"
          className="h-full w-full"
          style={{
            visibility: isOpen ? 'visible' : 'hidden',
            opacity: isOpen ? 1 : 0,
            transition: prefersReducedMotion
              ? 'none'
              : isOpen
                ? 'opacity 100ms ease-in-out, visibility 0ms'
                : 'opacity 100ms ease-in-out 500ms, visibility 0ms 600ms',
          }}
        >
        {/* Panel header: mono label + live status pill */}
        <Flex align="center" justify="between" className="border-base shrink-0 gap-2 border-b px-3 pt-3 sm:pl-6 sm:pr-8">
          <Flex align="center" gap="3" className="min-w-0 pb-3">
            <span className="gx-panel-label">Research</span>
            {isDeepResearchStreaming ? (
              <span className="gx-status-pill gx-status-pill--live">
                <span className="gx-status-dot" aria-hidden="true" />
                Researching&hellip;
              </span>
            ) : deepResearchStatus === 'success' ? (
              <span className="gx-status-pill">Complete</span>
            ) : deepResearchStatus === 'failure' ? (
              <span className="gx-status-pill">Failed</span>
            ) : null}
          </Flex>
          <Flex align="center" gap="density-xl" className="pb-3">
            {/* Close button */}
            <Button
              kind="tertiary"
              size="small"
              onClick={handleClose}
              aria-label="Close research panel"
              title="Close research panel"
              data-testid="research-panel-close"
            >
              <Close className="h-4 w-4" aria-hidden="true" />
            </Button>
          </Flex>
        </Flex>

        {/* Tabs and stop control */}
        <Flex align="center" justify="between" className="border-base shrink-0 flex-wrap gap-2 border-b px-3 py-3 sm:pl-6 sm:pr-8 sm:py-4">
          <Flex align="center" gap="density-xl" className="min-w-0 flex-1 overflow-x-auto scrollbar-hide">
            <SegmentedControl
              value={researchPanelTab}
              onValueChange={handleTabChange}
              size="medium"
              items={[
                { value: 'plan', children: 'Plan' },
                { value: 'batch', children: 'Batch' },
                { value: 'tasks', children: 'Activity' },
                { value: 'thinking', children: 'Thinking' },
                { value: 'citations', children: 'Sources' },
                { value: 'report', children: 'Report' },
              ]}
            />
            {/* Stop Researching button - always visible, disabled when not streaming */}
            <Button
              kind="tertiary"
              size="small"
              onClick={isDeepResearchStreaming ? handleStopResearch : undefined}
              disabled={!isDeepResearchStreaming || isCancelling}
              aria-label="Stop researching"
              title={isDeepResearchStreaming ? 'Cancel current research' : 'No active research'}
              data-testid="research-panel-stop"
            >
              <StopCircle className="h-4 w-4 sm:mr-2" aria-hidden="true" />
              <span className="hidden sm:inline">{isCancelling ? 'Cancelling...' : 'Cancel Research'}</span>
            </Button>
          </Flex>
        </Flex>

        {(isDeepResearchStreaming || deepResearchActivity) && (
          <div className="border-base shrink-0 border-b bg-surface-raised px-3 py-3 sm:px-6">
            <Flex align="center" justify="between" gap="3" className="min-w-0">
              <Flex align="center" gap="3" className="min-w-0 flex-1">
                {isDeepResearchStreaming ? (
                  <Spinner size="small" aria-label="Research activity" />
                ) : (
                  <Generate className="h-4 w-4 shrink-0 text-subtle" aria-hidden="true" />
                )}
                <Flex direction="col" gap="0" className="min-w-0 flex-1">
                  <Text kind="label/semibold/sm" className="truncate text-primary">
                    {deepResearchActivity?.message ||
                      (deepResearchStatus === 'submitted'
                        ? `${researchEngineLabel} research job queued`
                        : `${researchEngineLabel} research activity`)}
                  </Text>
                  <Text kind="body/regular/xs" className="truncate text-subtle">
                    {deepResearchActivity?.detail || 'Waiting for the next research event'}
                  </Text>
                </Flex>
              </Flex>
              <Flex align="center" gap="2" className="hidden shrink-0 md:flex">
                <Text kind="body/regular/xs" className="text-tertiary">
                  {activityStats.runningAgents > 0
                    ? `${activityStats.runningAgents} agents running`
                    : `${activityStats.completedAgents} agents`}
                </Text>
                <Text kind="body/regular/xs" className="text-tertiary">
                  {activityStats.runningTools > 0
                    ? `${activityStats.runningTools} tools running`
                    : `${activityStats.completedTools} tools`}
                </Text>
                {activityStats.files > 0 && (
                  <Text kind="body/regular/xs" className="text-tertiary">
                    {activityStats.files} files
                  </Text>
                )}
                {deepResearchActivity?.timestamp && (
                  <Text kind="body/regular/xs" className="text-tertiary">
                    {formatElapsed(deepResearchActivity.timestamp)}
                  </Text>
                )}
              </Flex>
            </Flex>
          </div>
        )}

        {/* Content Area - each tab manages its own scrolling and footer */}
        <Flex direction="col" className="flex-1 overflow-hidden px-3 py-4 sm:py-5 sm:pl-6 sm:pr-8">
          {isStreamLoading ? (
            <Flex direction="col" align="center" justify="center" className="h-full gap-4">
              <Spinner size="medium" aria-label="Loading research data" />
              <Text kind="body/regular/md" className="text-tertiary">
                {TABS_REQUIRING_STREAM.includes(researchPanelTab)
                  ? 'Loading research data...'
                  : 'Loading report...'}
              </Text>
            </Flex>
          ) : (
            <>
              {researchPanelTab === 'plan' && <PlanTab />}
              {researchPanelTab === 'batch' && <BatchResearchQueue />}
              {researchPanelTab === 'tasks' && <TasksTab />}
              {researchPanelTab === 'thinking' && <ThinkingTab />}
              {researchPanelTab === 'citations' && <CitationsTab />}
              {researchPanelTab === 'report' && <ReportTab>{children}</ReportTab>}
            </>
          )}
        </Flex>
      </Flex>
      </div>
    </div>
  )
})

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * useBatchResearchQueue Hook
 *
 * Turns approved batch queue items into deep research jobs one at a time.
 * Queue entries are stored in the chat store so they survive refreshes and
 * can be approved in parallel before the runner advances them sequentially.
 */

'use client'

import { useEffect, useMemo, useRef } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { submitDeepResearchJob, type DeepResearchJobStatus } from '@/adapters/api'
import { useChatStore } from '../store'
import { useAuth } from '@/adapters/auth'

const COMPLETED_STATUSES: ReadonlySet<DeepResearchJobStatus> = new Set([
  'success',
  'failure',
  'interrupted',
])

const APPROVAL_TIMEOUT_BUFFER_MS = 250

export const useBatchResearchQueue = (): void => {
  const { isAuthenticated, isLoading: isAuthLoading, authRequired } = useAuth()

  const state = useChatStore(
    useShallow((s) => ({
      currentConversation: s.currentConversation,
      deepResearchJobId: s.deepResearchJobId,
      isDeepResearchStreaming: s.isDeepResearchStreaming,
      deepResearchStatus: s.deepResearchStatus,
      batchResearchQueue: s.batchResearchQueue,
    }))
  )

  const addUserMessage = useChatStore((s) => s.addUserMessage)
  const clearReportContent = useChatStore((s) => s.clearReportContent)
  const clearPendingInteraction = useChatStore((s) => s.clearPendingInteraction)
  const addDeepResearchBanner = useChatStore((s) => s.addDeepResearchBanner)
  const addAgentResponseWithMeta = useChatStore((s) => s.addAgentResponseWithMeta)
  const startDeepResearch = useChatStore((s) => s.startDeepResearch)
  const approveBatchResearchItem = useChatStore((s) => s.approveBatchResearchItem)
  const markBatchResearchItemRunning = useChatStore((s) => s.markBatchResearchItemRunning)
  const markBatchResearchItemComplete = useChatStore((s) => s.markBatchResearchItemComplete)
  const removeBatchResearchItem = useChatStore((s) => s.removeBatchResearchItem)
  const addErrorCard = useChatStore((s) => s.addErrorCard)
  const setLoading = useChatStore((s) => s.setLoading)

  const approvalTimersRef = useRef<Map<string, number>>(new Map())
  const activeSubmissionRef = useRef<string | null>(null)

  const currentConversationId = state.currentConversation?.id ?? null
  const currentConversation = state.currentConversation
  const queuedItems = useMemo(
    () =>
      state.batchResearchQueue
        .filter((item) => item.conversationId === currentConversationId)
        .slice()
        .sort((a, b) => {
          const aTime = new Date(a.createdAt).getTime()
          const bTime = new Date(b.createdAt).getTime()
          return aTime - bTime
        }),
    [state.batchResearchQueue, currentConversationId]
  )

  // Auto-approve queued items after their grace period expires.
  useEffect(() => {
    for (const [itemId, timer] of approvalTimersRef.current.entries()) {
      const stillPending = queuedItems.some((item) => item.id === itemId && item.status === 'pending')
      if (!stillPending) {
        clearTimeout(timer)
        approvalTimersRef.current.delete(itemId)
      }
    }

    const now = Date.now()
    for (const item of queuedItems) {
      if (item.status !== 'pending') continue
      if (approvalTimersRef.current.has(item.id)) continue

      const target = item.autoApproveAt ? new Date(item.autoApproveAt).getTime() : NaN
      const delay = Number.isFinite(target) ? Math.max(0, target - now) + APPROVAL_TIMEOUT_BUFFER_MS : 0

      const timer = window.setTimeout(() => {
        approveBatchResearchItem(item.id)
        approvalTimersRef.current.delete(item.id)
      }, delay)

      approvalTimersRef.current.set(item.id, timer)
    }

  }, [approveBatchResearchItem, queuedItems])

  useEffect(() => {
    return () => {
      for (const timer of approvalTimersRef.current.values()) {
        clearTimeout(timer)
      }
      approvalTimersRef.current.clear()
    }
  }, [])

  // If an active job finished, finalize the running queue item and let the
  // effect below advance to the next approved task.
  useEffect(() => {
    const runningItem = queuedItems.find((item) => item.status === 'running' && item.jobId === state.deepResearchJobId)
    if (!runningItem) return
    if (state.isDeepResearchStreaming) return
    if (!state.deepResearchStatus || !COMPLETED_STATUSES.has(state.deepResearchStatus)) return

    markBatchResearchItemComplete(
      runningItem.id,
      state.deepResearchStatus === 'success'
        ? 'complete'
        : state.deepResearchStatus === 'interrupted'
          ? 'cancelled'
          : 'failed'
    )
    activeSubmissionRef.current = null
  }, [
    queuedItems,
    state.deepResearchJobId,
    state.deepResearchStatus,
    state.isDeepResearchStreaming,
    markBatchResearchItemComplete,
  ])

  useEffect(() => {
    if (isAuthLoading) return
    if (authRequired && !isAuthenticated) return
    if (state.isDeepResearchStreaming) return
    if (activeSubmissionRef.current) return
    if (!currentConversationId) return

    const nextItem = queuedItems.find((item) => item.status === 'approved')
    if (!nextItem) return

    activeSubmissionRef.current = nextItem.id

    const submit = async (): Promise<void> => {
      const conversation = currentConversation
      if (!conversation || conversation.id !== nextItem.conversationId) {
        removeBatchResearchItem(nextItem.id)
        activeSubmissionRef.current = null
        return
      }

      try {
        clearReportContent()
        clearPendingInteraction()

        addUserMessage(nextItem.query, {
          enabledDataSources: nextItem.enabledDataSources,
          messageFiles: nextItem.messageFiles,
        })

        const submission = await submitDeepResearchJob({
          agent_type: nextItem.researchEngine === 'claude_code' ? 'claude_researcher' : 'deep_researcher',
          input: nextItem.query,
          data_sources: nextItem.enabledDataSources,
          research_depth: nextItem.researchDepth,
        })

        addDeepResearchBanner(
          'starting',
          submission.job_id,
          conversation.id,
          undefined,
          nextItem.researchEngine
        )

        const messageId = addAgentResponseWithMeta('', false, {
          deepResearchJobId: submission.job_id,
          deepResearchJobStatus: 'submitted',
          isDeepResearchActive: true,
          planMessages: [],
        })

        markBatchResearchItemRunning(nextItem.id, submission.job_id, messageId)
        startDeepResearch(submission.job_id, messageId, nextItem.researchEngine)
        setLoading(false)
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Failed to submit queued research'
        addErrorCard('agent.deep_research_failed', errorMessage)
        markBatchResearchItemComplete(nextItem.id, 'failed', errorMessage)
        activeSubmissionRef.current = null
        setLoading(false)
      }
    }

    void submit()
  }, [
    addAgentResponseWithMeta,
    addDeepResearchBanner,
    addErrorCard,
    addUserMessage,
    authRequired,
    clearPendingInteraction,
    clearReportContent,
    currentConversationId,
    isAuthLoading,
    isAuthenticated,
    markBatchResearchItemComplete,
    markBatchResearchItemRunning,
    removeBatchResearchItem,
    queuedItems,
    setLoading,
    startDeepResearch,
    state.isDeepResearchStreaming,
    currentConversation,
  ])
}

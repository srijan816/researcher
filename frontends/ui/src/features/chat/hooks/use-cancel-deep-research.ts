// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelJob } from '@/adapters/api'
import { useAuth } from '@/adapters/auth'
import { useChatStore } from '../store'

const CANCEL_FALLBACK_TIMEOUT_MS = 5000

const findJobMessage = (jobId: string) => {
  const state = useChatStore.getState()

  for (const conversation of state.conversations) {
    const message = conversation.messages.find(
      (candidate) =>
        candidate.deepResearchJobId === jobId ||
        candidate.deepResearchBannerData?.jobId === jobId
    )
    if (message) {
      return { conversationId: conversation.id, messageId: message.id }
    }
  }

  return {
    conversationId: state.deepResearchOwnerConversationId || state.currentConversation?.id,
    messageId: state.activeDeepResearchMessageId || undefined,
  }
}

const finishCancelledJobLocally = (jobId: string) => {
  const state = useChatStore.getState()
  const owner = findJobMessage(jobId)
  const hasReport = Boolean(state.reportContent?.trim())

  if (owner.conversationId && owner.messageId) {
    state.patchConversationMessage(owner.conversationId, owner.messageId, {
      content: '',
      deepResearchJobStatus: 'interrupted',
      isDeepResearchActive: false,
      showViewReport: hasReport,
    })
  }

  state.stopAllDeepResearchSpinners()
  state.addDeepResearchBanner('cancelled', jobId, owner.conversationId || undefined)
  state.updateDeepResearchStatus('interrupted')
  state.setStreamLoaded(true)
  state.completeDeepResearch()
  state.setStreaming(false)
}

export const useCancelDeepResearchJob = () => {
  const { idToken } = useAuth()
  const fallbackRef = useRef<NodeJS.Timeout | null>(null)
  const [isCancelling, setIsCancelling] = useState(false)

  useEffect(() => {
    return () => {
      if (fallbackRef.current) {
        clearTimeout(fallbackRef.current)
        fallbackRef.current = null
      }
    }
  }, [])

  const cancelDeepResearchJob = useCallback(
    async (jobId?: string | null) => {
      const state = useChatStore.getState()
      const targetJobId = jobId || state.deepResearchJobId
      if (!targetJobId || isCancelling) return

      setIsCancelling(true)
      try {
        await cancelJob(targetJobId, idToken || undefined)

        if (fallbackRef.current) clearTimeout(fallbackRef.current)
        fallbackRef.current = setTimeout(() => {
          fallbackRef.current = null
          const latestState = useChatStore.getState()
          const matchingActiveJob =
            latestState.deepResearchJobId === targetJobId &&
            latestState.isDeepResearchStreaming
          const matchingPersistedJob = latestState.conversations.some((conversation) =>
            conversation.messages.some(
              (message) =>
                message.deepResearchJobId === targetJobId &&
                (message.deepResearchJobStatus === 'submitted' ||
                  message.deepResearchJobStatus === 'running' ||
                  message.isDeepResearchActive)
            )
          )

          if (matchingActiveJob || matchingPersistedJob) {
            finishCancelledJobLocally(targetJobId)
          }
        }, CANCEL_FALLBACK_TIMEOUT_MS)
      } catch (error) {
        console.error('Failed to cancel job:', error)
      } finally {
        setIsCancelling(false)
      }
    },
    [idToken, isCancelling]
  )

  return { cancelDeepResearchJob, isCancelling }
}

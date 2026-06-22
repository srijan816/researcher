// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * BatchResearchQueue Component
 *
 * Shows queued deep research tasks for the current conversation.
 * Items can be approved individually and then run sequentially by the
 * queue runner hook. Pending items auto-approve after their timeout.
 */

'use client'

import { type FC, memo, useEffect, useMemo, useState } from 'react'
import { Button, Flex, Text } from '@/adapters/ui'
import { Check, Clock, LoadingSpinner, Trash } from '@/adapters/ui/icons'
import { useChatStore } from '@/features/chat'
import { useLayoutStore } from '../store'

const formatCountdown = (target?: string): string => {
  if (!target) return 'auto-start soon'
  const targetTime = new Date(target).getTime()
  if (Number.isNaN(targetTime)) return 'auto-start soon'
  const remaining = Math.max(0, targetTime - Date.now())
  if (remaining === 0) return 'ready'
  const totalSeconds = Math.ceil(remaining / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes === 0) return `${seconds}s`
  return `${minutes}m ${seconds.toString().padStart(2, '0')}s`
}

const statusLabel = (status: string): string => {
  switch (status) {
    case 'pending':
      return 'Waiting for approval'
    case 'approved':
      return 'Approved'
    case 'running':
      return 'Running'
    case 'complete':
      return 'Completed'
    case 'failed':
      return 'Failed'
    case 'cancelled':
      return 'Cancelled'
    default:
      return status
  }
}

interface BatchResearchQueueProps {
  compact?: boolean
}

export const BatchResearchQueue: FC<BatchResearchQueueProps> = memo(function BatchResearchQueue({
  compact = false,
}) {
  const currentConversationId = useChatStore((state) => state.currentConversation?.id ?? null)
  const queue = useChatStore((state) =>
    state.batchResearchQueue.filter((item) => item.conversationId === currentConversationId)
  )
  const approveBatchResearchItem = useChatStore((state) => state.approveBatchResearchItem)
  const removeBatchResearchItem = useChatStore((state) => state.removeBatchResearchItem)
  const openRightPanel = useLayoutStore((state) => state.openRightPanel)
  const setResearchPanelTab = useLayoutStore((state) => state.setResearchPanelTab)

  const [, setNowTick] = useState(Date.now())
  useEffect(() => {
    if (queue.length === 0) return
    const timer = window.setInterval(() => setNowTick(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [queue.length])

  const stats = useMemo(() => {
    const pending = queue.filter((item) => item.status === 'pending').length
    const approved = queue.filter((item) => item.status === 'approved').length
    const running = queue.filter((item) => item.status === 'running').length
    const finished = queue.filter((item) => item.status === 'complete').length
    return { pending, approved, running, finished }
  }, [queue])

  if (!currentConversationId || queue.length === 0) {
    return compact ? null : (
      <Flex
        direction="col"
        align="center"
        justify="center"
        className="rounded-lg border border-dashed border-base bg-surface-raised px-4 py-6 text-center"
      >
        <Clock className="text-subtle mb-2 h-6 w-6" />
        <Text kind="label/semibold/sm" className="text-primary">
          No queued tasks yet
        </Text>
        <Text kind="body/regular/xs" className="text-subtle mt-1 max-w-xs">
          Add deep research tasks to queue them here and approve them before they auto-start.
        </Text>
      </Flex>
    )
  }

  return (
    <Flex direction="col" gap="3" className={compact ? 'w-full' : 'h-full min-h-0'}>
      {!compact && (
        <Flex align="center" justify="between" className="shrink-0">
          <Flex direction="col" gap="1">
            <Text kind="label/semibold/md" className="text-subtle">
              Batch Queue
            </Text>
            <Text kind="body/regular/xs" className="text-subtle">
              Approve tasks individually. Unapproved items auto-start after a short delay.
            </Text>
          </Flex>
          <Button
            kind="tertiary"
            size="small"
            onClick={() => {
              setResearchPanelTab('batch')
              openRightPanel('research')
            }}
            aria-label="Open batch queue"
          >
            Open Queue
          </Button>
        </Flex>
      )}

      <Flex direction="col" gap="2" className="min-h-0 overflow-y-auto">
        <Flex align="center" gap="2" className="shrink-0 flex-wrap">
          <Text kind="label/semibold/xs" className="text-subtle uppercase">
            Queue status
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            {queue.length} tasks
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            {stats.pending} pending
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            {stats.approved} approved
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            {stats.running} running
          </Text>
          <Text kind="body/regular/xs" className="text-subtle">
            {stats.finished} done
          </Text>
        </Flex>

        {queue.map((item) => {
          const countdown = item.status === 'pending' ? formatCountdown(item.autoApproveAt) : null
          const isRunning = item.status === 'running'

          return (
            <Flex
              key={item.id}
              direction="col"
              gap="2"
              className="rounded-lg border border-base bg-surface-sunken p-3"
            >
              <Flex align="start" justify="between" gap="3" className="min-w-0">
                <Flex direction="col" gap="1" className="min-w-0 flex-1">
                  <Text kind="label/semibold/sm" className="truncate text-primary">
                    {item.title}
                  </Text>
                  <Text kind="body/regular/xs" className="line-clamp-2 text-subtle">
                    {item.query}
                  </Text>
                </Flex>
                <Flex align="center" gap="2" className="shrink-0">
                  {item.status === 'pending' && <Clock className="h-4 w-4 text-subtle" />}
                  {item.status === 'approved' && <Check className="h-4 w-4 text-emerald-500" />}
                  {isRunning && <LoadingSpinner size="small" aria-label="Running batch task" />}
                </Flex>
              </Flex>

              <Flex align="center" gap="2" className="flex-wrap">
                <span className="rounded-md border border-base bg-surface-base px-2 py-1 text-[11px] uppercase text-subtle">
                  {item.researchDepth}
                </span>
                <span className="rounded-md border border-base bg-surface-base px-2 py-1 text-[11px] uppercase text-subtle">
                  {item.researchEngine === 'claude_code' ? 'Claude Code' : 'AIQ'}
                </span>
                <span className="rounded-md border border-base bg-surface-base px-2 py-1 text-[11px] uppercase text-subtle">
                  {statusLabel(item.status)}
                </span>
                {countdown && (
                  <span className="rounded-md border border-base bg-surface-base px-2 py-1 text-[11px] text-subtle">
                    Auto-start in {countdown === 'ready' ? 'now' : countdown}
                  </span>
                )}
              </Flex>

              <Flex align="center" justify="between" gap="2" className="flex-wrap">
                <Text kind="body/regular/xs" className="text-tertiary">
                  {item.enabledDataSources.length} sources • {item.messageFiles.length} files
                </Text>
                <Flex align="center" gap="2" className="flex-wrap">
                  {item.status === 'pending' && (
                    <Button
                      kind="secondary"
                      size="small"
                      onClick={() => approveBatchResearchItem(item.id)}
                    >
                      <Check className="h-4 w-4 sm:mr-2" />
                      <span className="hidden sm:inline">Approve</span>
                    </Button>
                  )}
                  <Button
                    kind="tertiary"
                    size="small"
                    color="danger"
                    onClick={() => removeBatchResearchItem(item.id)}
                    aria-label={`Remove ${item.title}`}
                  >
                    <Trash className="h-4 w-4" />
                  </Button>
                </Flex>
              </Flex>
            </Flex>
          )
        })}
      </Flex>
    </Flex>
  )
})

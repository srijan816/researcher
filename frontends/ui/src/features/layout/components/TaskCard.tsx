// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TaskCard Component
 *
 * Timeline entry for a single task/todo item with a progress dot,
 * one-line summary, and a status badge.
 *
 * SSE Events: artifact.update with type: "todo"
 */

'use client'

import { type FC } from 'react'
import { Flex, Text, Badge } from '@/adapters/ui'
import { LoadingSpinner } from '@/adapters/ui/icons'
import type { DeepResearchTodo, DeepResearchTodoStatus } from '@/features/chat/types'

interface TaskCardProps {
  /** Todo item from deep research */
  todo: DeepResearchTodo
}

/**
 * Get badge color based on task status
 * - green: completed
 * - teal: in_progress
 * - yellow: pending
 * - red: stopped (error state)
 */
const getBadgeColor = (status: DeepResearchTodoStatus): 'green' | 'teal' | 'yellow' | 'red' => {
  switch (status) {
    case 'completed':
      return 'green'
    case 'in_progress':
      return 'teal'
    case 'pending':
      return 'yellow'
    case 'stopped':
      return 'red'
    default:
      return 'yellow'
  }
}

/**
 * Get display text for status badge
 */
const getStatusText = (status: DeepResearchTodoStatus): string => {
  switch (status) {
    case 'completed':
      return 'complete'
    case 'in_progress':
      return 'in progress'
    case 'pending':
      return 'pending'
    case 'stopped':
      return 'stopped'
    default:
      return status
  }
}

/**
 * Timeline row showing a single task: status dot on a vertical line,
 * one-line summary, and a status badge (spinner while in progress).
 */
export const TaskCard: FC<TaskCardProps> = ({ todo }) => {
  const isComplete = todo.status === 'completed'
  const isActive = todo.status === 'in_progress'
  const badgeColor = getBadgeColor(todo.status)
  const statusText = getStatusText(todo.status)

  return (
    <Flex align="start" gap="3" className="py-1.5 pl-0">
      {/* Timeline dot: filled when done, pulsing when active, faint when pending */}
      <span
        className={`gx-timeline-dot ${
          isComplete ? 'gx-timeline-dot--done' : isActive ? 'gx-timeline-dot--active' : ''
        }`}
        aria-hidden="true"
      />

      {/* Task Name */}
      <Text
        kind="body/regular/md"
        className={`flex-1 min-w-0 ${isComplete ? 'text-subtle' : isActive ? 'text-primary' : 'text-secondary opacity-70'}`}
      >
        {todo.content}
      </Text>

      {/* Status Badge - with spinner for in_progress */}
      <Badge color={badgeColor}>
        <Flex align="center" gap="1">
          {todo.status === 'in_progress' && (
            <LoadingSpinner size="small" className="h-3 w-3" aria-label="In progress" />
          )}
          {statusText}
        </Flex>
      </Badge>
    </Flex>
  )
}

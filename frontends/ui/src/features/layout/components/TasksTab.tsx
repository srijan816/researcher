// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TasksTab Component
 *
 * Tab within ResearchPanel showing task/todo items from DEEP RESEARCH only.
 * Displays the running todo list from artifact.update events with type: "todo".
 *
 * SSE Events: artifact.update with type: "todo"
 */

'use client'

import { type FC } from 'react'
import { Flex, Text, ProgressBar } from '@/adapters/ui'
import { useShallow } from 'zustand/react/shallow'
import { CheckCircle } from '@/adapters/ui/icons'
import { useChatStore } from '@/features/chat'
import { TaskCard } from './TaskCard'
import type { DeepResearchTodoGroup } from '@/features/chat/types'

const hasVisibleTodos = (group: DeepResearchTodoGroup): boolean => group.todos.length > 0

/**
 * Tasks tab content showing todos/tasks from deep research.
 * Uses deepResearchTodos from the store (populated by SSE artifact.update events).
 */
export const TasksTab: FC = () => {
  const state =
    useChatStore(useShallow((s) => ({
      deepResearchTodos: s.deepResearchTodos,
      deepResearchTodoGroups: s.deepResearchTodoGroups,
      deepResearchJobId: s.deepResearchJobId,
      currentStatus: s.currentStatus,
      isDeepResearchStreaming: s.isDeepResearchStreaming,
      deepResearchAgents: s.deepResearchAgents,
      deepResearchToolCalls: s.deepResearchToolCalls,
      deepResearchFiles: s.deepResearchFiles,
      deepResearchActivity: s.deepResearchActivity,
    })))

  const deepResearchTodos = Array.isArray(state.deepResearchTodos) ? state.deepResearchTodos : []
  const deepResearchTodoGroups = Array.isArray(state.deepResearchTodoGroups) ? state.deepResearchTodoGroups : []
  const visibleTodoGroups = deepResearchTodoGroups.filter(hasVisibleTodos)
  const deepResearchAgents = Array.isArray(state.deepResearchAgents) ? state.deepResearchAgents : []
  const deepResearchToolCalls = Array.isArray(state.deepResearchToolCalls) ? state.deepResearchToolCalls : []
  const deepResearchFiles = Array.isArray(state.deepResearchFiles) ? state.deepResearchFiles : []
  const { deepResearchJobId, currentStatus, isDeepResearchStreaming, deepResearchActivity } = state

  const isEmpty = deepResearchTodos.length === 0 && visibleTodoGroups.length === 0

  // Calculate progress stats
  const allVisibleTodos = [...deepResearchTodos, ...visibleTodoGroups.flatMap((group) => group.todos)]
  const completedCount = allVisibleTodos.filter((t) => t.status === 'completed').length
  const totalCount = allVisibleTodos.length
  const rootCompletedCount = deepResearchTodos.filter((t) => t.status === 'completed').length
  const rootTotalCount = deepResearchTodos.length
  const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0
  const isWritingReport = isDeepResearchStreaming && currentStatus === 'writing'
  const runningAgents = deepResearchAgents.filter((agent) => agent.status === 'running').length
  const completedAgents = deepResearchAgents.filter((agent) => agent.status === 'complete').length
  const runningTools = deepResearchToolCalls.filter((tool) => tool.status === 'running').length
  const completedTools = deepResearchToolCalls.filter((tool) => tool.status === 'complete').length
  const latestFile = deepResearchFiles[deepResearchFiles.length - 1]
  const showLiveProgress =
    isDeepResearchStreaming &&
    (deepResearchAgents.length > 0 || deepResearchToolCalls.length > 0 || deepResearchFiles.length > 0)

  return (
    <Flex direction="col" gap="4" className="h-full min-h-0">
      {/* Header with progress indicator */}
      <Flex direction="col" gap="1" className="shrink-0">
        <Flex align="center" gap="2">
          <Text kind="label/semibold/md" className="text-subtle">
            Activity
          </Text>
          {deepResearchJobId && (
            <Text kind="body/regular/xs" className="text-tertiary">
              JobID: {deepResearchJobId}
            </Text>
          )}
          {totalCount > 0 && (
            <Text kind="body/regular/xs" className="text-subtle">
                  {completedCount}/{totalCount}
                </Text>
              )}
        </Flex>
        <Text kind="body/regular/xs" className="text-subtle">
          Research plan breakdown and progress during deep research.
        </Text>
      </Flex>

      {/* Content */}
      {isEmpty ? (
        <Flex
          direction="col"
          align="center"
          justify="center"
          className="flex-1 text-center py-8"
        >
          <CheckCircle className="text-subtle mb-3 h-8 w-8" />
          <Text kind="body/regular/md" className="text-subtle">
            Research activity will appear here.
          </Text>
          <Text kind="body/regular/sm" className="text-subtle mt-2">
            Shows the plan breakdown and progress during deep research.
          </Text>
        </Flex>
      ) : (
        <Flex direction="col" gap="3" className="flex-1 min-h-0 overflow-y-auto">
          {/* Progress bar showing completion percentage */}
          <div className="shrink-0">
            <ProgressBar value={progressPercent} aria-label="Task completion progress" />
          </div>

          {/* Writing report indicator */}
          {isWritingReport && (
            <Flex align="center" gap="2" className="shrink-0 rounded-md bg-blue-50 px-3 py-2 dark:bg-blue-950">
              <div className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
              <Text kind="body/regular/sm" className="text-blue-700 dark:text-blue-300">
                Writing final report... This may take a few minutes.
              </Text>
            </Flex>
          )}

          {showLiveProgress && (
            <Flex direction="col" gap="1" className="shrink-0 rounded-md border border-base bg-surface-raised px-3 py-2">
              <Flex align="center" gap="2">
                <div className="h-2 w-2 animate-pulse rounded-full bg-[var(--accent-info)]" />
                <Text kind="body/semibold/sm" className="text-primary">
                  Live research lanes
                </Text>
                <Text kind="body/regular/xs" className="text-subtle">
                  {runningAgents > 0 ? `${runningAgents} running` : `${completedAgents} complete`}
                  {deepResearchToolCalls.length > 0 &&
                    ` • ${completedTools}/${deepResearchToolCalls.length} tools`}
                  {runningTools > 0 && ` • ${runningTools} active`}
                </Text>
              </Flex>
              {deepResearchActivity && (
                <Text kind="body/regular/xs" className="text-subtle line-clamp-2">
                  {deepResearchActivity.message}
                  {deepResearchActivity.detail ? `: ${deepResearchActivity.detail}` : ''}
                </Text>
              )}
              {latestFile && (
                <Text kind="body/regular/xs" className="text-tertiary">
                  Latest artifact: {latestFile.filename}
                </Text>
              )}
            </Flex>
          )}

          {rootTotalCount > 0 && visibleTodoGroups.length > 0 && (
            <Text kind="body/regular/xs" className="text-tertiary">
              Orchestrator plan: {rootCompletedCount}/{rootTotalCount}
            </Text>
          )}

          {deepResearchTodos.length > 0 && (
            <div className="gx-timeline flex flex-col">
              {deepResearchTodos.map((todo) => (
                <div key={todo.id} className="shrink-0">
                  <TaskCard todo={todo} />
                </div>
              ))}
            </div>
          )}

          {visibleTodoGroups.map((group) => {
            const groupCompletedCount = group.todos.filter((todo) => todo.status === 'completed').length
            const groupActiveTodo = group.todos.find((todo) => todo.status === 'in_progress')

            return (
              <Flex key={group.id} direction="col" gap="2" className="shrink-0">
                <Flex align="center" gap="2" className="min-w-0">
                  <div
                    className={`h-2 w-2 shrink-0 rounded-full ${
                      groupActiveTodo ? 'animate-pulse bg-[var(--accent-info)]' : 'bg-[var(--border-color-base)]'
                    }`}
                  />
                  <Text kind="body/semibold/sm" className="min-w-0 truncate text-primary">
                    {group.label}
                  </Text>
                  <Text kind="body/regular/xs" className="shrink-0 text-tertiary">
                    {groupCompletedCount}/{group.todos.length}
                  </Text>
                </Flex>
                <div className="gx-timeline ml-1 flex flex-col border-l border-base pl-3">
                  {group.todos.map((todo) => (
                    <div key={todo.id} className="shrink-0">
                      <TaskCard todo={todo} />
                    </div>
                  ))}
                </div>
              </Flex>
            )
          })}
        </Flex>
      )}
    </Flex>
  )
}

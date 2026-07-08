// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * useDeepResearch Hook
 *
 * Manages the SSE connection lifecycle for deep research jobs.
 * Automatically connects when a job ID is set in the store,
 * routes events to appropriate UI components, and handles reconnection.
 * Includes timeout detection for hung jobs.
 */

'use client'

import { useEffect, useRef, useCallback, useState } from 'react'
import { useShallow } from 'zustand/react/shallow'
import {
  createDeepResearchClient,
  cancelJob,
  getJobStatus,
  resumeJob,
  type DeepResearchClient,
  type DeepResearchJobStatus,
  type TodoItem,
} from '@/adapters/api'
import { buildDeepResearchTodoGroup, mapDeepResearchTodos, useChatStore } from '../store'
import { useAuth } from '@/adapters/auth'
import { useLayoutStore } from '@/features/layout/store'
import { checkBackendHealthCached } from '@/shared/hooks/use-backend-health'
import { isLikelyAuthRelatedTransportError, isDeepResearchReplayCompleteMode } from '../lib/transport-auth-signals'

/** Timeout in milliseconds before showing a warning (60 seconds) */
const TIMEOUT_WARNING_MS = 60000
/** How often to check for timeouts (10 seconds) */
const TIMEOUT_CHECK_INTERVAL_MS = 10000
/** Fallback timeout after cancel POST succeeds: if SSE doesn't deliver
 *  job.status "interrupted" within this window, clean up locally so the
 *  UI never stays stuck in a streaming state. */
const CANCEL_FALLBACK_TIMEOUT_MS = 5000
const USER_CANCELLED_ERROR_MARKER = 'cancelled by user'

const isUserCancelledStatus = (
  status: DeepResearchJobStatus,
  error?: string
): boolean => (
  status === 'interrupted' &&
  error?.toLowerCase().includes(USER_CANCELLED_ERROR_MARKER) === true
)
const isRecoverableInterruptedStatus = (
  status: DeepResearchJobStatus,
  error?: string
): boolean => {
  if (status !== 'interrupted' || isUserCancelledStatus(status, error)) return false
  const normalized = error?.toLowerCase() ?? ''
  return (
    normalized.includes('resume') ||
    normalized.includes('heartbeat') ||
    normalized.includes('backend restarted') ||
    normalized.includes('worker') ||
    normalized.includes('ghost')
  )
}
const TRANSPORT_RECONNECT_DELAY_MS = 3000
const TRANSPORT_RECONNECT_MAX_DELAY_MS = 30000
const AUTO_RESUME_MAX_ATTEMPTS = 2
const CLAUDE_ARTIFACT_PREVIEW_CHARS = 1600

const parseEventDate = (timestamp?: string): Date => {
  if (!timestamp) return new Date()
  const parsed = new Date(timestamp)
  return Number.isNaN(parsed.getTime()) ? new Date() : parsed
}

const isClaudeResearchFile = (filename: string): boolean => filename.startsWith('/claude_research/')

const basename = (path: string): string => path.split('/').filter(Boolean).pop() || path

const buildClaudeArtifactPreview = (filename: string, content: string): string => {
  const trimmed = content.trim()
  const preview =
    trimmed.length > CLAUDE_ARTIFACT_PREVIEW_CHARS
      ? `${trimmed.slice(0, CLAUDE_ARTIFACT_PREVIEW_CHARS).trimEnd()}\n\n[Preview truncated. Open Files for the full artifact.]`
      : trimmed
  return `File: ${filename}\n\n${preview || '[empty file]'}`
}

type ActiveToolIds = {
  thinkingStepId?: string
  toolCallId?: string
}

const getToolInputPreview = (input?: Record<string, unknown>): string | undefined => {
  if (!input) return undefined

  const query = input.query || input.q || input.search_query
  if (typeof query === 'string' && query.trim()) return query.trim()

  const url = input.url || input.source_url
  if (typeof url === 'string' && url.trim()) return url.trim()

  const filePath = input.file_path || input.path
  if (typeof filePath === 'string' && filePath.trim()) return filePath.trim()

  if ('_raw' in input && typeof input._raw === 'string') return input._raw.slice(0, 240)
  return JSON.stringify(input).slice(0, 240)
}

const getToolActivity = (
  name: string,
  input?: Record<string, unknown>
): { kind: 'tool' | 'search' | 'file'; message: string; detail?: string } => {
  const detail = getToolInputPreview(input)
  if (name.includes('search')) return { kind: 'search', message: 'Searching the web', detail }
  if (name.includes('quote')) return { kind: 'search', message: 'Checking current market data', detail }
  if (name.includes('read_file')) return { kind: 'file', message: 'Reading research notes', detail }
  if (name.includes('write_file')) return { kind: 'file', message: 'Writing research output', detail }
  if (name.includes('verified_sources')) return { kind: 'tool', message: 'Verifying source list', detail }
  if (name.includes('write_todos')) return { kind: 'tool', message: 'Updating research tasks', detail }
  return { kind: 'tool', message: `Running ${name}`, detail }
}

const getToolStatus = (
  name: string,
  workflow?: string
): 'searching' | 'planning' | 'researching' | 'writing' => {
  const normalizedName = name.toLowerCase()
  const normalizedWorkflow = (workflow || '').toLowerCase()
  if (normalizedName.includes('search') || normalizedName.includes('quote')) return 'searching'
  if (normalizedName.includes('write_file') || normalizedName.includes('report')) return 'writing'
  if (normalizedName.includes('write_todos')) {
    return normalizedWorkflow.includes('planner') ? 'planning' : 'researching'
  }
  return 'researching'
}

interface UseDeepResearchReturn {
  /** Whether deep research is currently streaming */
  isStreaming: boolean
  /** Current job ID */
  jobId: string | null
  /** Current job status */
  status: DeepResearchJobStatus | null
  /** Whether we're showing a timeout warning (no events received for too long) */
  isTimedOut: boolean
  /** Manually disconnect from the stream */
  disconnect: () => void
  /** Manually reconnect to the stream (uses last event ID) */
  reconnect: () => void
  /** Cancel the current job (useful for hung jobs) */
  cancelCurrentJob: () => Promise<void>
}

/**
 * Hook for managing deep research SSE streaming
 *
 * Automatically:
 * - Connects when deepResearchJobId is set in the store
 * - Routes SSE events to appropriate store actions
 * - Updates UI state (report, citations, thinking steps)
 * - Handles completion and errors
 */
export const useDeepResearch = (): UseDeepResearchReturn => {
  // Refs for SSE client lifecycle
  const clientRef = useRef<DeepResearchClient | null>(null)
  const connectRef = useRef<((jobId: string, bufferReplay?: boolean) => void) | null>(null)
  const lastEventTimeRef = useRef<number>(Date.now())
  const timeoutIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const cancelFallbackRef = useRef<NodeJS.Timeout | null>(null)
  const researchStartTimeRef = useRef<number | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptRef = useRef(0)
  const autoResumeAttemptsRef = useRef<Map<string, number>>(new Map())


  // State for timeout warning
  const [isTimedOut, setIsTimedOut] = useState(false)

  // Auth token for authenticated requests
  // Note: idToken is used for backend auth, not accessToken
  const { idToken, authRequired, error: authError } = useAuth()

  // Chat store — reactive state only
  const { deepResearchJobId, isDeepResearchStreaming, deepResearchStatus } =
    useChatStore(useShallow((s) => ({
      deepResearchJobId: s.deepResearchJobId,
      isDeepResearchStreaming: s.isDeepResearchStreaming,
      deepResearchStatus: s.deepResearchStatus,
    })))

  // Actions — stable references, won't trigger re-renders
  const updateDeepResearchStatus = useChatStore((s) => s.updateDeepResearchStatus)
  const completeDeepResearch = useChatStore((s) => s.completeDeepResearch)
  const addDeepResearchCitation = useChatStore((s) => s.addDeepResearchCitation)
  const setReportContent = useChatStore((s) => s.setReportContent)
  const addThinkingStep = useChatStore((s) => s.addThinkingStep)
  const appendToThinkingStep = useChatStore((s) => s.appendToThinkingStep)
  const completeThinkingStep = useChatStore((s) => s.completeThinkingStep)
  const setCurrentStatus = useChatStore((s) => s.setCurrentStatus)
  const setStreaming = useChatStore((s) => s.setStreaming)
  const setDeepResearchTodos = useChatStore((s) => s.setDeepResearchTodos)
  const setDeepResearchTodoGroup = useChatStore((s) => s.setDeepResearchTodoGroup)
  const stopAllDeepResearchSpinners = useChatStore((s) => s.stopAllDeepResearchSpinners)
  const addDeepResearchLLMStep = useChatStore((s) => s.addDeepResearchLLMStep)
  const appendToDeepResearchLLMStep = useChatStore((s) => s.appendToDeepResearchLLMStep)
  const completeDeepResearchLLMStep = useChatStore((s) => s.completeDeepResearchLLMStep)
  const addDeepResearchAgentWithId = useChatStore((s) => s.addDeepResearchAgentWithId)
  const completeDeepResearchAgent = useChatStore((s) => s.completeDeepResearchAgent)
  const addDeepResearchToolCall = useChatStore((s) => s.addDeepResearchToolCall)
  const completeDeepResearchToolCall = useChatStore((s) => s.completeDeepResearchToolCall)
  const addDeepResearchFile = useChatStore((s) => s.addDeepResearchFile)
  const patchConversationMessage = useChatStore((s) => s.patchConversationMessage)
  const addDeepResearchBanner = useChatStore((s) => s.addDeepResearchBanner)
  const setStreamLoaded = useChatStore((s) => s.setStreamLoaded)
  const setDeepResearchActivity = useChatStore((s) => s.setDeepResearchActivity)
  const setDeepResearchLastEventId = useChatStore((s) => s.setDeepResearchLastEventId)

  /**
   * Check if callbacks still belong to the active deep research stream.
   * The owner session may be in the background while the job keeps running.
   */
  const isOwnerActive = useCallback((callbackJobId?: string): boolean => {
    const state = useChatStore.getState()
    return Boolean(
      state.isDeepResearchStreaming &&
        state.deepResearchOwnerConversationId &&
        (!callbackJobId || state.deepResearchJobId === callbackJobId)
    )
  }, [])

  // Layout store for opening research panel
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const setResearchPanelTab = useLayoutStore((s) => s.setResearchPanelTab)

  // Ref to track active thinking step IDs by name
  const activeStepIdsRef = useRef<Map<string, string>>(new Map())
  const activeToolIdsByEventRef = useRef<Map<string, ActiveToolIds>>(new Map())
  const activeToolIdsByNameRef = useRef<Map<string, ActiveToolIds[]>>(new Map())

  /**
   * Reset the timeout tracker - called when we receive any live event.
   */
  const resetTimeout = useCallback(() => {
    lastEventTimeRef.current = Date.now()
    reconnectAttemptRef.current = 0
    setIsTimedOut(false)
  }, [])

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  const scheduleReconnect = useCallback((jobId: string, reason: string): void => {
    clearReconnectTimer()
    const attempt = reconnectAttemptRef.current + 1
    reconnectAttemptRef.current = attempt
    const delay = Math.min(
      TRANSPORT_RECONNECT_DELAY_MS * Math.max(1, attempt),
      TRANSPORT_RECONNECT_MAX_DELAY_MS
    )

    setIsTimedOut(true)
    setCurrentStatus('researching')
    setDeepResearchActivity({
      kind: 'status',
      message: 'Research is still running. Reconnecting stream...',
      detail: reason,
    })
    clientRef.current?.disconnect()
    clientRef.current = null

    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null
      const state = useChatStore.getState()
      if (
        state.deepResearchJobId === jobId &&
        state.isDeepResearchStreaming &&
        !clientRef.current?.isConnected()
      ) {
        connectRef.current?.(jobId, true)
      }
    }, delay)
  }, [clearReconnectTimer, setCurrentStatus, setDeepResearchActivity])

  const tryAutoResume = useCallback(async (jobId: string, error?: string): Promise<boolean> => {
    const attempts = autoResumeAttemptsRef.current.get(jobId) ?? 0
    if (attempts >= AUTO_RESUME_MAX_ATTEMPTS) return false

    autoResumeAttemptsRef.current.set(jobId, attempts + 1)
    setDeepResearchActivity({
      kind: 'status',
      message: 'Reconnecting to the research job...',
      detail: error,
    })

    try {
      const response = await resumeJob(jobId, idToken || undefined)
      updateDeepResearchStatus(response.status)
      if (response.status === 'success') {
        setCurrentStatus('complete')
        return true
      }
      if (response.status === 'submitted' || response.status === 'running') {
        setCurrentStatus('researching')
        scheduleReconnect(jobId, 'Auto-resumed recoverable interruption')
        return true
      }
    } catch (resumeError) {
      console.warn('Automatic deep research resume failed:', resumeError)
    }

    return false
  }, [idToken, scheduleReconnect, setCurrentStatus, setDeepResearchActivity, updateDeepResearchStatus])

  /**
   * Classify a deep research stream failure as auth-related or generic.
   * Used when the backend is healthy but the SSE stream errored,
   * which typically means the auth cookie or token drifted.
   */
  const getDeepResearchStreamFailure = useCallback(
    (message: string, details?: string): { code: string; message: string; details?: string } => {
      if (!authRequired) {
        return { code: 'connection.failed', message, details }
      }
      if (authError === 'RefreshAccessTokenError' || isLikelyAuthRelatedTransportError(message)) {
        return {
          code: 'auth.session_expired',
          message: 'Your session has expired. Please sign in again to continue.',
          details,
        }
      }
      return { code: 'connection.failed', message, details }
    },
    [authRequired, authError]
  )

  /**
   * Create and connect to the SSE stream
   */
  /**
   * Connect to the SSE stream from the beginning.
   *
   * Single connection, two internal phases:
   * 1. Buffer phase: all replayed events accumulate in plain JS objects (zero store writes).
   *    After 500ms of silence the buffer is flushed in one useChatStore.setState() call.
   * 2. Live phase: subsequent events go straight to individual store actions (fine for low volume).
   */
  const connect = useCallback(
    (jobId: string, bufferReplay = false) => {
      if (clientRef.current) {
        clientRef.current.disconnect()
        clientRef.current = null
      }

      activeStepIdsRef.current.clear()
      activeToolIdsByEventRef.current.clear()
      activeToolIdsByNameRef.current.clear()
      resetTimeout()

      // ---------- inline buffer for replay phase ----------
      // bufferReplay=true on page-refresh reconnect: buffer ALL replayed events
      // with zero store writes (like streamFullJob). The backend sends a
      // stream.mode event with mode="live" when replay is done. On that signal,
      // flush the buffer in one setState and switch to per-event live streaming.
      // A safety timeout (30s) flushes if the signal never arrives.
      // bufferReplay=false for fresh new jobs: events go straight to store.
      const SAFETY_TIMEOUT_MS = 30000
      const buf = {
        active: bufferReplay,
        timer: null as NodeJS.Timeout | null,
        idCounter: 0,
        activeLLMStack: [] as string[],
        activeToolStacks: new Map<string, string[]>(),
        agents: new Map<string, { name: string; input?: string; output?: string; startedAt?: string; completedAt?: string }>(),
        llmSteps: new Map<string, { name: string; workflow?: string; content: string; thinking?: string; usage?: { input_tokens: number; output_tokens: number }; timestamp?: string }>(),
        toolCalls: new Map<string, { name: string; input?: Record<string, unknown>; output?: string; workflow?: string; agentId?: string; timestamp?: string }>(),
        todos: null as TodoItem[] | null,
        todoGroups: new Map<string, { todos: TodoItem[]; workflow?: string; agentId?: string; source?: string; timestamp?: string }>(),
        citations: [] as Array<{ url: string; content: string; isCited: boolean; timestamp?: string; title?: string; sourceClass?: string; publishedDate?: string }>,
        files: new Map<string, { content: string; timestamp?: string }>(),
        reportContent: null as string | null,
      }

      /** Flush buffer to store in one setState, deactivate buffer, switch to live. */
      const flushBuffer = (): void => {
        if (!buf.active) return
        buf.active = false
        if (buf.timer) { clearTimeout(buf.timer); buf.timer = null }

        const agents = Array.from(buf.agents.entries()).map(([id, a]) => ({ id, name: a.name, input: a.input, output: a.output, status: 'complete' as const, startedAt: parseEventDate(a.startedAt), completedAt: parseEventDate(a.completedAt ?? a.startedAt) }))
        const llmSteps = Array.from(buf.llmSteps.entries()).map(([id, s]) => ({ id, name: s.name, workflow: s.workflow, content: s.content, thinking: s.thinking, usage: s.usage, isComplete: true, timestamp: parseEventDate(s.timestamp) }))
        const toolCalls = Array.from(buf.toolCalls.entries()).map(([id, t]) => ({ id, name: t.name, input: t.input, output: t.output, workflow: t.workflow, agentId: t.agentId, status: 'complete' as const, timestamp: parseEventDate(t.timestamp) }))
        const citations = buf.citations.map((c, i) => ({ id: `citation-${i}`, url: c.url, content: c.content, isCited: c.isCited, timestamp: parseEventDate(c.timestamp), title: c.title, sourceClass: c.sourceClass, publishedDate: c.publishedDate }))
        const files = Array.from(buf.files.entries()).map(([filename, file], i) => ({ id: `file-${i}`, filename, content: file.content, timestamp: parseEventDate(file.timestamp) }))
        const todos = buf.todos ? mapDeepResearchTodos(buf.todos) : undefined
        const todoGroups = Array.from(buf.todoGroups.values()).map((group) =>
          buildDeepResearchTodoGroup(group.todos, group)
        )

        useChatStore.setState((state) => ({
          ...(buf.reportContent !== null && { reportContent: buf.reportContent }),
          ...(todos && todos.length > 0 && { deepResearchTodos: todos }),
          ...(todoGroups.length > 0 && { deepResearchTodoGroups: todoGroups }),
          ...(agents.length > 0 && { deepResearchAgents: agents }),
          ...(llmSteps.length > 0 && { deepResearchLLMSteps: llmSteps }),
          ...(toolCalls.length > 0 && { deepResearchToolCalls: toolCalls }),
          ...(citations.length > 0 && { deepResearchCitations: citations }),
          ...(files.length > 0 && { deepResearchFiles: files }),
          currentStatus: buf.reportContent !== null ? 'writing' : state.currentStatus,
        }))

        // Bridge in-progress items to live mode so their end events
        // (llm.end with usage, tool.end with output) can find them
        for (const id of buf.activeLLMStack) {
          const step = buf.llmSteps.get(id)
          if (step) {
            activeStepIdsRef.current.set(`llm:${step.name}`, id)
            activeStepIdsRef.current.set(`llmStep:${step.name}`, id)
          }
        }
        for (const [name, stack] of buf.activeToolStacks) {
          const lastId = stack[stack.length - 1]
          if (lastId) {
            activeStepIdsRef.current.set(`tool:${name}`, lastId)
            activeStepIdsRef.current.set(`toolCall:${name}`, lastId)
          }
        }
      }

      // Safety timeout: flush if the backend never sends the live signal
      if (bufferReplay) {
        buf.timer = setTimeout(flushBuffer, SAFETY_TIMEOUT_MS)
      }

      // Create SSE client — callbacks check buf.active to decide buffer vs real-time
      const lastEventId = useChatStore.getState().deepResearchLastEventId ?? undefined
      const client = createDeepResearchClient({
        jobId,
        authToken: idToken || undefined,
        lastEventId,
        callbacks: {
          onStreamStart: () => {
            if (buf.active) return
            if (!isOwnerActive(jobId)) return
            clearReconnectTimer()
            resetTimeout()
            researchStartTimeRef.current = Date.now()
            setCurrentStatus('researching')
            setDeepResearchActivity({ kind: 'status', message: 'Research stream connected', detail: jobId })
            setDeepResearchLastEventId(client.getLastEventId())
          },

          onStreamMode: (mode) => {
            if (isDeepResearchReplayCompleteMode(mode) && buf.active) {
              flushBuffer()
              setCurrentStatus('researching')
              setDeepResearchActivity({ kind: 'status', message: 'Live research stream active' })
              setDeepResearchLastEventId(client.getLastEventId())
            }
          },

          onJobStatus: async (status, error) => {
            if (buf.active) flushBuffer()
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            setDeepResearchLastEventId(client.getLastEventId())
            // Clear the cancel-fallback timer — the SSE stream delivered
            // the terminal status so optimistic cleanup is unnecessary.
            if (cancelFallbackRef.current) {
              clearTimeout(cancelFallbackRef.current)
              cancelFallbackRef.current = null
            }
            const state = useChatStore.getState()
            const ownerConvId = state.deepResearchOwnerConversationId
            const messageId = state.activeDeepResearchMessageId

            if (isRecoverableInterruptedStatus(status, error)) {
              setDeepResearchActivity({
                kind: 'status',
                message: 'Research worker heartbeat was lost. Checking for active work...',
                detail: error,
              })
              const resumed = await tryAutoResume(jobId, error)
              if (resumed) return
            }

            updateDeepResearchStatus(status)
            const statusMessage = (() => {
              if (status === 'submitted') return 'Research job queued'
              if (status === 'running') return 'Research job running'
              if (status === 'success') return 'Research completed'
              if (status === 'interrupted') {
                return isUserCancelledStatus(status, error) ? 'Research cancelled' : 'Research interrupted'
              }
              return 'Research failed'
            })()
            setDeepResearchActivity({
              kind: status === 'failure' ? 'warning' : 'status',
              message: statusMessage,
              detail: error,
            })

            if (status === 'success') {
              setCurrentStatus('complete')
              const { reportContent: currentReport, deepResearchLLMSteps, deepResearchToolCalls } = state
              const totalTokens = deepResearchLLMSteps.reduce((sum, step) => sum + (step.usage?.input_tokens || 0) + (step.usage?.output_tokens || 0), 0)
              const toolCallCount = deepResearchToolCalls.length
              const hasReport = Boolean(currentReport?.trim())

              if (ownerConvId && messageId) {
                patchConversationMessage(ownerConvId, messageId, {
                  content: '',
                  deepResearchJobStatus: 'success',
                  isDeepResearchActive: false,
                  showViewReport: hasReport,
                  reportContent: hasReport ? currentReport : undefined,
                })
              }
              addDeepResearchBanner('success', jobId, ownerConvId || undefined, { totalTokens, toolCallCount })
              researchStartTimeRef.current = null
              stopAllDeepResearchSpinners(true)
              setStreamLoaded(true)
              completeDeepResearch()
              setStreaming(false)
            } else if (status === 'failure' || status === 'interrupted') {
              setCurrentStatus('error')
              stopAllDeepResearchSpinners()
              const hasReport = Boolean(state.reportContent?.trim())
              const isUserCancelled = isUserCancelledStatus(status, error)

              if (ownerConvId && messageId) {
                patchConversationMessage(ownerConvId, messageId, {
                  content: '',
                  deepResearchJobStatus: status,
                  isDeepResearchActive: false,
                  showViewReport: hasReport,
                  reportContent: hasReport ? state.reportContent : undefined,
                })
              }
              addDeepResearchBanner(isUserCancelled ? 'cancelled' : 'failure', jobId, ownerConvId || undefined)
              researchStartTimeRef.current = null
              clientRef.current?.disconnect()
              setStreamLoaded(true)
              completeDeepResearch()
              setStreaming(false)
              if (error && !isUserCancelled) {
                const { addErrorCard } = useChatStore.getState()
                addErrorCard('agent.deep_research_failed', error)
              } else if (status === 'interrupted' && !isUserCancelled) {
                const { addErrorCard } = useChatStore.getState()
                addErrorCard('agent.deep_research_failed', 'Research was interrupted before completion.')
              }
            }
          },

          onHeartbeat: () => {
            if (buf.active) return
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            setDeepResearchActivity({ kind: 'status', message: 'Research job is still active' }, { preserveMessage: true })
          },

          onWorkflowStart: (name, input, eventId, agentId, timestamp) => {
            const id = agentId || eventId || `agent-${buf.idCounter++}`
            if (buf.active) {
              if (!buf.agents.has(id)) buf.agents.set(id, { name, input: input ? (typeof input === 'string' ? input : JSON.stringify(input)) : undefined, startedAt: timestamp })
              return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            setDeepResearchActivity({
              kind: 'agent',
              message: `${name} started`,
              detail: input,
            })
            const hasUserMsg = Boolean(useChatStore.getState().currentUserMessageId)
            if (hasUserMsg) {
              const stepId = addThinkingStep({ category: 'agents', functionName: name, displayName: name, content: input ? `Input: ${input}\n` : 'Starting...\n', isComplete: false, isDeepResearch: true })
              activeStepIdsRef.current.set(name, stepId)
            }
            const createdId = addDeepResearchAgentWithId(id, {
              name,
              input,
              ...(timestamp ? { startedAt: parseEventDate(timestamp) } : {}),
            })
            activeStepIdsRef.current.set(`agent:${id}`, createdId)
          },

          onWorkflowEnd: (name, output, _eventId, agentId, timestamp) => {
            if (buf.active) {
              if (agentId) { const a = buf.agents.get(agentId); if (a) { a.output = output ? (typeof output === 'string' ? output : JSON.stringify(output)) : undefined; a.completedAt = timestamp } }
              return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            setDeepResearchActivity({
              kind: 'agent',
              message: `${name} completed`,
              detail: output ? output.slice(0, 240) : undefined,
            })
            const stepId = activeStepIdsRef.current.get(name)
            if (stepId) { if (output) appendToThinkingStep(stepId, `\nOutput: ${output}`); completeThinkingStep(stepId); activeStepIdsRef.current.delete(name) }
            if (agentId) {
              if (timestamp) completeDeepResearchAgent(agentId, output, parseEventDate(timestamp))
              else completeDeepResearchAgent(agentId, output)
              activeStepIdsRef.current.delete(`agent:${agentId}`)
            }
          },

          onLLMStart: (name, workflow, timestamp) => {
            if (buf.active) {
              const id = `llm-${buf.idCounter++}`; buf.activeLLMStack.push(id); buf.llmSteps.set(id, { name, workflow, content: '', timestamp }); return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            setDeepResearchActivity({
              kind: 'model',
              message: workflow ? `${workflow} is thinking` : `${name} is thinking`,
            })

            const hasUserMsg = Boolean(useChatStore.getState().currentUserMessageId)
            if (hasUserMsg) {
              const displayName = workflow ? `${workflow} > ${name}` : name
              const stepId = addThinkingStep({ category: 'agents', functionName: `llm:${name}`, displayName, content: 'Generating...\n', isComplete: false, isDeepResearch: true })
              activeStepIdsRef.current.set(`llm:${name}`, stepId)
            }
            const llmStepId = addDeepResearchLLMStep({
              name,
              workflow,
              content: '',
              ...(timestamp ? { timestamp: parseEventDate(timestamp) } : {}),
            })
            activeStepIdsRef.current.set(`llmStep:${name}`, llmStepId)
          },

          onLLMChunk: (chunk) => {
            if (buf.active) {
              const id = buf.activeLLMStack[buf.activeLLMStack.length - 1]; if (id) { const s = buf.llmSteps.get(id); if (s) s.content += chunk }; return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            const llmStepId = Array.from(activeStepIdsRef.current.entries()).filter(([k]) => k.startsWith('llm:')).pop()?.[1]
            if (llmStepId) appendToThinkingStep(llmStepId, chunk)
            const llmStepKeys = Array.from(activeStepIdsRef.current.entries()).filter(([k]) => k.startsWith('llmStep:'))
            if (llmStepKeys.length > 0) appendToDeepResearchLLMStep(llmStepKeys[llmStepKeys.length - 1][1], chunk)
          },

          onLLMEnd: (_output, thinking, usage) => {
            if (buf.active) {
              const id = buf.activeLLMStack.pop(); if (id) { const s = buf.llmSteps.get(id); if (s) { s.thinking = thinking; s.usage = usage } }; return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            setDeepResearchActivity({ kind: 'model', message: 'Model response completed' })
            const llmSteps = Array.from(activeStepIdsRef.current.entries()).filter(([k]) => k.startsWith('llm:'))
            if (llmSteps.length > 0) { const [key, stepId] = llmSteps[llmSteps.length - 1]; if (thinking) appendToThinkingStep(stepId, `\n\nThinking: ${thinking}`); completeThinkingStep(stepId); activeStepIdsRef.current.delete(key) }
            const llmStepKeys = Array.from(activeStepIdsRef.current.entries()).filter(([k]) => k.startsWith('llmStep:'))
            if (llmStepKeys.length > 0) { const [key, llmStepId] = llmStepKeys[llmStepKeys.length - 1]; completeDeepResearchLLMStep(llmStepId, thinking, usage); activeStepIdsRef.current.delete(key) }
          },

          onToolStart: (name, input, workflow, eventId, agentId, timestamp) => {
            if (name === 'task') return
            if (buf.active) {
              const id = `tool-${buf.idCounter++}`; buf.toolCalls.set(id, { name, input, workflow, agentId, timestamp })
              let stack = buf.activeToolStacks.get(name); if (!stack) { stack = []; buf.activeToolStacks.set(name, stack) }; stack.push(id); return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout(); setCurrentStatus(getToolStatus(name, workflow))
            setDeepResearchActivity(getToolActivity(name, input))
            const hasUserMsg = Boolean(useChatStore.getState().currentUserMessageId)
            let thinkingStepId: string | undefined
            if (hasUserMsg) {
              const inputText = input ? ('_raw' in input && typeof input._raw === 'string' ? input._raw : JSON.stringify(input, null, 2)) : null
              thinkingStepId = addThinkingStep({ category: 'tools', functionName: name, displayName: name, content: inputText ? `Input: ${inputText}\n` : 'Executing...\n', isComplete: false, isDeepResearch: true })
            }
            const toolCallId = addDeepResearchToolCall({
              name,
              input,
              workflow,
              agentId,
              ...(timestamp ? { timestamp: parseEventDate(timestamp) } : {}),
            })
            const ids = { thinkingStepId, toolCallId }
            if (eventId) activeToolIdsByEventRef.current.set(eventId, ids)
            const stack = activeToolIdsByNameRef.current.get(name) || []
            stack.push(ids)
            activeToolIdsByNameRef.current.set(name, stack)
          },

          onToolEnd: (name, output, eventId) => {
            if (name === 'task') return
            if (buf.active) {
              const stack = buf.activeToolStacks.get(name); const id = stack?.pop(); if (id) { const t = buf.toolCalls.get(id); if (t) t.output = output ? JSON.stringify(output) : undefined }; return
            }
            if (!isOwnerActive(jobId)) return
            resetTimeout()
            const stack = activeToolIdsByNameRef.current.get(name) || []
            const ids =
              eventId && activeToolIdsByEventRef.current.has(eventId)
                ? activeToolIdsByEventRef.current.get(eventId)
                : stack.pop()
            if (eventId) activeToolIdsByEventRef.current.delete(eventId)
            if (ids && eventId) {
              const index = stack.indexOf(ids)
              if (index >= 0) stack.splice(index, 1)
            }
            if (stack.length > 0) activeToolIdsByNameRef.current.set(name, stack)
            else activeToolIdsByNameRef.current.delete(name)

            if (ids?.thinkingStepId) {
              if (output) { const truncated = output.length > 500 ? output.substring(0, 500) + '...' : output; appendToThinkingStep(ids.thinkingStepId, `\nOutput: ${truncated}`) }
              completeThinkingStep(ids.thinkingStepId)
            }
            if (ids?.toolCallId) completeDeepResearchToolCall(ids.toolCallId, output)
            setDeepResearchActivity({
              ...getToolActivity(name),
              message: `${getToolActivity(name).message} completed`,
              detail: output ? output.slice(0, 240) : undefined,
            })
            setCurrentStatus('researching')
          },

          onTodoUpdate: (todos: TodoItem[], workflow?: string, timestamp?: string, agentId?: string, source?: string) => {
            const isWorkflowScoped = Boolean(workflow || agentId || source === 'agent')
            if (isWorkflowScoped) {
              if (buf.active) {
                const group = buildDeepResearchTodoGroup(todos, { workflow, agentId, source, timestamp })
                buf.todoGroups.set(group.id, { todos, workflow, agentId, source, timestamp })
                return
              }
              if (!isOwnerActive(jobId)) return
              resetTimeout()
              setDeepResearchTodoGroup(todos, { workflow, agentId, source, timestamp })
              const activeTodo = todos.find((todo) => todo.status === 'in_progress')
              if (activeTodo) {
                setCurrentStatus(workflow?.toLowerCase().includes('planner') ? 'planning' : 'researching')
                setDeepResearchActivity({
                  kind: 'agent',
                  message: `${workflow || 'Research lane'} progress`,
                  detail: activeTodo.content,
                })
              }
              return
            }
            if (buf.active) { buf.todos = todos; return }
            if (!isOwnerActive(jobId)) return
            resetTimeout(); setDeepResearchTodos(todos)
            const activeTodo = todos.find((todo) => todo.status === 'in_progress')
            if (activeTodo) {
              setDeepResearchActivity({ kind: 'status', message: 'Research task in progress', detail: activeTodo.content })
            }

          },

          onCitationUpdate: (url, content, isCited, timestamp, meta) => {
            if (buf.active) { buf.citations.push({ url, content, isCited: isCited ?? false, timestamp, title: meta?.title, sourceClass: meta?.sourceClass, publishedDate: meta?.publishedDate }); return }
            if (!isOwnerActive(jobId)) return
            resetTimeout(); addDeepResearchCitation(url, content, isCited, meta)
            setDeepResearchActivity({
              kind: 'search',
              message: isCited ? 'Added cited source' : 'Discovered source',
              detail: url,
            })
          },

          onFileUpdate: (filename, content, timestamp) => {
            if (buf.active) { buf.files.set(filename, { content, timestamp }); return }
            if (!isOwnerActive(jobId)) return
            resetTimeout(); addDeepResearchFile({
              filename,
              content,
              ...(timestamp ? { timestamp: parseEventDate(timestamp) } : {}),
            })
            if (isClaudeResearchFile(filename)) {
              const hasUserMsg = Boolean(useChatStore.getState().currentUserMessageId)
              if (hasUserMsg) {
                const stepId = addThinkingStep({
                  category: 'agents',
                  functionName: `claude_artifact:${filename}:${timestamp || Date.now()}`,
                  displayName: `GenAlphAI updated ${basename(filename)}`,
                  content: buildClaudeArtifactPreview(filename, content),
                  isComplete: false,
                })
                if (stepId) completeThinkingStep(stepId)
              }
            }
            setDeepResearchActivity({
              kind: filename.endsWith('report.md') || filename.endsWith('final.md') ? 'report' : 'file',
              message: isClaudeResearchFile(filename)
                ? `GenAlphAI updated ${basename(filename)}`
                : filename.endsWith('report.md')
                  ? 'Rendering report draft'
                  : 'Saved research file',
              detail: filename,
            })
            // report.md artifact arrives 1-2 min before the final_report output event —
            // render it immediately so the final report panel never stays blank
            // if the terminal assistant message is only a short status line.
            if (filename.endsWith('report.md') || filename.endsWith('final.md')) {
              setReportContent(content, 'final_report')
              setCurrentStatus('writing')
            }
          },

          onOutputUpdate: (content, outputCategory, _workflow) => {
            if (outputCategory === 'intermediate') return
            if (buf.active) {
              if (outputCategory === 'final_report' || !outputCategory) { buf.reportContent = content }
              // research_notes are already captured via write_file artifacts — skip to avoid duplicates
              return
            }
            if (!isOwnerActive(jobId)) return
            if (outputCategory === 'research_notes') {
              // Skip — research notes are already tracked via write_file tool artifacts
              void 0
            } else if (outputCategory === 'final_report' || !outputCategory) {
              setReportContent(content)
              setCurrentStatus('writing')
              setDeepResearchActivity({ kind: 'report', message: 'Rendering final report' })
            }
          },

          onComplete: () => {
            if (buf.active) flushBuffer()
          },

          onError: async (error) => {
            console.warn('Deep research SSE error:', error.message)
            if (buf.active) flushBuffer()
            const { isDeepResearchStreaming, deepResearchStatus } = useChatStore.getState()
            if (isDeepResearchStreaming && deepResearchStatus !== 'interrupted' && deepResearchStatus !== 'failure') {
              const backendUp = await checkBackendHealthCached()

              if (backendUp) {
                try {
                  const statusResponse = await getJobStatus(jobId, idToken || undefined)
                  const serverStatus = statusResponse.status

                  if (serverStatus === 'submitted' || serverStatus === 'running') {
                    console.warn(
                      'Deep research SSE transport failed, but backend job is still active. Reconnecting:',
                      jobId,
                      serverStatus
                    )
                    updateDeepResearchStatus(serverStatus)
                    setCurrentStatus('researching')
                    scheduleReconnect(jobId, error.message)
                    return
                  }

                  if (serverStatus === 'success') {
                    const state = useChatStore.getState()
                    const ownerConvId = state.deepResearchOwnerConversationId
                    const messageId = state.activeDeepResearchMessageId
                    const hasReport = Boolean(state.reportContent?.trim())
                    if (ownerConvId && messageId) {
                      patchConversationMessage(ownerConvId, messageId, {
                        content: '',
                        deepResearchJobStatus: 'success',
                        isDeepResearchActive: false,
                        showViewReport: hasReport,
                        reportContent: hasReport ? state.reportContent : undefined,
                      })
                    }
                    addDeepResearchBanner('success', jobId, ownerConvId || undefined)
                    updateDeepResearchStatus('success')
                    setCurrentStatus('complete')
                    stopAllDeepResearchSpinners(true)
                    setStreamLoaded(true)
                    completeDeepResearch()
                    setStreaming(false)
                    return
                  }

                  if (serverStatus === 'failure' || serverStatus === 'interrupted') {
                    if (isRecoverableInterruptedStatus(serverStatus, statusResponse.error ?? error.message)) {
                      const resumed = await tryAutoResume(jobId, statusResponse.error ?? error.message)
                      if (resumed) return
                    }

                    const errorInfo = getDeepResearchStreamFailure(error.message, error.stack)
                    console.error('Deep research SSE failed and backend reported terminal status:', error)
                    setCurrentStatus('error')

                    const state = useChatStore.getState()
                    const ownerConvId = state.deepResearchOwnerConversationId
                    const messageId = state.activeDeepResearchMessageId
                    const hasReport = Boolean(state.reportContent?.trim())
                    const isUserCancelled = isUserCancelledStatus(serverStatus, statusResponse.error ?? undefined)

                    if (ownerConvId && messageId) {
                      patchConversationMessage(ownerConvId, messageId, {
                        content: '',
                        deepResearchJobStatus: serverStatus,
                        isDeepResearchActive: false,
                        showViewReport: hasReport,
                        reportContent: hasReport ? state.reportContent : undefined,
                      })
                    }

                    if (!isUserCancelled) {
                      state.addErrorCard(
                        errorInfo.code as Parameters<typeof state.addErrorCard>[0],
                        statusResponse.error || errorInfo.message,
                        errorInfo.details
                      )
                    }
                    addDeepResearchBanner(isUserCancelled ? 'cancelled' : 'failure', jobId, ownerConvId || undefined)
                    stopAllDeepResearchSpinners()
                    clientRef.current?.disconnect()
                    setStreamLoaded(true)
                    completeDeepResearch()
                    setStreaming(false)
                    return
                  }
                } catch (statusError) {
                  console.warn('Could not verify deep research job status after SSE error:', statusError)
                }
              }

              console.warn(
                backendUp
                  ? 'Deep research SSE failed but terminal status could not be verified. Keeping job active.'
                  : 'Deep research SSE failed while backend was unreachable. Keeping job active.',
                error
              )
              scheduleReconnect(jobId, error.message)
            }
          },

          onDisconnect: () => {
            if (buf.active) flushBuffer()
            const state = useChatStore.getState()
            if (
              state.deepResearchJobId === jobId &&
              state.isDeepResearchStreaming &&
              state.deepResearchStatus !== 'success' &&
              state.deepResearchStatus !== 'failure' &&
              state.deepResearchStatus !== 'interrupted'
            ) {
              console.warn('Deep research SSE disconnected while job is active. Reconnecting:', jobId)
              clientRef.current = null
              scheduleReconnect(jobId, 'SSE disconnected')
            }
          },
        },
      })

      clientRef.current = client
      client.connect()
    },
    [
      idToken, resetTimeout, isOwnerActive, updateDeepResearchStatus, completeDeepResearch,
      addDeepResearchCitation, setReportContent, addThinkingStep, appendToThinkingStep,
      completeThinkingStep, setCurrentStatus, setDeepResearchTodos, setDeepResearchTodoGroup, stopAllDeepResearchSpinners,
      addDeepResearchLLMStep, appendToDeepResearchLLMStep,
      completeDeepResearchLLMStep, addDeepResearchAgentWithId, completeDeepResearchAgent,
      addDeepResearchToolCall, completeDeepResearchToolCall, addDeepResearchFile,
      patchConversationMessage, addDeepResearchBanner, setStreaming, setStreamLoaded,
      setDeepResearchLastEventId, setDeepResearchActivity, getDeepResearchStreamFailure,
      clearReconnectTimer, scheduleReconnect, tryAutoResume,
    ]
  )

  // Keep ref in sync so the effect always uses the latest connect without re-triggering
  connectRef.current = connect

  /**
   * Disconnect from the SSE stream
   */
  const disconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.disconnect()
      clientRef.current = null
    }
  }, [])

  /**
   * Reconnect to the SSE stream from the beginning
   */
  const reconnect = useCallback(() => {
    if (deepResearchJobId && !clientRef.current?.isConnected()) {
      connectRef.current?.(deepResearchJobId, true)
    }
  }, [deepResearchJobId])

  /**
   * Cancel the current job (useful for hung jobs)
   */
  const cancelCurrentJob = useCallback(async () => {
    if (!deepResearchJobId) return
    const cancelledJobId = deepResearchJobId

    try {
      await cancelJob(cancelledJobId, idToken || undefined)
      setIsTimedOut(false)

      // Fallback: if the SSE stream is broken or stalled and never delivers
      // the job.status: "interrupted" event, clean up locally after a short
      // grace period so the UI doesn't stay stuck in "streaming" state.
      // If the SSE event arrives in time, onJobStatus clears this timer.
      if (cancelFallbackRef.current) clearTimeout(cancelFallbackRef.current)
      cancelFallbackRef.current = setTimeout(() => {
        cancelFallbackRef.current = null
        const state = useChatStore.getState()
        if (!state.isDeepResearchStreaming || state.deepResearchJobId !== cancelledJobId) {
          return // SSE already handled cleanup — nothing to do
        }
        console.warn(
          '[DeepResearch] Cancel fallback: SSE did not deliver interrupted status within',
          CANCEL_FALLBACK_TIMEOUT_MS,
          'ms. Cleaning up locally.'
        )
        const ownerConvId = state.deepResearchOwnerConversationId
        const messageId = state.activeDeepResearchMessageId
        const hasReport = Boolean(state.reportContent?.trim())
        if (ownerConvId && messageId) {
          patchConversationMessage(ownerConvId, messageId, {
            content: '',
            deepResearchJobStatus: 'interrupted',
            isDeepResearchActive: false,
            showViewReport: hasReport,
            reportContent: hasReport ? state.reportContent : undefined,
          })
        }
        addDeepResearchBanner('cancelled', cancelledJobId, ownerConvId || undefined)
        stopAllDeepResearchSpinners()
        clientRef.current?.disconnect()
        clientRef.current = null
        setStreamLoaded(true)
        completeDeepResearch()
        setStreaming(false)
      }, CANCEL_FALLBACK_TIMEOUT_MS)
    } catch (error) {
      console.error('Failed to cancel job:', error)
    }
  }, [deepResearchJobId, idToken, patchConversationMessage, addDeepResearchBanner, stopAllDeepResearchSpinners, completeDeepResearch, setStreaming, setStreamLoaded])

  /**
   * Auto-connect when job ID changes
   * Uses lastEventId from store for reconnection scenarios (session restore, tab reopen)
   */
  useEffect(() => {
    // Capture values at effect start to detect stale effects
    const effectJobId = deepResearchJobId
    const effectStreaming = isDeepResearchStreaming
    let connectTimeout: NodeJS.Timeout | null = null
    let cancelled = false

    if (effectJobId && effectStreaming) {
      // Verify state hasn't changed before connecting (prevents race conditions)
      const currentState = useChatStore.getState()
      if (currentState.deepResearchJobId !== effectJobId || !currentState.isDeepResearchStreaming) {
        return // State changed, don't connect
      }

      // Defer connect by 50ms so React StrictMode cleanup can cancel it.
      // StrictMode sequence: mount1-effect → mount1-cleanup → mount2-effect.
      // The cleanup clears the timeout, preventing mount1 from ever connecting.
      // Only mount2's deferred connect actually fires.
      connectTimeout = setTimeout(async () => {
        if (cancelled) return

        // Determine if this is a reconnection (page refresh) or a fresh job start.
        // Fresh jobs (status 'submitted') use per-event store writes for live updates.
        // Reconnections (status 'running') buffer historical events then flush once
        // when the backend sends stream.mode: "live".
        const isReconnect = useChatStore.getState().deepResearchStatus !== 'submitted'
        connectRef.current?.(effectJobId, isReconnect)

        setResearchPanelTab('tasks')
        openRightPanel('research')

        // Start timeout check interval
        timeoutIntervalRef.current = setInterval(() => {
          const timeSinceLastEvent = Date.now() - lastEventTimeRef.current
          if (timeSinceLastEvent > TIMEOUT_WARNING_MS) {
            setIsTimedOut(true)
          }
        }, TIMEOUT_CHECK_INTERVAL_MS)
      }, 50)

      // Session persistence is now handled by debounced resetTimeout()
      // (fires 2s after each event instead of fixed 10s interval)
    }

    return () => {
      cancelled = true
      // Cancel the deferred connect if it hasn't fired yet
      if (connectTimeout) clearTimeout(connectTimeout)
      // Cleanup on unmount or job ID change
      disconnect()
      // Clear timeout interval
      if (timeoutIntervalRef.current) {
        clearInterval(timeoutIntervalRef.current)
        timeoutIntervalRef.current = null
      }
      clearReconnectTimer()
      // Clear cancel fallback timer
      if (cancelFallbackRef.current) {
        clearTimeout(cancelFallbackRef.current)
        cancelFallbackRef.current = null
      }
      setIsTimedOut(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- connectRef avoids re-triggering on token refresh; store actions are stable refs
  }, [deepResearchJobId, isDeepResearchStreaming, disconnect, setResearchPanelTab, openRightPanel])

  return {
    isStreaming: isDeepResearchStreaming,
    jobId: deepResearchJobId,
    status: deepResearchStatus,
    isTimedOut,
    disconnect,
    reconnect,
    cancelCurrentJob,
  }
}

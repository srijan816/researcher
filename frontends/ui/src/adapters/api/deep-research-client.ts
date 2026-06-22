// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Deep Research SSE Client
 *
 * Handles Server-Sent Events streaming for deep research async jobs.
 * Uses native EventSource for proper SSE protocol support including
 * event types, event IDs, and automatic reconnection.
 *
 * @see docs/api.md - Deep Research API (Async Jobs) section
 */

import { apiConfig } from './config'

// ============================================================
// Types
// ============================================================

/** Job status values */
export type DeepResearchJobStatus = 'submitted' | 'running' | 'success' | 'failure' | 'interrupted'

/** Backend-backed research history item. */
export interface JobHistoryItem {
  job_id: string
  status: DeepResearchJobStatus
  agent_type?: string | null
  input?: string | null
  title?: string | null
  error?: string | null
  created_at?: string | null
  updated_at?: string | null
  has_report: boolean
  report_ready?: boolean
  terminal?: boolean
  poll_after_seconds?: number | null
  message?: string | null
  status_url?: string | null
  report_url?: string | null
  state_url?: string | null
  stream_url?: string | null
}

/** Backend-backed research history response. */
export interface JobHistoryResponse {
  jobs: JobHistoryItem[]
}

/** Backend job status response. */
export interface DeepResearchJobStatusResponse {
  job_id: string
  status: DeepResearchJobStatus
  agent_type?: string | null
  error: string | null
  created_at?: string | null
  updated_at?: string | null
  has_report?: boolean
  report_ready?: boolean
  terminal?: boolean
  poll_after_seconds?: number | null
  message?: string | null
  status_url?: string | null
  report_url?: string | null
  state_url?: string | null
  stream_url?: string | null
}

/** Backend final-report response. */
export interface DeepResearchJobReportResponse {
  job_id: string
  has_report: boolean
  report_ready?: boolean
  terminal?: boolean
  poll_after_seconds?: number | null
  message?: string | null
  status?: DeepResearchJobStatus
  error?: string | null
  content_type?: string
  report: string | null
  report_markdown?: string | null
  status_url?: string | null
  report_url?: string | null
  state_url?: string | null
  stream_url?: string | null
  sources_found?: number | null
  sources_cited?: number | null
  found_urls?: string[] | null
  cited_urls?: string[] | null
}

/** Async job submission payload for deep research. */
export interface DeepResearchJobSubmitRequest {
  agent_type: string
  input: string
  data_sources?: string[] | null
  research_depth?: 'shallow' | 'medium' | 'deeper' | 'deep'
  job_id?: string | null
  expiry_seconds?: number | null
}

/** SSE event types from the deep research stream */
export type DeepResearchEventType =
  | 'stream.start'
  | 'stream.mode'
  | 'job.status'
  | 'job.heartbeat'
  | 'workflow.start'
  | 'workflow.end'
  | 'llm.start'
  | 'llm.chunk'
  | 'llm.end'
  | 'tool.start'
  | 'tool.end'
  | 'artifact.update'

/** Artifact types in artifact.update events */
export type ArtifactType = 'todo' | 'citation_source' | 'citation_use' | 'file' | 'output'

/** Base SSE event structure */
export interface DeepResearchSSEEvent {
  event: DeepResearchEventType
  id?: string
  timestamp?: string
}

/** stream.start event */
export interface StreamStartEvent extends DeepResearchSSEEvent {
  event: 'stream.start'
  job_id: string
}

/** job.status event */
export interface JobStatusEvent extends DeepResearchSSEEvent {
  event: 'job.status'
  data: {
    status: DeepResearchJobStatus
    error?: string
  }
}

/** workflow.start event */
export interface WorkflowStartEvent extends DeepResearchSSEEvent {
  event: 'workflow.start'
  data: {
    name: string
    data?: {
      input?: string
    }
  }
}

/** workflow.end event */
export interface WorkflowEndEvent extends DeepResearchSSEEvent {
  event: 'workflow.end'
  data: {
    name: string
    data?: {
      output?: string
    }
  }
}

/** llm.start event */
export interface LLMStartEvent extends DeepResearchSSEEvent {
  event: 'llm.start'
  data: {
    name: string
    metadata?: {
      workflow?: string
    }
  }
}

/** llm.chunk event */
export interface LLMChunkEvent extends DeepResearchSSEEvent {
  event: 'llm.chunk'
  data: {
    chunk: string
  }
}

/** llm.end event */
export interface LLMEndEvent extends DeepResearchSSEEvent {
  event: 'llm.end'
  data: {
    output: string
    metadata?: {
      thinking?: string
      usage?: {
        prompt_tokens: number
        completion_tokens: number
      }
    }
  }
}

/** tool.start event */
export interface ToolStartEvent extends DeepResearchSSEEvent {
  event: 'tool.start'
  data: {
    name: string
    data?: {
      input?: Record<string, unknown>
    }
    metadata?: {
      workflow?: string
      agent_id?: string
    }
  }
}

/** tool.end event */
export interface ToolEndEvent extends DeepResearchSSEEvent {
  event: 'tool.end'
  data: {
    name: string
    data?: {
      output?: string
    }
  }
}

/** Optional metadata carried on citation artifact events */
export interface CitationEventMeta {
  /** Source page title when the backend extracted one */
  title?: string
  /** Source classification (first_party, academic, trade_press, forum, unknown, ...) */
  sourceClass?: string
  /** Citation key for non-URL sources */
  citationKey?: string
  /** Published date (ISO string) when available */
  publishedDate?: string
}

/** Todo item in artifact.update */
export interface TodoItem {
  id: string
  content: string
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled'
}

/** artifact.update event */
export interface ArtifactUpdateEvent extends DeepResearchSSEEvent {
  event: 'artifact.update'
  data: {
    id: string
    timestamp: string
    data: {
      type: ArtifactType
      content: string | TodoItem[]
      url?: string // For citation_source and citation_use types
    }
    metadata?: {
      workflow?: string
    }
  }
}

/** Union of all SSE event types */
export type DeepResearchEvent =
  | StreamStartEvent
  | JobStatusEvent
  | WorkflowStartEvent
  | WorkflowEndEvent
  | LLMStartEvent
  | LLMChunkEvent
  | LLMEndEvent
  | ToolStartEvent
  | ToolEndEvent
  | ArtifactUpdateEvent

// ============================================================
// Callbacks
// ============================================================

/** Callbacks for deep research SSE events */
export interface DeepResearchCallbacks {
  /** Called when stream connection is established */
  onStreamStart?: (jobId: string) => void
  /** Called when stream mode changes (e.g., replay → live) */
  onStreamMode?: (mode: string) => void
  /** Called on job status updates */
  onJobStatus?: (status: DeepResearchJobStatus, error?: string) => void
  /** Called on workflow events */
  onWorkflowStart?: (name: string, input?: string, eventId?: string, agentId?: string, timestamp?: string) => void
  onWorkflowEnd?: (name: string, output?: string, eventId?: string, agentId?: string, timestamp?: string) => void
  /** Called on LLM events */
  onLLMStart?: (name: string, workflow?: string, timestamp?: string) => void
  onLLMChunk?: (chunk: string) => void
  onLLMEnd?: (
    output: string,
    thinking?: string,
    usage?: { input_tokens: number; output_tokens: number },
    timestamp?: string
  ) => void
  /** Called on tool events */
  onToolStart?: (name: string, input?: Record<string, unknown>, workflow?: string, eventId?: string, agentId?: string, timestamp?: string) => void
  onToolEnd?: (name: string, output?: string, eventId?: string, agentId?: string, timestamp?: string) => void
  /** Called on artifact updates */
  onTodoUpdate?: (todos: TodoItem[], workflow?: string, timestamp?: string) => void
  onCitationUpdate?: (url: string, content: string, isCited?: boolean, timestamp?: string, meta?: CitationEventMeta) => void
  onFileUpdate?: (filename: string, content: string, timestamp?: string) => void
  onOutputUpdate?: (content: string, outputCategory?: string, workflow?: string, timestamp?: string) => void
  /** Called on job heartbeat (confirms job is alive during long operations) */
  onHeartbeat?: (uptimeSeconds: number) => void
  /** Called when job completes successfully */
  onComplete?: () => void
  /** Called on errors */
  onError?: (error: Error) => void
  /** Called when connection is lost */
  onDisconnect?: () => void
}

// ============================================================
// Utilities
// ============================================================

const TEXTISH_FIELDS = [
  'text',
  'content',
  'thinking',
  'reasoning_content',
  'reasoning',
  'delta',
]

/**
 * Convert provider-specific streaming payloads into render-safe text.
 *
 * MiniMax M3's Anthropic-compatible stream can deliver content blocks such as:
 *   [{ type: "thinking", thinking: "...", index: 0 }]
 *   [{ type: "thinking", signature: "...", index: 0 }]
 *   [{ type: "tool_use", name: "read_file", input: {}, index: 1 }]
 *
 * React can only render primitives. This adapter keeps useful text/reasoning
 * blocks and drops protocol-only blocks that are already represented by tool
 * start/end events elsewhere in the UI.
 */
export const coerceSSEText = (value: unknown): string => {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)

  if (Array.isArray(value)) {
    return value.map(coerceSSEText).filter(Boolean).join('')
  }

  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    const blockType = typeof record.type === 'string' ? record.type : undefined

    if (blockType === 'tool_use' || blockType === 'input_json_delta') return ''
    if (blockType === 'thinking' && typeof record.signature === 'string' && !record.thinking) return ''

    for (const field of TEXTISH_FIELDS) {
      if (field in record) {
        const text = coerceSSEText(record[field])
        if (text) return text
      }
    }

    try {
      return JSON.stringify(record)
    } catch {
      return String(value)
    }
  }

  return String(value)
}

/**
 * Normalize tool input from the backend SSE stream.
 *
 * The backend's `_trim_tool_input` converts large dict inputs to Python `str()`
 * repr when they exceed 500 characters. For example:
 *   "{'query': 'search term', 'params': {'max_results': 10, ...}"
 *
 * This is NOT valid JSON (single quotes, Python booleans/None). The frontend
 * expects `Record<string, unknown>` for tool input, and downstream code calls
 * `JSON.stringify(input, null, 2)` to display it — which double-quotes a raw
 * string, producing confusing output like `"\"{'query': ...\""`
 *
 * This utility normalizes the input at the adapter boundary so all downstream
 * consumers receive a proper object:
 *   1. If input is already an object → pass through unchanged
 *   2. If input is a JSON string → parse it back to an object
 *   3. If input is a Python repr string → wrap in { _raw, _truncated } sentinel
 */
const normalizeToolInput = (input: unknown): Record<string, unknown> | undefined => {
  if (input == null) return undefined

  // Already a proper object — pass through
  if (typeof input === 'object' && !Array.isArray(input)) {
    return input as Record<string, unknown>
  }

  if (typeof input === 'string') {
    // Try JSON parse first (handles the case where backend used json.dumps)
    try {
      const parsed = JSON.parse(input)
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      // Not valid JSON — likely a Python repr string from str(dict)
    }

    // Python repr or other non-JSON string — wrap in sentinel object
    // so downstream JSON.stringify produces clean output
    return { _raw: input, _truncated: true }
  }

  // Unexpected primitive (number, boolean) — wrap for safety
  return { _raw: String(input) }
}

/**
 * Extract optional citation metadata from a citation_source/citation_use
 * artifact payload. The backend emits flat fields alongside type/content/url:
 * title, source_class, citation_key, and (when available) published/published_date.
 * All fields are optional — older events simply yield an empty meta object.
 */
export const extractCitationMeta = (raw: Record<string, unknown>): CitationEventMeta | undefined => {
  const pickString = (...keys: string[]): string | undefined => {
    for (const key of keys) {
      const value = raw[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
    }
    return undefined
  }

  const meta: CitationEventMeta = {
    title: pickString('title'),
    sourceClass: pickString('source_class', 'sourceClass'),
    citationKey: pickString('citation_key', 'citationKey'),
    publishedDate: pickString('published_date', 'published', 'publishedDate'),
  }

  return meta.title || meta.sourceClass || meta.citationKey || meta.publishedDate ? meta : undefined
}

// ============================================================
// Deep Research SSE Client
// ============================================================

export interface DeepResearchStreamOptions {
  /** Job ID to stream */
  jobId: string
  /** Event callbacks */
  callbacks: DeepResearchCallbacks
  /** Last event ID for reconnection (optional) */
  lastEventId?: string
  /** Auth token for authenticated requests */
  authToken?: string
}

export interface DeepResearchClient {
  /** Connect to the SSE stream */
  connect: () => void
  /** Disconnect from the SSE stream */
  disconnect: () => void
  /** Check if connected */
  isConnected: () => boolean
  /** Get the last received event ID (for reconnection) */
  getLastEventId: () => string | null
}

/**
 * Create a deep research SSE client
 *
 * Uses native EventSource for proper SSE protocol support.
 * Handles event types, reconnection, and routes events to callbacks.
 */
/** Max consecutive reconnection failures before surfacing an error to the caller */
const MAX_RECONNECT_ATTEMPTS = 5

export const createDeepResearchClient = (options: DeepResearchStreamOptions): DeepResearchClient => {
  const { jobId, callbacks, lastEventId, authToken } = options

  let eventSource: EventSource | null = null
  let lastReceivedEventId: string | null = lastEventId || null
  let isTerminated = false
  let reconnectAttempts = 0

  /**
   * Build the stream URL with optional last event ID for reconnection
   * Uses local API route in browser to avoid CORS issues
   */
  const buildStreamUrl = (): string => {
    // Use local API route in browser, direct backend URL on server
    const isBrowser = typeof window !== 'undefined'
    const baseUrl = isBrowser ? '' : apiConfig.baseUrl

    let url = `${baseUrl}/api/jobs/async/job/${jobId}/stream`

    if (lastReceivedEventId) {
      url += `/${lastReceivedEventId}`
    }

    // Add auth token as query param if provided (EventSource doesn't support headers)
    if (authToken) {
      url += `?token=${encodeURIComponent(authToken)}`
    }

    return url
  }

  /**
   * Parse SSE event data
   */
  const parseEventData = (data: string): unknown => {
    try {
      return JSON.parse(data)
    } catch {
      return data
    }
  }

  /**
   * Handle incoming SSE message
   *
   * SSE events have a nested structure where the actual payload is inside a 'data' property:
   * {
   *   "id": "uuid",
   *   "name": "workflow-name",
   *   "timestamp": "...",
   *   "data": { actual payload },
   *   "metadata": { workflow context }
   * }
   */
  const handleMessage = (event: MessageEvent, eventType: string) => {
    // Track last event ID for reconnection
    if (event.lastEventId) {
      lastReceivedEventId = event.lastEventId
    }

    const rawData = parseEventData(event.data)

    switch (eventType) {
      case 'stream.start': {
        const streamData = rawData as { job_id: string }
        callbacks.onStreamStart?.(streamData.job_id || jobId)
        break
      }

      case 'stream.mode': {
        const modeData = rawData as { mode?: string }
        if (modeData.mode) {
          callbacks.onStreamMode?.(modeData.mode)
        }
        break
      }

      case 'job.heartbeat': {
        const heartbeatData = rawData as { data?: { uptime_seconds?: number }; uptime_seconds?: number }
        const uptimeSeconds = heartbeatData.data?.uptime_seconds ?? heartbeatData.uptime_seconds ?? 0
        callbacks.onHeartbeat?.(uptimeSeconds)
        break
      }

      case 'job.status': {
        // job.status wraps status in data property
        const statusWrapper = rawData as { data?: { status: DeepResearchJobStatus; error?: string }; status?: DeepResearchJobStatus; error?: string }
        const statusData = statusWrapper.data || statusWrapper
        callbacks.onJobStatus?.(statusData.status!, statusData.error)

        // Check for terminal states — close EventSource immediately to prevent
        // auto-reconnection loops (EventSource reconnects on its own if left open)
        if (statusData.status === 'success') {
          isTerminated = true
          eventSource?.close()
          callbacks.onComplete?.()
        } else if (statusData.status === 'failure' || statusData.status === 'interrupted') {
          isTerminated = true
          eventSource?.close()
          // Terminal job failures are already handled by onJobStatus. Reserve
          // onError for transport/parser failures so the UI does not report a
          // clean job.status event as an SSE exception.
        }
        break
      }

      case 'workflow.start': {
        // workflow events have nested structure: { id, name, timestamp, data: { input }, metadata: { agent_id } }
        const workflowData = rawData as { id?: string; name: string; timestamp?: string; data?: { input?: unknown }; metadata?: { agent_id?: string } }
        callbacks.onWorkflowStart?.(workflowData.name, coerceSSEText(workflowData.data?.input), workflowData.id, workflowData.metadata?.agent_id, workflowData.timestamp)
        break
      }

      case 'workflow.end': {
        const workflowData = rawData as { id?: string; name: string; timestamp?: string; data?: { output?: unknown }; metadata?: { agent_id?: string } }
        callbacks.onWorkflowEnd?.(workflowData.name, coerceSSEText(workflowData.data?.output), workflowData.id, workflowData.metadata?.agent_id, workflowData.timestamp)
        break
      }

      case 'llm.start': {
        const llmData = rawData as { name: string; timestamp?: string; metadata?: { workflow?: string } }
        callbacks.onLLMStart?.(llmData.name, llmData.metadata?.workflow, llmData.timestamp)
        break
      }

      case 'llm.chunk': {
        const chunkData = rawData as { chunk?: unknown; data?: { chunk?: unknown } }
        const chunk = coerceSSEText(chunkData.chunk ?? chunkData.data?.chunk)
        if (chunk) callbacks.onLLMChunk?.(chunk)
        break
      }

      case 'llm.end': {
        // llm.end has nested structure: { id, name, timestamp, metadata: { thinking, usage }, data: { output } }
        const endData = rawData as {
          timestamp?: string
          data?: { output?: unknown }
          metadata?: {
            thinking?: unknown
            usage?: { input_tokens?: number; output_tokens?: number; prompt_tokens?: number; completion_tokens?: number }
          }
        }
        const output = coerceSSEText(endData.data?.output)
        const thinking = coerceSSEText(endData.metadata?.thinking)
        // Handle both naming conventions for usage (input_tokens/output_tokens or prompt_tokens/completion_tokens)
        const rawUsage = endData.metadata?.usage
        const usage = rawUsage
          ? {
              input_tokens: rawUsage.input_tokens ?? rawUsage.prompt_tokens ?? 0,
              output_tokens: rawUsage.output_tokens ?? rawUsage.completion_tokens ?? 0,
            }
          : undefined
        callbacks.onLLMEnd?.(output, thinking || undefined, usage, endData.timestamp)
        break
      }

      case 'tool.start': {
        // tool events have nested structure: { id, name, timestamp, data: { input }, metadata: { workflow, agent_id } }
        // Note: data.input may be a Python repr string when the backend trims large inputs via str()
        const toolData = rawData as {
          id?: string
          name: string
          timestamp?: string
          data?: { input?: unknown }
          metadata?: { workflow?: string; agent_id?: string }
        }
        const normalizedInput = normalizeToolInput(toolData.data?.input)
        callbacks.onToolStart?.(toolData.name, normalizedInput, toolData.metadata?.workflow, toolData.id, toolData.metadata?.agent_id, toolData.timestamp)
        break
      }

      case 'tool.end': {
        const toolData = rawData as { id?: string; name: string; timestamp?: string; data?: { output?: unknown }; metadata?: { agent_id?: string } }
        callbacks.onToolEnd?.(toolData.name, coerceSSEText(toolData.data?.output), toolData.id, toolData.metadata?.agent_id, toolData.timestamp)
        break
      }

      case 'artifact.update': {
        // artifact.update has nested structure: { id, timestamp, data: { type, content, url?, output_category? }, metadata?: { workflow } }
        const artifactWrapper = rawData as {
          timestamp?: string
          data?: { type: ArtifactType; content: unknown; url?: string; output_category?: string }
          type?: ArtifactType
          content?: unknown
          url?: string
          output_category?: string
          metadata?: { workflow?: string }
        }
        // Handle both nested (data.type) and flat (type) structures
        const artifactData = artifactWrapper.data || artifactWrapper
        const artifactWorkflow = artifactWrapper.metadata?.workflow
        const artifactTimestamp = artifactWrapper.timestamp

        switch (artifactData.type) {
          case 'todo':
            callbacks.onTodoUpdate?.(artifactData.content as TodoItem[], artifactWorkflow, artifactTimestamp)
            break
          case 'citation_source':
            // citation_source = "Referenced" sources (discovered during search)
            callbacks.onCitationUpdate?.(
              artifactData.url || '',
              coerceSSEText(artifactData.content),
              false,
              artifactTimestamp,
              extractCitationMeta(artifactData as Record<string, unknown>)
            )
            break
          case 'citation_use':
            // citation_use = "Cited" sources (actually used in the report)
            callbacks.onCitationUpdate?.(
              artifactData.url || '',
              coerceSSEText(artifactData.content),
              true,
              artifactTimestamp,
              extractCitationMeta(artifactData as Record<string, unknown>)
            )
            break
          case 'file': {
            // file artifacts are written during research. Preserve the virtual
            // path so Claude Code run-folder files can be surfaced in chat.
            const raw = artifactData as Record<string, unknown>
            const filePath = (raw.file_path || raw.path || artifactData.url || 'unknown') as string
            callbacks.onFileUpdate?.(filePath, coerceSSEText(artifactData.content), artifactTimestamp)
            break
          }
          case 'output':
            callbacks.onOutputUpdate?.(coerceSSEText(artifactData.content), artifactData.output_category, artifactWorkflow, artifactTimestamp)
            break
          default:
            if (process.env.NODE_ENV === 'development') {
              console.warn(`[SSE:artifact.update] Unknown artifact type: ${artifactData.type}`)
            }
        }
        break
      }

      default:
        // Unknown event type - log in dev
        if (process.env.NODE_ENV === 'development') {
          console.warn(`[SSE] Unknown event type: ${eventType}`, rawData)
        }
        break
    }
  }

  /**
   * Connect to the SSE stream
   */
  const connect = () => {
    if (eventSource) {
      return // Already connected
    }

    isTerminated = false
    const url = buildStreamUrl()
    eventSource = new EventSource(url)

    // Handle connection open
    eventSource.onopen = () => {
      // Connection established (or re-established after reconnect) — reset counter
      reconnectAttempts = 0
    }

    // Handle generic messages (fallback)
    eventSource.onmessage = (event) => {
      handleMessage(event, 'message')
    }

    // Register handlers for each known event type
    const eventTypes: DeepResearchEventType[] = [
      'stream.start',
      'stream.mode',
      'job.status',
      'job.heartbeat',
      'workflow.start',
      'workflow.end',
      'llm.start',
      'llm.chunk',
      'llm.end',
      'tool.start',
      'tool.end',
      'artifact.update',
    ]

    eventTypes.forEach((eventType) => {
      eventSource?.addEventListener(eventType, (event) => {
        handleMessage(event as MessageEvent, eventType)
      })
    })

    // Handle errors
    // EventSource fires onerror on *any* connection issue, then auto-reconnects
    // (readyState transitions to CONNECTING). This is normal SSE behaviour, not a
    // fatal error. We only surface an error to the caller after the reconnection
    // retry threshold is exceeded or the browser has given up (readyState CLOSED).
    eventSource.onerror = () => {
      if (isTerminated) {
        // Expected disconnection after terminal state — close to stop auto-reconnect
        eventSource?.close()
        eventSource = null
        return
      }

      if (eventSource?.readyState === EventSource.CLOSED) {
        // Browser gave up reconnecting — treat as a real disconnect
        callbacks.onDisconnect?.()
        eventSource = null
      } else if (eventSource?.readyState === EventSource.CONNECTING) {
        // EventSource is auto-reconnecting — this is expected behaviour.
        // Only escalate to an error after repeated consecutive failures.
        reconnectAttempts++
        if (reconnectAttempts <= MAX_RECONNECT_ATTEMPTS) {
          if (process.env.NODE_ENV === 'development') {
            console.warn(`[SSE] Reconnecting (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})…`)
          }
          return // let EventSource retry on its own
        }
        // Too many consecutive reconnection failures — give up
        eventSource?.close()
        eventSource = null
        callbacks.onError?.(
          new Error(`SSE connection failed after ${reconnectAttempts} reconnection attempts`)
        )
      } else {
        // Unexpected state (e.g. OPEN) — should not happen per SSE spec, but
        // handle defensively so the caller is never left in an unknown state.
        const readyState = eventSource?.readyState
        eventSource?.close()
        eventSource = null
        callbacks.onError?.(
          new Error(`SSE connection error in unexpected readyState: ${readyState ?? 'unknown'}`)
        )
      }
    }
  }

  /**
   * Disconnect from the SSE stream
   */
  const disconnect = () => {
    if (eventSource) {
      isTerminated = true
      eventSource.close()
      eventSource = null
    }
  }

  /**
   * Check if connected
   */
  const isConnected = (): boolean => {
    return eventSource !== null && eventSource.readyState === EventSource.OPEN
  }

  /**
   * Get the last received event ID for reconnection
   */
  const getLastEventId = (): string | null => {
    return lastReceivedEventId
  }

  return {
    connect,
    disconnect,
    isConnected,
    getLastEventId,
  }
}

// ============================================================
// REST API Functions (for non-streaming operations)
// ============================================================

/**
 * Get the base URL for deep research API
 * Uses local API route in browser to avoid CORS issues
 */
const getDeepResearchBaseUrl = (): string => {
  const isBrowser = typeof window !== 'undefined'
  return isBrowser ? '/api/jobs/async' : `${apiConfig.baseUrl}/v1/jobs/async`
}

/** Get job status */
export const getJobStatus = async (
  jobId: string,
  authToken?: string
): Promise<DeepResearchJobStatusResponse> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    throw new Error(`Failed to get job status: ${response.status}`)
  }

  return response.json()
}

/** List persisted async research jobs visible to the caller. */
export const listJobs = async (
  authToken?: string,
  limit = 100
): Promise<JobHistoryResponse> => {
  const url = `${getDeepResearchBaseUrl()}/jobs?limit=${encodeURIComponent(String(limit))}`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    throw new Error(`Failed to list jobs: ${response.status}`)
  }

  return response.json()
}

/** Submit a deep research job through the same-origin proxy. */
export const submitDeepResearchJob = async (
  payload: DeepResearchJobSubmitRequest
): Promise<DeepResearchJobStatusResponse> => {
  const response = await fetch('/api/jobs/async/submit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const errorText = await response.text().catch(() => '')
    throw new Error(`Failed to submit deep research job: ${response.status}${errorText ? ` ${errorText}` : ''}`)
  }

  return response.json()
}

/** Get job report */
export const getJobReport = async (
  jobId: string,
  authToken?: string
): Promise<DeepResearchJobReportResponse> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/report`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    throw new Error(`Failed to get job report: ${response.status}`)
  }

  return response.json()
}

/** Cancel a running job */
export const cancelJob = async (
  jobId: string,
  authToken?: string
): Promise<{ cancelled?: boolean; job_id?: string; status?: DeepResearchJobStatus; task_cancelled?: boolean }> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/cancel`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
  })

  if (!response.ok) {
    throw new Error(`Failed to cancel job: ${response.status}`)
  }

  return response.json()
}

/** Resume a failed/interrupted job using persisted research artifacts. */
export const resumeJob = async (
  jobId: string,
  authToken?: string
): Promise<DeepResearchJobStatusResponse> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/resume`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
  })

  if (!response.ok) {
    throw new Error(`Failed to resume job: ${response.status}`)
  }

  return response.json()
}

/** Job state/artifacts response */
export interface JobStateResponse {
  job_id: string
  has_state: boolean
  state: Record<string, unknown> | null
  artifacts: {
    tools: Array<{
      name: string
      input?: Record<string, unknown>
      output?: string
      timestamp?: string
    }>
    outputs: Array<{
      type: string
      content: string
      timestamp?: string
    }>
  } | null
}

/** Get job state/artifacts (for catching up on missed events) */
export const getJobState = async (
  jobId: string,
  authToken?: string
): Promise<JobStateResponse> => {
  const url = `${getDeepResearchBaseUrl()}/job/${jobId}/state`
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`
  }

  const response = await fetch(url, { headers })

  if (!response.ok) {
    throw new Error(`Failed to get job state: ${response.status}`)
  }

  return response.json()
}

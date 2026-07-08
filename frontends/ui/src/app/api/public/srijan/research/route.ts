// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHmac, randomUUID } from 'node:crypto'
import { NextResponse } from 'next/server'

export const runtime = 'nodejs'

type ResearchDepth = 'shallow' | 'medium' | 'deeper' | 'deep'
type PublicView = 'status' | 'report' | 'state' | 'stream'

interface PublicResearchRequest {
  query?: unknown
  input?: unknown
  research_depth?: unknown
  data_sources?: unknown
  expiry_seconds?: unknown
  job_id?: unknown
}

const PUBLIC_OWNER = 'srijan'
const PUBLIC_JOB_PREFIX = 'public_'
const DEFAULT_RESEARCH_DEPTH: ResearchDepth = 'deeper'
const MAX_QUERY_LENGTH = 12000
const MIN_EXPIRY_SECONDS = 600
const MAX_EXPIRY_SECONDS = 604800
const TOKEN_PREFIX = 'aiq_local.'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
  'Cache-Control': 'no-store',
}

const jsonHeaders = {
  ...corsHeaders,
  'Content-Type': 'application/json',
}

const getBackendUrl = (): string => {
  const url = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000'
  return url.replace(/\/$/, '')
}

const publicOrigin = (req: Request): string => {
  const url = new URL(req.url)
  const forwardedHost = req.headers.get('x-forwarded-host')?.split(',')[0]?.trim()
  const host = forwardedHost || req.headers.get('host') || url.host
  const forwardedProto = req.headers.get('x-forwarded-proto')?.split(',')[0]?.trim()
  const proto = forwardedProto || (host.includes('localhost') || host.startsWith('127.') ? 'http' : url.protocol.replace(':', ''))
  return `${proto}://${host}`
}

const jsonResponse = (body: unknown, init?: ResponseInit): NextResponse =>
  new NextResponse(JSON.stringify(body), {
    ...init,
    headers: {
      ...jsonHeaders,
      ...(init?.headers ?? {}),
    },
  })

const errorResponse = (status: number, code: string, message: string): NextResponse =>
  jsonResponse({ error: { code, message } }, { status })

const base64Url = (input: string | Buffer): string =>
  Buffer.from(input).toString('base64url')

const sortedJson = (value: Record<string, unknown>): string => {
  const sorted: Record<string, unknown> = {}
  for (const key of Object.keys(value).sort()) sorted[key] = value[key]
  return JSON.stringify(sorted)
}

const requireLocalAuthSecret = (): string => {
  const secret = process.env.AIQ_AUTH_TOKEN_SECRET || process.env.NEXTAUTH_SECRET
  if (!secret) {
    throw new Error('AIQ_AUTH_TOKEN_SECRET or NEXTAUTH_SECRET must be configured')
  }
  return secret
}

const issueSrijanToken = (): string => {
  const nowSeconds = Math.floor(Date.now() / 1000)
  const expiresSeconds = nowSeconds + 15 * 60
  const payload = {
    email: `${PUBLIC_OWNER}@local`,
    exp: expiresSeconds,
    iat: nowSeconds,
    jti: randomUUID(),
    name: 'Srijan',
    remember_me: false,
    role: 'admin',
    sub: PUBLIC_OWNER,
    typ: 'aiq_local_user',
    type: 'local_user',
    username: PUBLIC_OWNER,
  }
  const payloadB64 = base64Url(sortedJson(payload))
  const signature = createHmac('sha256', requireLocalAuthSecret()).update(payloadB64).digest('base64url')
  return `${TOKEN_PREFIX}${payloadB64}.${signature}`
}

const backendAuthHeaders = (): Record<string, string> => {
  const token = issueSrijanToken()
  return {
    Authorization: `Bearer ${token}`,
    Cookie: `idToken=${token}`,
    'X-AIQ-Access-Channel': 'public-srijan-research',
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const parseBody = async (req: Request): Promise<PublicResearchRequest> => {
  try {
    const body = await req.json()
    return isRecord(body) ? body : {}
  } catch {
    return {}
  }
}

const normalizeQuery = (body: PublicResearchRequest): string | null => {
  const raw = body.query ?? body.input
  if (typeof raw !== 'string') return null
  const query = raw.trim()
  if (!query || query.length > MAX_QUERY_LENGTH) return null
  return query
}

const normalizeResearchDepth = (value: unknown): ResearchDepth | null => {
  if (value === undefined || value === null || value === '') return DEFAULT_RESEARCH_DEPTH
  if (value === 'shallow' || value === 'medium' || value === 'deeper' || value === 'deep') return value
  return null
}

const normalizeDataSources = (value: unknown): string[] | null | undefined => {
  if (value === undefined || value === null) return undefined
  if (!Array.isArray(value)) return null

  const sources = value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean)

  if (sources.length > 10) return null
  if (sources.some((source) => !/^[a-zA-Z0-9_-]{1,64}$/.test(source))) return null
  return Array.from(new Set(sources))
}

const normalizeExpirySeconds = (value: unknown): number | null => {
  if (value === undefined || value === null || value === '') return 86400
  if (typeof value !== 'number' || !Number.isInteger(value)) return null
  if (value < MIN_EXPIRY_SECONDS || value > MAX_EXPIRY_SECONDS) return null
  return value
}

const normalizeCustomJobId = (value: unknown): string | null | undefined => {
  if (value === undefined || value === null || value === '') return undefined
  if (typeof value !== 'string') return null
  const jobId = value.trim()
  if (!isPublicJobId(jobId)) return null
  return jobId
}

const isPublicJobId = (jobId: string): boolean =>
  jobId.startsWith(PUBLIC_JOB_PREFIX) && /^[a-zA-Z0-9_-]{8,64}$/.test(jobId)

const buildPublicLinks = (req: Request, jobId: string) => {
  const base = `${publicOrigin(req)}/api/public/srijan/research`
  return {
    status_url: `${base}?job_id=${encodeURIComponent(jobId)}`,
    report_url: `${base}?job_id=${encodeURIComponent(jobId)}&view=report`,
    state_url: `${base}?job_id=${encodeURIComponent(jobId)}&view=state`,
    stream_url: `${base}?job_id=${encodeURIComponent(jobId)}&view=stream`,
  }
}

const submitPayload = (
  jobId: string,
  query: string,
  researchDepth: ResearchDepth,
  dataSources: string[] | undefined,
  expirySeconds: number
) => ({
  agent_type: 'deep_researcher',
  input: query,
  research_depth: researchDepth,
  job_id: jobId,
  expiry_seconds: expirySeconds,
  ...(dataSources !== undefined ? { data_sources: dataSources } : {}),
})

const requestDocs = (req: Request) => {
  const base = `${publicOrigin(req)}/api/public/srijan/research`
  return {
    endpoint: base,
    owner: PUBLIC_OWNER,
    start: {
      method: 'POST',
      body: {
        query: 'Research question or topic',
        research_depth: DEFAULT_RESEARCH_DEPTH,
        data_sources: ['web_search'],
        expiry_seconds: 86400,
      },
    },
    poll: `${base}?job_id=public_JOB_ID`,
    report: `${base}?job_id=public_JOB_ID&view=report`,
    stream: `${base}?job_id=public_JOB_ID&view=stream`,
    research_depth_values: ['shallow', 'medium', 'deeper', 'deep'],
  }
}

export async function OPTIONS(): Promise<Response> {
  return new NextResponse(null, { status: 204, headers: corsHeaders })
}

export async function POST(req: Request): Promise<Response> {
  const body = await parseBody(req)
  const query = normalizeQuery(body)
  if (!query) {
    return errorResponse(400, 'INVALID_QUERY', `Provide query or input as a non-empty string up to ${MAX_QUERY_LENGTH} characters.`)
  }

  const researchDepth = normalizeResearchDepth(body.research_depth)
  if (!researchDepth) {
    return errorResponse(400, 'INVALID_RESEARCH_DEPTH', 'research_depth must be shallow, medium, deeper, or deep.')
  }

  const dataSources = normalizeDataSources(body.data_sources)
  if (dataSources === null) {
    return errorResponse(400, 'INVALID_DATA_SOURCES', 'data_sources must be an array of source IDs.')
  }

  const expirySeconds = normalizeExpirySeconds(body.expiry_seconds)
  if (expirySeconds === null) {
    return errorResponse(400, 'INVALID_EXPIRY_SECONDS', 'expiry_seconds must be an integer from 600 to 604800.')
  }

  const customJobId = normalizeCustomJobId(body.job_id)
  if (customJobId === null) {
    return errorResponse(400, 'INVALID_JOB_ID', 'job_id must start with public_ and contain only letters, numbers, underscores, or hyphens.')
  }

  const jobId = customJobId ?? `${PUBLIC_JOB_PREFIX}${randomUUID()}`
  const backendResponse = await fetch(`${getBackendUrl()}/v1/jobs/async/submit`, {
    method: 'POST',
    headers: {
      ...backendAuthHeaders(),
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(submitPayload(jobId, query, researchDepth, dataSources, expirySeconds)),
  })

  if (!backendResponse.ok) {
    const message = await backendResponse.text()
    return errorResponse(backendResponse.status, 'BACKEND_ERROR', message || `Backend returned ${backendResponse.status}`)
  }

  const data = await backendResponse.json()
  return jsonResponse(
    {
      ...data,
      owner: PUBLIC_OWNER,
      public: true,
      ...buildPublicLinks(req, jobId),
    },
    { status: 202 }
  )
}

export async function GET(req: Request): Promise<Response> {
  const url = new URL(req.url)
  const jobId = url.searchParams.get('job_id')?.trim()
  if (!jobId) return jsonResponse(requestDocs(req))
  if (!isPublicJobId(jobId)) {
    return errorResponse(400, 'INVALID_JOB_ID', 'Only public_ job IDs created through this endpoint are readable here.')
  }

  const view = (url.searchParams.get('view') || 'status') as PublicView
  if (!['status', 'report', 'state', 'stream'].includes(view)) {
    return errorResponse(400, 'INVALID_VIEW', 'view must be status, report, state, or stream.')
  }

  const backendPath =
    view === 'status'
      ? `/v1/jobs/async/job/${encodeURIComponent(jobId)}`
      : `/v1/jobs/async/job/${encodeURIComponent(jobId)}/${view}`
  const backendUrl = new URL(`${getBackendUrl()}${backendPath}`)

  if (view === 'report') {
    const format = url.searchParams.get('format')
    if (format) backendUrl.searchParams.set('format', format)
  }

  const backendResponse = await fetch(backendUrl, {
    method: 'GET',
    headers: {
      ...backendAuthHeaders(),
      Accept: view === 'stream' ? 'text/event-stream' : 'application/json',
    },
    ...(view === 'stream' ? { signal: req.signal } : {}),
  })

  if (!backendResponse.ok) {
    const message = await backendResponse.text()
    return errorResponse(backendResponse.status, 'BACKEND_ERROR', message || `Backend returned ${backendResponse.status}`)
  }

  if (view === 'stream') {
    return new NextResponse(backendResponse.body, {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'text/event-stream',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    })
  }

  const contentType = backendResponse.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    return new NextResponse(backendResponse.body, {
      status: backendResponse.status,
      headers: {
        ...corsHeaders,
        'Content-Type': contentType || 'application/octet-stream',
      },
    })
  }

  const data = await backendResponse.json()
  return jsonResponse({
    ...data,
    owner: PUBLIC_OWNER,
    public: true,
    ...buildPublicLinks(req, jobId),
  })
}

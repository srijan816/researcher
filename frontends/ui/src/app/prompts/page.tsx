// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Prompt Library Page
 *
 * Browser-readable view of the prompt templates served by the backend.
 */

import { type ReactNode } from 'react'
import { cookies } from 'next/headers'
import { isAuthRequired } from '@/adapters/auth/config'

interface PromptTemplate {
  id: string
  agent: string
  role: string
  description: string
  path: string
  template: string
  variables: string[]
  character_count: number
}

interface PromptListResponse {
  count: number
  prompts: PromptTemplate[]
  missing: Array<{ id: string; path: string; error: string }>
}

interface PromptFetchResult {
  data: PromptListResponse | null
  error: string | null
}

const getBackendUrl = (): string => {
  const url = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000'
  return url.replace(/\/$/, '')
}

const getAuthHeaders = async (): Promise<Record<string, string>> => {
  if (!isAuthRequired()) return {}
  const cookieStore = await cookies()
  const idToken = cookieStore.get('idToken')?.value
  return idToken ? { Cookie: `idToken=${idToken}` } : {}
}

const fetchPrompts = async (): Promise<PromptFetchResult> => {
  const backendBase = getBackendUrl()
  const authHeaders = await getAuthHeaders()
  const candidates = [`${backendBase}/v1/prompts`, `${backendBase}/prompts`]
  const errors: string[] = []

  for (const url of candidates) {
    try {
      const response = await fetch(url, {
        cache: 'no-store',
        headers: {
          ...authHeaders,
          Accept: 'application/json',
        },
      })

      if (!response.ok) {
        const body = await response.text()
        errors.push(`${url} returned ${response.status}: ${body}`)
        continue
      }

      return { data: await response.json() as PromptListResponse, error: null }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      errors.push(`${url} failed: ${message}`)
    }
  }

  return {
    data: null,
    error: errors.join('\n'),
  }
}

const formatNumber = (value: number): string => new Intl.NumberFormat('en-US').format(value)

const PromptPage = async (): Promise<ReactNode> => {
  const result = await fetchPrompts()

  return (
    <main className="min-h-screen bg-surface-base text-primary">
      <section className="border-b border-base bg-surface-raised">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-3 px-6 py-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-normal">Prompt Library</h1>
              <p className="mt-2 max-w-3xl text-sm text-secondary">
                Raw prompt templates used by chat routing, clarification, shallow research, and deep research.
              </p>
            </div>
            {result.data ? (
              <div className="text-sm text-secondary">
                {formatNumber(result.data.count)} prompts
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-7xl px-6 py-6">
        {result.error ? (
          <div className="rounded-md border border-base bg-surface-raised p-4">
            <h2 className="text-base font-semibold">Prompts are not available from the backend yet</h2>
            <p className="mt-2 text-sm text-secondary">
              Rebuild and restart the app2 backend with the latest code, then refresh this page.
            </p>
            <pre className="mt-4 max-h-80 overflow-auto rounded-md bg-surface-sunken p-3 text-xs text-secondary">
              {result.error}
            </pre>
          </div>
        ) : null}

        {result.data?.missing.length ? (
          <div className="mb-6 rounded-md border border-base bg-surface-raised p-4">
            <h2 className="text-base font-semibold">Missing Prompt Files</h2>
            <ul className="mt-3 space-y-2 text-sm text-secondary">
              {result.data.missing.map((missing) => (
                <li key={missing.id}>
                  <span className="font-medium text-primary">{missing.id}</span>: {missing.path} ({missing.error})
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="space-y-5">
          {result.data?.prompts.map((prompt) => (
            <article key={prompt.id} className="rounded-md border border-base bg-surface-raised">
              <header className="border-b border-base px-4 py-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h2 className="text-base font-semibold">{prompt.id}</h2>
                    <p className="mt-1 text-sm text-secondary">{prompt.description}</p>
                  </div>
                  <div className="text-right text-xs text-secondary">
                    <div>{prompt.agent}</div>
                    <div>{prompt.role}</div>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-secondary">
                  <span className="rounded border border-base px-2 py-1">{prompt.path}</span>
                  <span className="rounded border border-base px-2 py-1">
                    {formatNumber(prompt.character_count)} chars
                  </span>
                  {prompt.variables.map((variable) => (
                    <span key={variable} className="rounded border border-base px-2 py-1">
                      {variable}
                    </span>
                  ))}
                </div>
              </header>
              <pre className="max-h-[36rem] overflow-auto whitespace-pre-wrap break-words p-4 text-xs leading-5 text-primary">
                {prompt.template}
              </pre>
            </article>
          ))}
        </div>
      </section>
    </main>
  )
}

export default PromptPage

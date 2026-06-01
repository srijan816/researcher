// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Data Sources API Route
 *
 * Proxies available data sources from the backend registry.
 */

import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'

const getBackendUrl = (): string => {
  const url = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000'
  return url.replace(/\/$/, '')
}

const getOptionalAuthHeaders = async (req: NextRequest): Promise<Record<string, string>> => {
  const authToken = req.headers.get('Authorization')
  const cookieStore = await cookies()
  const idToken = cookieStore.get('idToken')?.value

  return {
    ...(authToken ? { Authorization: authToken } : {}),
    ...(idToken ? { Cookie: `idToken=${idToken}` } : {}),
  }
}

export async function GET(req: NextRequest): Promise<Response> {
  try {
    const authHeaders = await getOptionalAuthHeaders(req)
    const response = await fetch(`${getBackendUrl()}/v1/data_sources`, {
      method: 'GET',
      headers: {
        ...authHeaders,
        Accept: 'application/json',
      },
      cache: 'no-store',
    })

    if (!response.ok) {
      const errorText = await response.text()
      return new NextResponse(
        JSON.stringify({
          error: { code: 'BACKEND_ERROR', message: `Backend returned ${response.status}: ${errorText}` },
        }),
        { status: response.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error'
    return new NextResponse(
      JSON.stringify({ error: { code: 'PROXY_ERROR', message } }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}

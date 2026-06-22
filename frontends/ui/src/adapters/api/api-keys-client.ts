// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API key management client.
 */

import { authenticatedFetch } from './authenticated-fetch'

export interface APIKeyMetadata {
  id: string
  name: string
  prefix: string
  created_at: string
  last_used_at: string | null
}

export interface APIKeyCreateResponse extends APIKeyMetadata {
  key: string
}

export interface APIKeyListResponse {
  api_keys: APIKeyMetadata[]
}

const baseUrl = '/api/v1/api-keys'

const parseError = async (response: Response): Promise<Error> => {
  const text = await response.text()
  return new Error(`API key request failed (${response.status}): ${text || response.statusText}`)
}

export const listAPIKeys = async (): Promise<APIKeyListResponse> => {
  const response = await authenticatedFetch(baseUrl)
  if (!response.ok) {
    throw await parseError(response)
  }
  return response.json()
}

export const createAPIKey = async (name: string): Promise<APIKeyCreateResponse> => {
  const response = await authenticatedFetch(baseUrl, {
    method: 'POST',
    body: JSON.stringify({ name }),
  })
  if (!response.ok) {
    throw await parseError(response)
  }
  return response.json()
}

export const revokeAPIKey = async (id: string): Promise<void> => {
  const response = await authenticatedFetch(`${baseUrl}/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
  if (!response.ok) {
    throw await parseError(response)
  }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Conversation } from '@/features/chat/types'

export interface ConversationSnapshotsResponse {
  conversations: Conversation[]
}

const CONVERSATIONS_BASE = '/api/v1/conversations'

const jsonHeaders = { 'Content-Type': 'application/json' }

const requireOk = async (response: Response, action: string): Promise<void> => {
  if (response.ok) return
  const text = await response.text().catch(() => '')
  throw new Error(`${action} failed: ${response.status}${text ? ` ${text}` : ''}`)
}

export const listConversationSnapshots = async (): Promise<ConversationSnapshotsResponse> => {
  const response = await fetch(CONVERSATIONS_BASE, {
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  })
  await requireOk(response, 'List conversations')
  return response.json()
}

export const syncConversationSnapshots = async (
  conversations: Conversation[]
): Promise<ConversationSnapshotsResponse> => {
  const response = await fetch(`${CONVERSATIONS_BASE}/sync`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ conversations }),
  })
  await requireOk(response, 'Sync conversations')
  return response.json()
}

export const deleteConversationSnapshot = async (conversationId: string): Promise<void> => {
  const response = await fetch(`${CONVERSATIONS_BASE}/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
  })
  await requireOk(response, 'Delete conversation')
}

export const deleteAllConversationSnapshots = async (): Promise<void> => {
  const response = await fetch(CONVERSATIONS_BASE, {
    method: 'DELETE',
  })
  await requireOk(response, 'Delete conversations')
}

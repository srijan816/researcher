// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import CredentialsProvider from 'next-auth/providers/credentials'
import type { TokenRefreshResult } from './types'

interface LocalAuthUser {
  username: string
  display_name: string
  email?: string | null
  role: string
  must_change_password: boolean
}

interface LocalAuthResponse {
  token: string
  expires_in: number
  expires_at: string
  user: LocalAuthUser
}

const getBackendUrl = (): string => {
  const url = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000'
  return url.replace(/\/$/, '')
}

const expiresAtSeconds = (value: string): number => {
  const parsed = new Date(value).getTime()
  if (Number.isNaN(parsed)) {
    return Math.floor(Date.now() / 1000) + 24 * 60 * 60
  }
  return Math.floor(parsed / 1000)
}

const postAuth = async (
  path: string,
  body?: Record<string, unknown>,
  token?: string
): Promise<LocalAuthResponse> => {
  const response = await fetch(`${getBackendUrl()}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  })

  if (!response.ok) {
    throw new Error(`Local auth request failed: ${response.status}`)
  }

  return response.json()
}

export const LocalUsersProvider = CredentialsProvider({
  id: 'local-users',
  name: 'Deep Research Users',
  credentials: {
    username: { label: 'Username', type: 'text' },
    password: { label: 'Password', type: 'password' },
    remember_me: { label: 'Remember me', type: 'checkbox' },
  },
  async authorize(credentials) {
    const username = credentials?.username?.trim()
    const password = credentials?.password
    if (!username || !password) return null

    const rememberMeValue = String(credentials?.remember_me ?? '').toLowerCase()
    const remember_me = ['true', 'on', '1', 'yes'].includes(rememberMeValue)

    const data = await postAuth('/v1/auth/login', { username, password, remember_me })
    return {
      id: data.user.username,
      name: data.user.display_name,
      email: data.user.email ?? `${data.user.username}@local`,
      image: null,
      idToken: data.token,
      refreshToken: data.token,
      idTokenExpiresAt: expiresAtSeconds(data.expires_at),
      role: data.user.role,
      mustChangePassword: data.user.must_change_password,
    }
  },
})

export const refreshLocalUserToken = async (refreshToken: string): Promise<TokenRefreshResult> => {
  const data = await postAuth('/v1/auth/refresh', undefined, refreshToken)
  return {
    access_token: data.token,
    id_token: data.token,
    expires_in: data.expires_in,
    refresh_token: data.token,
  }
}

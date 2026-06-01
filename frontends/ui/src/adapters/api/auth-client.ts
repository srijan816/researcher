// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface ChangePasswordRequest {
  current_password: string
  new_password: string
}

export const changePassword = async (request: ChangePasswordRequest): Promise<void> => {
  const response = await fetch('/api/v1/auth/change-password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(text || `Password change failed: ${response.status}`)
  }
}

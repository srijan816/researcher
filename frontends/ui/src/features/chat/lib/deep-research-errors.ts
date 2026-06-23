// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const isUnavailableDeepResearchJobError = (error: unknown): boolean => {
  if (!(error instanceof Error)) return false
  return /(?:404|410)|expired|deleted|not found/i.test(error.message)
}

/** Detect HTTP 401/403 from the wrapped "Failed to get job status: <code>"
 *  error message. Used to dispatch a re-auth error card instead of the
 *  generic "research data unavailable" message. */
export const isAuthRequiredDeepResearchError = (error: unknown): boolean => {
  if (!(error instanceof Error)) return false
  return /(?:401|403)/.test(error.message)
}

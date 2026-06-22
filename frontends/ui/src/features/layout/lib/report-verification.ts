// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Verification report utilities (pure functions)
 *
 * The adversarial claim verifier writes a virtual file
 * `/shared/verification_report.json` shaped like:
 *
 *   {
 *     "summary": {
 *       "supported": 12,
 *       "partially_supported": 3,
 *       "contradicted": 1,
 *       "not_addressed": 2
 *     },
 *     "claims": [...]
 *   }
 *
 * The file arrives in the UI through ordinary file artifact events
 * (store: deepResearchFiles). When the file is absent or malformed the
 * parser returns null and the UI renders nothing.
 */

/** Virtual file path suffix for the claim verification report */
export const VERIFICATION_REPORT_FILENAME = 'verification_report.json'

export interface VerificationSummary {
  supported: number
  partiallySupported: number
  contradicted: number
  notAddressed: number
  total: number
}

const toCount = (value: unknown): number => {
  const num = typeof value === 'string' ? Number(value) : value
  return typeof num === 'number' && Number.isFinite(num) && num >= 0 ? Math.floor(num) : 0
}

/**
 * Parse the verification report JSON content into a summary.
 * Returns null for missing/malformed content or an all-zero summary.
 */
export const parseVerificationSummary = (content: string | undefined | null): VerificationSummary | null => {
  if (!content?.trim()) return null

  let parsed: unknown
  try {
    parsed = JSON.parse(content)
  } catch {
    return null
  }

  if (!parsed || typeof parsed !== 'object') return null
  const summary = (parsed as Record<string, unknown>).summary
  if (!summary || typeof summary !== 'object') return null

  const record = summary as Record<string, unknown>
  const result: VerificationSummary = {
    supported: toCount(record.supported),
    partiallySupported: toCount(record.partially_supported ?? record.partiallySupported),
    contradicted: toCount(record.contradicted),
    notAddressed: toCount(record.not_addressed ?? record.notAddressed ?? record.unverified),
    total: 0,
  }
  result.total = result.supported + result.partiallySupported + result.contradicted + result.notAddressed

  return result.total > 0 ? result : null
}

/**
 * Find and parse the verification summary from research file artifacts.
 * Accepts any path ending in verification_report.json (e.g. /shared/...).
 */
export const findVerificationSummary = (
  files: ReadonlyArray<{ filename: string; content: string }> | undefined | null
): VerificationSummary | null => {
  if (!files?.length) return null
  // Use the most recent matching file artifact (later events overwrite earlier ones)
  for (let i = files.length - 1; i >= 0; i--) {
    const file = files[i]
    if (file?.filename?.endsWith(VERIFICATION_REPORT_FILENAME)) {
      const summary = parseVerificationSummary(file.content)
      if (summary) return summary
    }
  }
  return null
}

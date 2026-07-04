// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Display-layer text sanitizers.
 *
 * The backend still emits internal engine names (e.g. "AIQ agent working…")
 * in activity/status streams. These helpers rewrite such names to the public
 * product name at PRESENTATION time only — stored data is never mutated.
 */

const ENGINE_NAME_PATTERN = /\bAI-?Q\b/gi

/** Replace internal engine names in backend-streamed display text. */
export const sanitizeEngineText = (text: string | undefined | null): string => {
  if (!text) return ''
  return text.replace(ENGINE_NAME_PATTERN, 'GenAlphAI')
}

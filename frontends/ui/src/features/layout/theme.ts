// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Theme utilities
 *
 * The interface is dark-only. Older builds allowed a persisted "light" value
 * in localStorage; every entry point now ignores and overwrites it so the app
 * cannot boot into the removed light theme.
 */

import type { ThemeMode } from './types'

export const THEME_STORAGE_KEY = 'gx-theme'

export type ResolvedTheme = 'dark'

/**
 * Resolve any requested theme to the only supported theme.
 */
export const resolveTheme = (_theme: ThemeMode): ResolvedTheme => {
  return 'dark'
}

/** Read the persisted theme. Light is no longer supported, so this always returns dark. */
export const getInitialTheme = (): ResolvedTheme => {
  if (typeof window === 'undefined') return 'dark'
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark')
  } catch {
    // Storage unavailable (private mode) — fall through to dark default
  }
  return 'dark'
}

/** Apply the theme to <html> and persist the choice. */
export const applyTheme = (theme: ResolvedTheme): void => {
  if (typeof document === 'undefined') return
  document.documentElement.setAttribute('data-theme', theme)
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // Storage unavailable — theme still applies for this session
  }
}

/**
 * Inline bootstrap script (stringified into the root layout <head>) that sets
 * data-theme before first paint. Must stay dependency-free ES5.
 */
export const THEME_BOOTSTRAP_SCRIPT = `(function(){try{localStorage.setItem('${THEME_STORAGE_KEY}','dark');document.documentElement.setAttribute('data-theme','dark')}catch(e){document.documentElement.setAttribute('data-theme','dark')}})()`

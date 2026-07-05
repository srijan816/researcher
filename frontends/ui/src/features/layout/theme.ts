// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Theme utilities
 *
 * The theme is applied via <html data-theme="dark|light">. Dark token values
 * live on :root in globals.css; html[data-theme='light'] overrides them.
 * An inline script in the root layout applies the persisted (or
 * prefers-color-scheme) theme before hydration to avoid a flash.
 */

import type { ThemeMode } from './types'

export const THEME_STORAGE_KEY = 'gx-theme'

/** The two concrete themes; 'system' resolves to one of these. */
export type ResolvedTheme = 'dark' | 'light'

/** Resolve a ThemeMode to a concrete theme using prefers-color-scheme. */
export const resolveTheme = (theme: ThemeMode): ResolvedTheme => {
  if (theme === 'light' || theme === 'dark') return theme
  if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  }
  return 'dark'
}

/** Read the persisted theme, falling back to the system preference. */
export const getInitialTheme = (): ResolvedTheme => {
  if (typeof window === 'undefined') return 'dark'
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // Storage unavailable (private mode) — fall through to system preference
  }
  return resolveTheme('system')
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
export const THEME_BOOTSTRAP_SCRIPT = `(function(){try{var t=localStorage.getItem('${THEME_STORAGE_KEY}');if(t!=='light'&&t!=='dark'){t=window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'}document.documentElement.setAttribute('data-theme',t)}catch(e){document.documentElement.setAttribute('data-theme','dark')}})()`

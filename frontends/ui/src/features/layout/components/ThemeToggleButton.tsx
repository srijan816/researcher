// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ThemeToggleButton
 *
 * Switches between the dark (default) and light themes. Two shapes:
 * - compact: icon-only rail button for the NavRail bottom area
 * - full: labeled segmented row for the SettingsPanel
 *
 * The theme is persisted to localStorage and applied to <html data-theme>
 * by the layout store's setTheme action.
 */

'use client'

import { type FC } from 'react'
import { useLayoutStore } from '../store'

const SunIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
  </svg>
)

const MoonIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    aria-hidden="true"
  >
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
)

export interface ThemeToggleButtonProps {
  /** Icon-only rail form when true; labeled segmented row otherwise */
  compact?: boolean
}

export const ThemeToggleButton: FC<ThemeToggleButtonProps> = ({ compact = false }) => {
  const theme = useLayoutStore((s) => s.theme)
  const setTheme = useLayoutStore((s) => s.setTheme)
  const isLight = theme === 'light'

  if (compact) {
    return (
      <button
        type="button"
        onClick={() => setTheme(isLight ? 'dark' : 'light')}
        aria-label={isLight ? 'Switch to dark theme' : 'Switch to light theme'}
        title={isLight ? 'Switch to dark theme' : 'Switch to light theme'}
        className="gx-rail-button flex h-10 w-14 items-center justify-center outline-none"
        data-testid="theme-toggle-compact"
      >
        {isLight ? <MoonIcon className="h-5 w-5" /> : <SunIcon className="h-5 w-5" />}
      </button>
    )
  }

  return (
    <div
      className="flex items-center gap-0.5 rounded-full border border-base bg-surface-raised-30 p-0.5"
      role="group"
      aria-label="Interface theme"
      data-testid="theme-toggle"
    >
      {(['dark', 'light'] as const).map((mode) => {
        const selected = theme === mode
        return (
          <button
            key={mode}
            type="button"
            onClick={() => setTheme(mode)}
            aria-pressed={selected}
            aria-label={`Use ${mode} theme`}
            className={`flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium leading-none transition-colors ${
              selected
                ? 'bg-[var(--accent-soft)] text-[var(--accent-primary)]'
                : 'text-subtle hover:bg-surface-raised hover:text-primary'
            }`}
          >
            {mode === 'dark' ? <MoonIcon className="h-3.5 w-3.5" /> : <SunIcon className="h-3.5 w-3.5" />}
            {mode === 'dark' ? 'Dark' : 'Light'}
          </button>
        )
      })}
    </div>
  )
}

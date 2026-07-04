// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Logo Component Shim
 *
 * Replaces the internal @kui/foundations-react Logo component which is
 * not available in the public @nvidia/foundations-react-core package.
 * Renders the Alpha Research "α" mark as an inline SVG: a rounded-square
 * tile with a lime alpha glyph plus a small blue dot constellation.
 */

import { type FC } from 'react'

interface LogoProps {
  /** 'horizontal' renders mark + wordmark; 'logo-only' renders just the α mark */
  kind?: 'horizontal' | 'logo-only'
  size?: 'small' | 'medium' | 'large'
  className?: string
}

/** Dimensions when showing the full logo (wider aspect for horizontal layout) */
const fullSizeMap = {
  small: { width: 120, height: 28 },
  medium: { width: 168, height: 40 },
  large: { width: 224, height: 56 },
} as const

/** Dimensions when showing just the mark */
const markSizeMap = {
  small: { width: 28, height: 28 },
  medium: { width: 40, height: 40 },
  large: { width: 56, height: 56 },
} as const

const LIME = '#C7FF3D'
const BLUE = '#5AA7FF'
const INK = '#0B0C0E'

/** The α mark: rounded-square tile, lime alpha, blue dot constellation. */
const AlphaMark: FC<{ px: number }> = ({ px }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 40 40"
    width={px}
    height={px}
    aria-label="Alpha Research"
    role="img"
  >
    <rect x="1" y="1" width="38" height="38" rx="10" fill={INK} stroke={LIME} strokeWidth="1.5" />
    <text
      x="18"
      y="28"
      fontFamily="inherit"
      fontStyle="italic"
      fontWeight="600"
      fontSize="24"
      fill={LIME}
      textAnchor="middle"
    >
      α
    </text>
    <circle cx="32" cy="11" r="1.6" fill={BLUE} />
    <circle cx="32" cy="18" r="1.3" fill={BLUE} />
    <circle cx="32" cy="24.5" r="1.1" fill={BLUE} />
    <circle cx="32" cy="30" r="0.9" fill={BLUE} />
  </svg>
)

/**
 * Renders the Alpha Research logo.
 *
 * 'logo-only' shows the α tile; 'horizontal' adds the "Alpha Research"
 * wordmark in Inter 600.
 */
export const Logo: FC<LogoProps> = ({ kind = 'horizontal', size = 'medium', className }) => {
  const isMarkOnly = kind === 'logo-only'
  const dims = isMarkOnly ? markSizeMap[size] : fullSizeMap[size]
  const markPx = dims.height

  return (
    <span
      className={className}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: isMarkOnly ? 0 : Math.round(markPx * 0.25),
        width: dims.width,
        height: dims.height,
      }}
    >
      <AlphaMark px={markPx} />
      {!isMarkOnly && (
        <span
          style={{
            fontFamily: 'var(--font-sans), Inter, sans-serif',
            fontWeight: 600,
            fontSize: Math.round(markPx * 0.42),
            letterSpacing: '-0.01em',
            color: '#F4F5F7',
            whiteSpace: 'nowrap',
          }}
        >
          Alpha Research
        </span>
      )}
    </span>
  )
}

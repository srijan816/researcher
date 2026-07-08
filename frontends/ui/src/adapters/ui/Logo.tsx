// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Logo Component
 *
 * Renders the official GenAlphAI logo artwork (public/brand/):
 * - 'horizontal': logo-dark.png — the real "Gen α i" logo
 *   (red α with circuit detail) on a transparent background.
 * - 'logo-only': alpha-mark.png — just the α glyph.
 *
 * These are raster brand assets and intentionally do not theme with currentColor.
 */

'use client'

import { type FC } from 'react'

interface LogoProps {
  /** 'horizontal' renders the wordmark; 'logo-only' renders just the α glyph */
  kind?: 'horizontal' | 'logo-only'
  size?: 'small' | 'medium' | 'large'
  className?: string
}

/** Dimensions for the full wordmark (asset aspect ≈ 1.61) */
const fullSizeMap = {
  small: { width: 45, height: 28 },
  medium: { width: 64, height: 40 },
  large: { width: 90, height: 56 },
} as const

/** Dimensions when showing just the α glyph (square asset) */
const markSizeMap = {
  small: { width: 28, height: 28 },
  medium: { width: 40, height: 40 },
  large: { width: 56, height: 56 },
} as const

const LOGO_SRC = {
  horizontal: '/brand/logo-dark.png',
  mark: '/brand/alpha-mark.png',
} as const

export const Logo: FC<LogoProps> = ({ kind = 'horizontal', size = 'medium', className }) => {
  const isMarkOnly = kind === 'logo-only'
  const dims = isMarkOnly ? markSizeMap[size] : fullSizeMap[size]
  const src = isMarkOnly ? LOGO_SRC.mark : LOGO_SRC.horizontal

  return (
    // eslint-disable-next-line @next/next/no-img-element -- static brand asset, no optimization needed
    <img
      src={src}
      alt="GenAlphAI"
      width={dims.width}
      height={dims.height}
      className={className}
      style={{ objectFit: 'contain' }}
      draggable={false}
    />
  )
}

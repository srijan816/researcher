// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Logo Component
 *
 * Renders the official GenAlphAI logo artwork (public/brand/):
 * - 'horizontal': logo-dark.png — the real "Gen α i" logo (white text,
 *   rose α with circuit detail and glow) on a transparent background,
 *   produced from the brand upload with the α recolored to the rose accent.
 * - 'logo-only': alpha-mark.png — just the rose α glyph, square-cropped.
 *
 * These are raster brand assets and intentionally do not theme with
 * currentColor; they are made for dark surfaces.
 */

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

export const Logo: FC<LogoProps> = ({ kind = 'horizontal', size = 'medium', className }) => {
  const isMarkOnly = kind === 'logo-only'
  const dims = isMarkOnly ? markSizeMap[size] : fullSizeMap[size]
  const src = isMarkOnly ? '/brand/alpha-mark.png' : '/brand/logo-dark.png'

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

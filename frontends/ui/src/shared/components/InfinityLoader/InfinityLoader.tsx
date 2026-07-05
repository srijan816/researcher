// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * InfinityLoader
 *
 * The running-research indicator: an ∞ shape whose LEFT loop is a solid
 * α-like stroke and whose RIGHT loop is dotted, with bright dots flowing
 * continuously along the full infinity path (SVG animateMotion).
 *
 * - Brand red stroke, one blue flowing dot among the red ones.
 * - prefers-reduced-motion: flowing dots are hidden via CSS
 *   (.gx-infinity-motion) and the static ∞ remains.
 */

import { type FC } from 'react'

/** Full infinity path: two loops meeting at the center (viewBox 100x48) */
const INFINITY_PATH =
  'M50,24 C50,7 21,7 21,24 C21,41 50,41 50,24 C50,7 79,7 79,24 C79,41 50,41 50,24'
/** Left loop only — drawn solid, α-like */
const LEFT_LOOP_PATH = 'M50,24 C50,7 21,7 21,24 C21,41 50,41 50,24'
/** Right loop only — drawn dotted */
const RIGHT_LOOP_PATH = 'M50,24 C50,7 79,7 79,24 C79,41 50,41 50,24'

export interface InfinityLoaderProps {
  /** Rendered height in px (28-40 recommended). Default 32. */
  size?: number
  className?: string
  /** Accessible label. Default "Research in progress". */
  label?: string
}

export const InfinityLoader: FC<InfinityLoaderProps> = ({
  size = 32,
  className,
  label = 'Research in progress',
}) => {
  const width = Math.round((size * 100) / 48)

  return (
    <span
      className={['inline-flex items-center', className].filter(Boolean).join(' ')}
      role="status"
      aria-label={label}
      data-testid="infinity-loader"
    >
      <svg
        width={width}
        height={size}
        viewBox="0 0 100 48"
        fill="none"
        aria-hidden="true"
      >
        {/* Left loop: solid α-like stroke */}
        <path
          d={LEFT_LOOP_PATH}
          stroke="var(--accent-primary, #F43F5E)"
          strokeWidth="5"
          strokeLinecap="round"
          opacity="0.9"
        />
        {/* Right loop: dotted */}
        <path
          d={RIGHT_LOOP_PATH}
          stroke="var(--accent-primary, #F43F5E)"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray="0.5 8"
          opacity="0.75"
        />
        {/* Flowing dots along the full infinity path */}
        <circle className="gx-infinity-motion" r="3.4" fill="var(--accent-primary-hover, #FB7185)">
          <animateMotion dur="2.4s" repeatCount="indefinite" path={INFINITY_PATH} />
        </circle>
        <circle className="gx-infinity-motion" r="2.6" fill="var(--accent-info, #38BDF8)">
          <animateMotion dur="2.4s" begin="-0.8s" repeatCount="indefinite" path={INFINITY_PATH} />
        </circle>
        <circle className="gx-infinity-motion" r="2.2" fill="var(--accent-primary, #F43F5E)">
          <animateMotion dur="2.4s" begin="-1.6s" repeatCount="indefinite" path={INFINITY_PATH} />
        </circle>
      </svg>
    </span>
  )
}

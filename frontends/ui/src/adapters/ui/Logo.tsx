// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Logo Component
 *
 * Renders the real GenAlphAI brand assets (see public/brand/):
 * - 'horizontal': the "Gen α AI" wordmark, inlined from logo-plain.svg with
 *   every glyph on currentColor so it themes with the surrounding text
 *   (off-white at rest, accent rose on hover in the nav rail). The α glyph
 *   always renders in the rose brand accent.
 * - 'logo-only': just the α glyph extracted from the same wordmark.
 */

import { type FC } from 'react'

interface LogoProps {
  /** 'horizontal' renders the wordmark; 'logo-only' renders just the α glyph */
  kind?: 'horizontal' | 'logo-only'
  size?: 'small' | 'medium' | 'large'
  className?: string
}

/** Dimensions for the full wordmark (viewBox aspect ≈ 2.41) */
const fullSizeMap = {
  small: { width: 68, height: 28 },
  medium: { width: 96, height: 40 },
  large: { width: 135, height: 56 },
} as const

/** Dimensions when showing just the α glyph */
const markSizeMap = {
  small: { width: 28, height: 28 },
  medium: { width: 40, height: 40 },
  large: { width: 56, height: 56 },
} as const

/** α glyph path from the GenAlphAI wordmark (currentColor). */
const ALPHA_PATH =
  'M507 -24C775 -24 908 168 983 330L1006 0H1213L1142 559L1383 1118H1176L1056 797C1037 954 957 1132 693 1132C393 1132 153 900 95 555C39 215 209 -24 507 -24ZM968 559 967 556C910 405 768 165 541 165C361 165 279 314 318 553C358 795 503 945 692 945C923 945 958 711 968 562Z'

/** The α is the brand accent bearer — always rendered in rose. */
const ALPHA_ACCENT = '#F43F5E'

const WORDMARK_PATHS: Array<{ d: string; transform: string; fill?: string }> = [
  {
    d: 'M791 -20C1152 -20 1413 216 1413 593V764H825V576H1190C1185 338 1028 188 792 188C529 188 339 386 339 745C339 1105 532 1302 783 1302C979 1302 1112 1193 1166 1019H1400C1350 1309 1102 1510 781 1510C395 1510 113 1220 113 744C113 273 386 -20 791 -20Z',
    transform: 'translate(24.00,84) scale(0.02148,-0.02148)',
  },
  {
    d: 'M630 -23C869 -23 1040 94 1093 268L886 317C847 213 755 159 632 159C449 159 325 276 317 492H1110V570C1110 973 866 1132 611 1132C298 1132 96 895 96 551C96 204 300 -23 630 -23ZM318 656C330 817 436 950 612 950C781 950 876 832 892 656Z',
    transform: 'translate(56.89,84) scale(0.02148,-0.02148)',
  },
  {
    d: 'M368 662C368 839 478 940 628 940C774 940 864 844 864 683V0H1084V710C1084 986 932 1132 704 1132C546 1132 429 1061 359 903L358 1118H148V0H368Z',
    transform: 'translate(82.74,84) scale(0.02148,-0.02148)',
  },
  {
    d: ALPHA_PATH,
    transform: 'translate(120.94,84) scale(0.02148,-0.02148)',
    fill: ALPHA_ACCENT,
  },
  {
    d: 'M461 -25C642 -25 748 66 791 153H802V0H1015V742C1015 1065 758 1132 578 1132C379 1132 191 1054 115 854L322 800C353 876 435 953 581 953C721 953 795 881 795 757V751C795 672 710 673 513 650C302 626 83 570 83 316C83 96 248 -25 461 -25ZM510 152C387 152 299 207 299 313C299 428 401 470 525 487C593 496 763 514 796 546V403C796 272 689 152 510 152Z',
    transform: 'translate(164.59,84) scale(0.02148,-0.02148)',
  },
  {
    d: 'M148 0H368V1118H148ZM259 1289C335 1289 398 1347 398 1419C398 1492 335 1550 259 1550C183 1550 120 1492 120 1419C120 1347 183 1289 259 1289Z',
    transform: 'translate(189.57,84) scale(0.02148,-0.02148)',
  },
]

/**
 * Renders the GenAlphAI logo.
 * All paths use currentColor, so set text color on the parent to theme it.
 */
export const Logo: FC<LogoProps> = ({ kind = 'horizontal', size = 'medium', className }) => {
  const isMarkOnly = kind === 'logo-only'
  const dims = isMarkOnly ? markSizeMap[size] : fullSizeMap[size]

  if (isMarkOnly) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="120.5 58.5 32 27"
        width={dims.width}
        height={dims.height}
        role="img"
        aria-label="GenAlphAI"
        className={className}
      >
        <title>GenAlphAI</title>
        <path d={ALPHA_PATH} fill={ALPHA_ACCENT} transform="translate(120.94,84) scale(0.02148,-0.02148)" />
      </svg>
    )
  }

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="18.0 22 192.7 80"
      width={dims.width}
      height={dims.height}
      role="img"
      aria-label="GenAlphAI"
      className={className}
    >
      <title>Gen α AI</title>
      {WORDMARK_PATHS.map((path, index) => (
        <path key={index} d={path.d} fill={path.fill ?? 'currentColor'} transform={path.transform} />
      ))}
    </svg>
  )
}

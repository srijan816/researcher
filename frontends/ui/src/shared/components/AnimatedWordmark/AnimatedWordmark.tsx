// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AnimatedWordmark
 *
 * The GenAlphAI logo whose α glyph is alive. Layered inside a container
 * locked to the brand canvas aspect ratio (1005x626):
 *
 * 1. A <canvas> particle field constrained to the α glyph. The soft mask
 *    /brand/alpha-shape.png is sampled offscreen (alpha > threshold)
 *    to build allowed positions; particles drift and twinkle inside it.
 *    Palette: primarily brand red (#F43F5E→#FB7185 across the glyph) with
 *    ~12% green, ~12% blue and ~8% neutral sprinkles — festive but coherent.
 * 2. /brand/text-overlay.png — the lettering — sits on top, exactly
 *    as in the static logo (the α is behind the text).
 *
 * Density is size-responsive: ~760 particles at canvas widths >= 480px,
 * ~440 below; the `maxParticles` prop caps this (e.g. small NavRail usage).
 *
 * `active` (default true) controls animation: when false the particle field
 * is drawn once as a static frame — used as an ambient "research running"
 * indicator in the NavRail.
 *
 * prefers-reduced-motion: no canvas at all; renders the static logo instead.
 *
 * Animation pauses when the tab is hidden (visibilitychange) and when the
 * element is off-screen (IntersectionObserver). Clean unmount cancels the
 * rAF loop and disconnects observers.
 */

'use client'

import { type FC, useEffect, useRef } from 'react'
import { useReducedMotion } from '@/hooks/use-reduced-motion'

const ASSETS = {
  mask: '/brand/alpha-shape.png',
  textOverlay: '/brand/text-overlay.png',
  staticLogo: '/brand/logo-dark.png',
  aspectRatio: '1005 / 626',
} as const

/** Size-responsive particle density */
const WIDE_CANVAS_MIN_WIDTH = 480
const PARTICLE_COUNT_WIDE = 760
const PARTICLE_COUNT_NARROW = 440
const MAX_DPR = 2
/** Mask alpha threshold (0–1) above which a pixel is an allowed position */
const MASK_ALPHA_THRESHOLD = 0.35
/** Sprinkle ratios — hue by meaning kept subtle inside the brand mark */
const GREEN_RATIO = 0.12
const BLUE_RATIO = 0.12
const NEUTRAL_RATIO = 0.08
/** Maximum drift distance (CSS px) around a particle's home point */
const DRIFT_RADIUS = 3.5
const MIN_RADIUS = 1.6
const MAX_RADIUS = 4.4
const GLOBAL_OPACITY = 0.9
const HALO_OPACITY = 0.25

type RGB = readonly [number, number, number]

interface WordmarkPalette {
  redStart: RGB
  redEnd: RGB
  green: RGB
  blue: RGB
  neutral: RGB
}

/** Dark surfaces: bright rose reds, mint green, sky blue, warm off-white */
const DARK_PALETTE: WordmarkPalette = {
  redStart: [244, 63, 94], // #F43F5E
  redEnd: [251, 113, 133], // #FB7185
  green: [52, 211, 153], // #34D399
  blue: [56, 189, 248], // #38BDF8
  neutral: [243, 239, 230], // #F3EFE6
}

interface MaskPoint {
  x: number
  y: number
}

interface MaskBuildResult {
  points: MaskPoint[]
  clipCanvas: HTMLCanvasElement | null
}

interface WordmarkParticle {
  homeX: number
  homeY: number
  radius: number
  baseOpacity: number
  phase: number
  driftPhase: number
  speed: number
  color: string
}

const lerpChannel = (a: number, b: number, t: number): number =>
  Math.round(a + (b - a) * t)

/** Interpolate the red gradient by normalized x position across the glyph. */
const gradientColor = (palette: WordmarkPalette, t: number): string => {
  const r = lerpChannel(palette.redStart[0], palette.redEnd[0], t)
  const g = lerpChannel(palette.redStart[1], palette.redEnd[1], t)
  const b = lerpChannel(palette.redStart[2], palette.redEnd[2], t)
  return `${r}, ${g}, ${b}`
}

/** Pick a particle color: red gradient with green/blue/neutral sprinkles. */
const particleColor = (palette: WordmarkPalette, normalizedX: number): string => {
  const roll = Math.random()
  if (roll < GREEN_RATIO) return palette.green.join(', ')
  if (roll < GREEN_RATIO + BLUE_RATIO) return palette.blue.join(', ')
  if (roll < GREEN_RATIO + BLUE_RATIO + NEUTRAL_RATIO) return palette.neutral.join(', ')
  return gradientColor(palette, normalizedX)
}

/** Particle count for a given canvas width, optionally capped. */
export const particleCountForWidth = (
  width: number,
  maxParticles?: number
): number => {
  const base = width >= WIDE_CANVAS_MIN_WIDTH ? PARTICLE_COUNT_WIDE : PARTICLE_COUNT_NARROW
  return maxParticles ? Math.min(base, maxParticles) : base
}

/** Sample allowed positions from the alpha channel of the shape mask image. */
const buildMask = (
  image: HTMLImageElement,
  width: number,
  height: number,
  thresholdRatio: number
): MaskBuildResult => {
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const ctx = off.getContext('2d')
  if (!ctx) return { points: [], clipCanvas: null }

  ctx.drawImage(image, 0, 0, width, height)

  let data: Uint8ClampedArray
  try {
    data = ctx.getImageData(0, 0, width, height).data
  } catch {
    return { points: [], clipCanvas: null }
  }

  const threshold = thresholdRatio * 255
  const points: MaskPoint[] = []
  const clipCanvas = document.createElement('canvas')
  clipCanvas.width = width
  clipCanvas.height = height
  const clipCtx = clipCanvas.getContext('2d')

  if (!clipCtx) return { points, clipCanvas: null }

  const clipImage = clipCtx.createImageData(width, height)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4
      if (data[index + 3] > threshold) {
        if (x % 2 === 0 && y % 2 === 0) points.push({ x, y })
        clipImage.data[index] = 255
        clipImage.data[index + 1] = 255
        clipImage.data[index + 2] = 255
        clipImage.data[index + 3] = 255
      }
    }
  }

  clipCtx.putImageData(clipImage, 0, 0)
  return { points, clipCanvas }
}

const createParticles = (
  mask: MaskPoint[],
  count: number,
  palette: WordmarkPalette
): WordmarkParticle[] => {
  if (mask.length === 0) return []

  let minX = Infinity
  let maxX = -Infinity
  for (const p of mask) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
  }
  const span = Math.max(maxX - minX, 1)

  const particles: WordmarkParticle[] = []
  for (let i = 0; i < count; i++) {
    const point = mask[Math.floor(Math.random() * mask.length)]
    particles.push({
      homeX: point.x,
      homeY: point.y,
      radius: MIN_RADIUS + Math.random() * (MAX_RADIUS - MIN_RADIUS),
      baseOpacity: 0.3 + Math.random() * 0.6,
      phase: Math.random() * Math.PI * 2,
      driftPhase: Math.random() * Math.PI * 2,
      speed: 0.4 + Math.random() * 0.8,
      color: particleColor(palette, (point.x - minX) / span),
    })
  }
  return particles
}

export interface AnimatedWordmarkProps {
  className?: string
  /**
   * Whether the particle field animates. When false a single static frame is
   * drawn (ambient/idle state). Defaults to true.
   */
  active?: boolean
  /** Optional cap on the particle count (small renders, e.g. NavRail). */
  maxParticles?: number
}

export const AnimatedWordmark: FC<AnimatedWordmarkProps> = ({
  className,
  active = true,
  maxParticles,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const prefersReducedMotion = useReducedMotion()

  useEffect(() => {
    if (prefersReducedMotion) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR)
    const width = Math.floor(rect.width)
    const height = Math.floor(rect.height)
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    let particles: WordmarkParticle[] = []
    let clipCanvas: HTMLCanvasElement | null = null
    let frameId = 0
    let isRunning = false
    let isTabVisible = document.visibilityState !== 'hidden'
    let isInView = true
    let isDisposed = false
    const startTime = performance.now()

    const drawFrame = (t: number): void => {
      ctx.clearRect(0, 0, width, height)
      for (const p of particles) {
        const drift = t * p.speed
        const x = p.homeX + Math.cos(drift + p.driftPhase) * DRIFT_RADIUS
        const y = p.homeY + Math.sin(drift * 0.8 + p.driftPhase) * DRIFT_RADIUS
        const twinkle = 0.55 + 0.45 * Math.sin(t * p.speed * 2 + p.phase)
        const alpha = p.baseOpacity * twinkle * GLOBAL_OPACITY

        // Soft halo: 2x radius at low alpha behind the core dot
        ctx.globalAlpha = alpha * HALO_OPACITY
        ctx.fillStyle = `rgb(${p.color})`
        ctx.beginPath()
        ctx.arc(x, y, p.radius * 2, 0, Math.PI * 2)
        ctx.fill()

        ctx.globalAlpha = alpha
        ctx.beginPath()
        ctx.arc(x, y, p.radius, 0, Math.PI * 2)
        ctx.fill()
      }
      if (clipCanvas) {
        ctx.globalAlpha = 1
        ctx.globalCompositeOperation = 'destination-in'
        ctx.drawImage(clipCanvas, 0, 0, width, height)
        ctx.globalCompositeOperation = 'source-over'
      }
      ctx.globalAlpha = 1
    }

    const animate = (now: number): void => {
      drawFrame((now - startTime) / 1000)
      frameId = requestAnimationFrame(animate)
    }

    const syncRunningState = (): void => {
      const shouldRun =
        !isDisposed && active && isTabVisible && isInView && particles.length > 0
      if (shouldRun && !isRunning) {
        isRunning = true
        frameId = requestAnimationFrame(animate)
      } else if (!shouldRun && isRunning) {
        isRunning = false
        cancelAnimationFrame(frameId)
      }
      // Idle: particles frozen mid-twinkle as a single static frame
      if (!isDisposed && !active && !isRunning && particles.length > 0) {
        drawFrame(0)
      }
    }

    const handleVisibility = (): void => {
      isTabVisible = document.visibilityState !== 'hidden'
      syncRunningState()
    }
    document.addEventListener('visibilitychange', handleVisibility)

    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        isInView = entry.isIntersecting
      }
      syncRunningState()
    })
    observer.observe(canvas)

    const maskImage = new Image()
    maskImage.decoding = 'async'
    maskImage.onload = () => {
      if (isDisposed) return
      const mask = buildMask(
        maskImage,
        width,
        height,
        MASK_ALPHA_THRESHOLD
      )
      clipCanvas = mask.clipCanvas
      particles = createParticles(
        mask.points,
        particleCountForWidth(width, maxParticles),
        DARK_PALETTE
      )
      syncRunningState()
    }
    maskImage.onerror = () => {
      // Mask unavailable — the text overlay still renders; skip particles.
    }
    maskImage.src = ASSETS.mask

    return () => {
      isDisposed = true
      cancelAnimationFrame(frameId)
      document.removeEventListener('visibilitychange', handleVisibility)
      observer.disconnect()
    }
  }, [prefersReducedMotion, active, maxParticles])

  if (prefersReducedMotion) {
    return (
      <div
        className={['relative', className].filter(Boolean).join(' ')}
        style={{ aspectRatio: ASSETS.aspectRatio }}
        data-testid="animated-wordmark-static"
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- static brand asset */}
        <img
          src={ASSETS.staticLogo}
          alt="GenAlphAI"
          className="absolute inset-0 h-full w-full object-contain"
          draggable={false}
        />
      </div>
    )
  }

  return (
    <div
      className={['relative', className].filter(Boolean).join(' ')}
      style={{ aspectRatio: ASSETS.aspectRatio }}
      data-testid="animated-wordmark"
      role="img"
      aria-label="GenAlphAI"
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      />
      {/* eslint-disable-next-line @next/next/no-img-element -- static brand asset */}
      <img
        src={ASSETS.textOverlay}
        alt=""
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full"
        draggable={false}
      />
    </div>
  )
}

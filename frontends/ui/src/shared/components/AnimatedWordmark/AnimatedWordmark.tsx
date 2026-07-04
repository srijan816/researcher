// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AnimatedWordmark
 *
 * The GenAlphAI logo whose α glyph is alive. Layered inside a container
 * locked to the brand canvas aspect ratio (1005x626):
 *
 * 1. A <canvas> particle field constrained to the α glyph. The soft mask
 *    /brand/alpha-shape.png is sampled offscreen (alpha > threshold) to
 *    build allowed positions; particles drift and twinkle inside it,
 *    colored violet→cyan across the glyph with off-white sparkles.
 * 2. /brand/text-overlay.png — the white "Gen … i" lettering — sits on
 *    top, exactly as in the static logo (the α is behind the text).
 *
 * prefers-reduced-motion: no canvas at all; renders the static
 * /brand/logo-dark.png wordmark instead.
 *
 * Animation pauses when the tab is hidden (visibilitychange) and when the
 * element is off-screen (IntersectionObserver). Clean unmount cancels the
 * rAF loop and disconnects observers.
 */

'use client'

import { type FC, useEffect, useRef } from 'react'
import { useReducedMotion } from '@/hooks/use-reduced-motion'

const ASPECT_RATIO = '1005 / 626'
const MASK_SRC = '/brand/alpha-shape.png'
const TEXT_OVERLAY_SRC = '/brand/text-overlay.png'
const STATIC_LOGO_SRC = '/brand/logo-dark.png'

const PARTICLE_COUNT = 520
const MAX_DPR = 2
/** Mask alpha threshold (0–1) above which a pixel is an allowed position */
const MASK_ALPHA_THRESHOLD = 0.35
/** Fraction of particles rendered as off-white sparkles */
const OFF_WHITE_RATIO = 0.12
/** Maximum drift distance (CSS px) around a particle's home point */
const DRIFT_RADIUS = 3.5
const MIN_RADIUS = 0.8
const MAX_RADIUS = 2.2
const GLOBAL_OPACITY = 0.9

/** Brand gradient endpoints across the glyph (left → right) */
const VIOLET: readonly [number, number, number] = [139, 92, 246] // #8B5CF6
const CYAN: readonly [number, number, number] = [34, 211, 238] // #22D3EE
const OFF_WHITE: readonly [number, number, number] = [243, 239, 230] // #F3EFE6

interface MaskPoint {
  x: number
  y: number
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

/** Interpolate violet→cyan by normalized x position across the glyph. */
const gradientColor = (t: number): string => {
  const r = lerpChannel(VIOLET[0], CYAN[0], t)
  const g = lerpChannel(VIOLET[1], CYAN[1], t)
  const b = lerpChannel(VIOLET[2], CYAN[2], t)
  return `${r}, ${g}, ${b}`
}

/** Sample allowed positions from the alpha channel of the shape mask image. */
const buildMaskPoints = (
  image: HTMLImageElement,
  width: number,
  height: number
): MaskPoint[] => {
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const ctx = off.getContext('2d')
  if (!ctx) return []

  ctx.drawImage(image, 0, 0, width, height)

  let data: Uint8ClampedArray
  try {
    data = ctx.getImageData(0, 0, width, height).data
  } catch {
    return []
  }

  const threshold = MASK_ALPHA_THRESHOLD * 255
  const points: MaskPoint[] = []
  // Sample every 2nd pixel to keep the candidate pool small
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      if (data[(y * width + x) * 4 + 3] > threshold) {
        points.push({ x, y })
      }
    }
  }
  return points
}

const createParticles = (mask: MaskPoint[]): WordmarkParticle[] => {
  if (mask.length === 0) return []

  let minX = Infinity
  let maxX = -Infinity
  for (const p of mask) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
  }
  const span = Math.max(maxX - minX, 1)

  const particles: WordmarkParticle[] = []
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const point = mask[Math.floor(Math.random() * mask.length)]
    const isOffWhite = Math.random() < OFF_WHITE_RATIO
    particles.push({
      homeX: point.x,
      homeY: point.y,
      radius: MIN_RADIUS + Math.random() * (MAX_RADIUS - MIN_RADIUS),
      baseOpacity: 0.3 + Math.random() * 0.6,
      phase: Math.random() * Math.PI * 2,
      driftPhase: Math.random() * Math.PI * 2,
      speed: 0.4 + Math.random() * 0.8,
      color: isOffWhite
        ? OFF_WHITE.join(', ')
        : gradientColor((point.x - minX) / span),
    })
  }
  return particles
}

export interface AnimatedWordmarkProps {
  className?: string
}

export const AnimatedWordmark: FC<AnimatedWordmarkProps> = ({ className }) => {
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
    let frameId = 0
    let isRunning = false
    let isTabVisible = document.visibilityState !== 'hidden'
    let isInView = true
    let isDisposed = false
    const startTime = performance.now()

    const animate = (now: number): void => {
      const t = (now - startTime) / 1000
      ctx.clearRect(0, 0, width, height)
      for (const p of particles) {
        const drift = t * p.speed
        const x = p.homeX + Math.cos(drift + p.driftPhase) * DRIFT_RADIUS
        const y = p.homeY + Math.sin(drift * 0.8 + p.driftPhase) * DRIFT_RADIUS
        const twinkle = 0.55 + 0.45 * Math.sin(t * p.speed * 2 + p.phase)
        const alpha = p.baseOpacity * twinkle * GLOBAL_OPACITY

        // Soft halo: 2x radius at low alpha behind the core dot
        ctx.globalAlpha = alpha * 0.25
        ctx.fillStyle = `rgb(${p.color})`
        ctx.beginPath()
        ctx.arc(x, y, p.radius * 2, 0, Math.PI * 2)
        ctx.fill()

        ctx.globalAlpha = alpha
        ctx.beginPath()
        ctx.arc(x, y, p.radius, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1
      frameId = requestAnimationFrame(animate)
    }

    const syncRunningState = (): void => {
      const shouldRun =
        !isDisposed && isTabVisible && isInView && particles.length > 0
      if (shouldRun && !isRunning) {
        isRunning = true
        frameId = requestAnimationFrame(animate)
      } else if (!shouldRun && isRunning) {
        isRunning = false
        cancelAnimationFrame(frameId)
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
      particles = createParticles(buildMaskPoints(maskImage, width, height))
      syncRunningState()
    }
    maskImage.onerror = () => {
      // Mask unavailable — the text overlay still renders; skip particles.
    }
    maskImage.src = MASK_SRC

    return () => {
      isDisposed = true
      cancelAnimationFrame(frameId)
      document.removeEventListener('visibilitychange', handleVisibility)
      observer.disconnect()
    }
  }, [prefersReducedMotion])

  if (prefersReducedMotion) {
    return (
      <div
        className={['relative', className].filter(Boolean).join(' ')}
        style={{ aspectRatio: ASPECT_RATIO }}
        data-testid="animated-wordmark-static"
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- static brand asset */}
        <img
          src={STATIC_LOGO_SRC}
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
      style={{ aspectRatio: ASPECT_RATIO }}
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
        src={TEXT_OVERLAY_SRC}
        alt=""
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full"
        draggable={false}
      />
    </div>
  )
}

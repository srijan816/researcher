// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AlphaParticles
 *
 * Decorative particle field constrained to the shape of an italic-serif α
 * glyph — the GenAlphAI brand mark. An offscreen canvas renders the glyph,
 * its pixel alpha builds an allowed-position mask, and a small set of
 * particles drift and twinkle only inside that mask.
 *
 * - Violet + off-white particles, varied opacity, subtle by design.
 * - requestAnimationFrame with capped DPR for performance.
 * - prefers-reduced-motion: renders a single static faint α made of dots.
 */

'use client'

import { type FC, useEffect, useRef } from 'react'
import { useReducedMotion } from '@/hooks/use-reduced-motion'

const PARTICLE_COUNT = 200
const MAX_DPR = 2
const VIOLET = '139, 92, 246'
const OFF_WHITE = '243, 239, 230'
/** Fraction of particles rendered in the off-white secondary color */
const OFF_WHITE_RATIO = 0.3
/** Maximum drift distance (px, in CSS pixels) around a particle's home point */
const DRIFT_RADIUS = 3
const GLOBAL_OPACITY = 0.5

interface MaskPoint {
  x: number
  y: number
}

interface AlphaParticle {
  homeX: number
  homeY: number
  size: number
  baseOpacity: number
  phase: number
  driftPhase: number
  speed: number
  isOffWhite: boolean
}

/** Sample allowed positions from the alpha of an offscreen 'α' rendering. */
const buildGlyphMask = (width: number, height: number): MaskPoint[] => {
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const ctx = off.getContext('2d')
  if (!ctx) return []

  ctx.font = `italic 600 ${Math.floor(height * 0.92)}px Fraunces, Georgia, 'Times New Roman', serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = '#fff'
  ctx.fillText('α', width / 2, height * 0.56)

  const { data } = ctx.getImageData(0, 0, width, height)
  const points: MaskPoint[] = []
  // Sample every 2nd pixel to keep the candidate pool small
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      if (data[(y * width + x) * 4 + 3] > 128) {
        points.push({ x, y })
      }
    }
  }
  return points
}

const createParticles = (mask: MaskPoint[]): AlphaParticle[] => {
  const particles: AlphaParticle[] = []
  if (mask.length === 0) return particles
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const point = mask[Math.floor(Math.random() * mask.length)]
    particles.push({
      homeX: point.x,
      homeY: point.y,
      size: 0.6 + Math.random() * 1.1,
      baseOpacity: 0.25 + Math.random() * 0.65,
      phase: Math.random() * Math.PI * 2,
      driftPhase: Math.random() * Math.PI * 2,
      speed: 0.4 + Math.random() * 0.8,
      isOffWhite: Math.random() < OFF_WHITE_RATIO,
    })
  }
  return particles
}

export interface AlphaParticlesProps {
  className?: string
}

export const AlphaParticles: FC<AlphaParticlesProps> = ({ className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const prefersReducedMotion = useReducedMotion()

  useEffect(() => {
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

    const mask = buildGlyphMask(width, height)
    const particles = createParticles(mask)
    if (particles.length === 0) return

    const drawStatic = (): void => {
      ctx.clearRect(0, 0, width, height)
      for (const p of particles) {
        ctx.globalAlpha = p.baseOpacity * GLOBAL_OPACITY * 0.8
        ctx.fillStyle = p.isOffWhite ? `rgb(${OFF_WHITE})` : `rgb(${VIOLET})`
        ctx.beginPath()
        ctx.arc(p.homeX, p.homeY, p.size, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1
    }

    if (prefersReducedMotion) {
      // Static faint α made of dots — no animation
      drawStatic()
      return
    }

    let frameId = 0
    const startTime = performance.now()

    const animate = (now: number): void => {
      const t = (now - startTime) / 1000
      ctx.clearRect(0, 0, width, height)
      for (const p of particles) {
        // Gentle circular drift around the home point
        const drift = t * p.speed
        const x = p.homeX + Math.cos(drift + p.driftPhase) * DRIFT_RADIUS
        const y = p.homeY + Math.sin(drift * 0.8 + p.driftPhase) * DRIFT_RADIUS
        // Twinkle
        const twinkle = 0.6 + 0.4 * Math.sin(t * p.speed * 2 + p.phase)
        ctx.globalAlpha = p.baseOpacity * twinkle * GLOBAL_OPACITY
        ctx.fillStyle = p.isOffWhite ? `rgb(${OFF_WHITE})` : `rgb(${VIOLET})`
        ctx.beginPath()
        ctx.arc(x, y, p.size, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1
      frameId = requestAnimationFrame(animate)
    }

    frameId = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frameId)
  }, [prefersReducedMotion])

  return (
    <canvas
      ref={canvasRef}
      className={['block', 'h-full', 'w-full', className].filter(Boolean).join(' ')}
      aria-hidden="true"
    />
  )
}

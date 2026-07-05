// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { AnimatedWordmark } from './AnimatedWordmark'
import { useReducedMotion } from '@/hooks/use-reduced-motion'
import { useLayoutStore } from '@/features/layout/store'

vi.mock('@/hooks/use-reduced-motion', () => ({
  useReducedMotion: vi.fn(() => false),
}))

const mockUseReducedMotion = vi.mocked(useReducedMotion)

describe('AnimatedWordmark', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseReducedMotion.mockReturnValue(false)
    // Assets are theme-dependent; pin the default dark theme for these tests
    useLayoutStore.setState({ theme: 'dark' })
  })

  describe('animated rendering', () => {
    test('renders the layered wordmark with canvas and text overlay', () => {
      const { container } = render(<AnimatedWordmark />)

      const wordmark = screen.getByTestId('animated-wordmark')
      expect(wordmark).toBeInTheDocument()
      expect(wordmark).toHaveAttribute('aria-label', 'GenAlphAI')

      // Canvas particle layer
      expect(container.querySelector('canvas')).toBeInTheDocument()

      // Text lettering sits on top of the particles
      const overlay = container.querySelector('img')
      expect(overlay).toHaveAttribute('src', '/brand/text-overlay.png')
    })

    test('applies the provided className to the container', () => {
      render(<AnimatedWordmark className="custom-class" />)
      expect(screen.getByTestId('animated-wordmark')).toHaveClass('custom-class')
    })
  })

  describe('theme awareness', () => {
    test('uses the light overlay assets on the light theme', () => {
      useLayoutStore.setState({ theme: 'light' })
      const { container } = render(<AnimatedWordmark />)

      const overlay = container.querySelector('img')
      expect(overlay).toHaveAttribute('src', '/brand/text-overlay-light.png')
    })
  })

  describe('reduced-motion fallback', () => {
    test('renders the static logo instead of the canvas', () => {
      mockUseReducedMotion.mockReturnValue(true)
      const { container } = render(<AnimatedWordmark />)

      expect(screen.getByTestId('animated-wordmark-static')).toBeInTheDocument()
      expect(container.querySelector('canvas')).not.toBeInTheDocument()

      const staticLogo = screen.getByAltText('GenAlphAI')
      expect(staticLogo).toHaveAttribute('src', '/brand/logo-dark.png')
    })
  })
})

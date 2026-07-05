// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { describe, test, expect } from 'vitest'
import { InfinityLoader } from './InfinityLoader'

describe('InfinityLoader', () => {
  test('renders an accessible status indicator with the infinity paths', () => {
    const { container } = render(<InfinityLoader />)

    const loader = screen.getByTestId('infinity-loader')
    expect(loader).toBeInTheDocument()
    expect(loader).toHaveAttribute('role', 'status')
    expect(loader).toHaveAttribute('aria-label', 'Research in progress')

    // Solid left loop + dotted right loop
    const paths = container.querySelectorAll('path')
    expect(paths.length).toBe(2)
    expect(paths[1]).toHaveAttribute('stroke-dasharray')

    // Flowing dots along the full path
    const dots = container.querySelectorAll('circle.gx-infinity-motion')
    expect(dots.length).toBe(3)
    expect(container.querySelectorAll('animateMotion').length).toBe(3)
  })

  test('scales width from the size prop and accepts a custom label', () => {
    const { container } = render(<InfinityLoader size={48} label="Loading data" />)

    expect(screen.getByTestId('infinity-loader')).toHaveAttribute('aria-label', 'Loading data')
    const svg = container.querySelector('svg')
    expect(svg).toHaveAttribute('height', '48')
    expect(svg).toHaveAttribute('width', '100')
  })
})

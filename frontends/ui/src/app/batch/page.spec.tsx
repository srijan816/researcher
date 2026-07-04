// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, waitFor } from '@testing-library/react'
import { describe, test, expect, vi, beforeEach } from 'vitest'
import BatchPage from './page'
import { useLayoutStore } from '@/features/layout'

const replaceMock = vi.fn()

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    replace: replaceMock,
    push: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
}))

describe('BatchPage', () => {
  beforeEach(() => {
    replaceMock.mockClear()
    useLayoutStore.setState({ rightPanel: null })
  })

  test('opens the batch-queue panel and redirects to /', async () => {
    render(<BatchPage />)

    await waitFor(() => {
      expect(useLayoutStore.getState().rightPanel).toBe('batch-queue')
      expect(replaceMock).toHaveBeenCalledWith('/')
    })
  })

  test('renders nothing (thin redirect, no duplicate UI)', () => {
    const { container } = render(<BatchPage />)

    expect(container).toBeEmptyDOMElement()
  })
})

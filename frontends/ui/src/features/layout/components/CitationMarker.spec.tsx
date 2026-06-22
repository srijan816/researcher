// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@/test-utils'
import { describe, test, expect } from 'vitest'
import { CitationMarker } from './CitationMarker'
import type { ReportCitationDetail } from '../lib/report-citations'

const FULL_DETAIL: ReportCitationDetail = {
  number: 3,
  title: 'NVIDIA Q3 Earnings',
  url: 'https://investor.nvidia.com/q3',
  sourceClass: 'first_party',
  publishedDate: '2025-11-20',
  quote: 'Record data center revenue of $14.5B in Q3, up 41% year over year.',
}

const MINIMAL_DETAIL: ReportCitationDetail = {
  number: 7,
  title: 'example.com',
  url: 'https://example.com/post',
}

describe('CitationMarker', () => {
  test('renders the inline marker without the popover initially', () => {
    render(<CitationMarker detail={FULL_DETAIL} />)

    expect(screen.getByTestId('citation-marker-3')).toHaveTextContent('[3]')
    expect(screen.queryByTestId('citation-popover-3')).not.toBeInTheDocument()
  })

  test('opens the evidence popover on click with title, badge, date, quote, and link', () => {
    render(<CitationMarker detail={FULL_DETAIL} />)

    fireEvent.click(screen.getByTestId('citation-marker-3'))

    const popover = screen.getByTestId('citation-popover-3')
    expect(popover).toHaveTextContent('NVIDIA Q3 Earnings')
    expect(popover).toHaveTextContent('First party')
    expect(popover).toHaveTextContent('Published Nov 20, 2025')
    expect(popover).toHaveTextContent('Record data center revenue')

    const link = screen.getByRole('link')
    expect(link).toHaveAttribute('href', 'https://investor.nvidia.com/q3')
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'))
  })

  test('degrades gracefully to title and URL only', () => {
    render(<CitationMarker detail={MINIMAL_DETAIL} />)

    fireEvent.click(screen.getByTestId('citation-marker-7'))

    const popover = screen.getByTestId('citation-popover-7')
    expect(popover).toHaveTextContent('example.com')
    expect(popover).not.toHaveTextContent('Published')
    expect(screen.getByRole('link')).toHaveAttribute('href', 'https://example.com/post')
  })

  test('opens on focus and closes on Escape (keyboard accessible)', () => {
    render(<CitationMarker detail={FULL_DETAIL} />)
    const marker = screen.getByTestId('citation-marker-3')

    fireEvent.focus(marker)
    expect(screen.getByTestId('citation-popover-3')).toBeInTheDocument()
    expect(marker).toHaveAttribute('aria-expanded', 'true')

    fireEvent.keyDown(marker, { key: 'Escape' })
    expect(screen.queryByTestId('citation-popover-3')).not.toBeInTheDocument()
    expect(marker).toHaveAttribute('aria-expanded', 'false')
  })

  test('opens on hover', () => {
    render(<CitationMarker detail={FULL_DETAIL} />)

    fireEvent.mouseEnter(screen.getByTestId('citation-marker-3').parentElement as HTMLElement)
    expect(screen.getByTestId('citation-popover-3')).toBeInTheDocument()
  })
})

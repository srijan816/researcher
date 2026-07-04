// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen, waitFor } from '@/test-utils'
import { vi, describe, test, expect, afterEach } from 'vitest'
import { ReportTab } from './ReportTab'

// Mock auth + report API (used by the fresh-report refetch effect)
vi.mock('@/adapters/auth', () => ({
  useAuth: () => ({
    idToken: null,
    isAuthenticated: false,
    authRequired: false,
    user: null,
    signIn: vi.fn(),
    signOut: vi.fn(),
  }),
}))

vi.mock('@/adapters/api', () => ({
  getJobReport: vi.fn(() => Promise.resolve({ has_report: false })),
}))

// Mock the chat store
vi.mock('@/features/chat', () => ({
  useChatStore: vi.fn((selector?: (s: any) => any) => {
    const state = {
      reportContent: '',
      reportContentCategory: null,
      isStreaming: false,
      currentStatus: null,
    }
    return selector ? selector(state) : state
  }),
}))

// Mock MarkdownRenderer
vi.mock('@/shared/components/MarkdownRenderer', () => ({
  MarkdownRenderer: ({ content, isStreaming }: { content: string; isStreaming?: boolean }) => (
    <div data-testid="markdown" data-streaming={isStreaming}>
      {content}
      {isStreaming && <span data-testid="streaming-indicator">Generating report...</span>}
    </div>
  ),
}))

// Mock ExportFooter
vi.mock('./ExportFooter', () => ({
  ExportFooter: () => <div data-testid="export-footer">Export Footer</div>,
}))

import { useChatStore } from '@/features/chat'

describe('ReportTab', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  test('displays empty state when no report content', () => {
    render(<ReportTab />)

    expect(screen.getByText(/report content will appear here/i)).toBeInTheDocument()
    // Icon is rendered as SVG, verify by checking the document icon is present
    expect(document.querySelector('svg')).toBeInTheDocument()
  })

  test('renders report content via MarkdownRenderer', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: '# Report Title\n\nReport content here',
        reportContentCategory: 'final_report',
        isStreaming: false,
        currentStatus: null,
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    expect(screen.getByTestId('markdown')).toHaveTextContent('# Report Title')
  })

  test('renders title when provided', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'Some content',
        reportContentCategory: 'final_report',
        isStreaming: false,
        currentStatus: null,
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    expect(screen.getByText('Some content')).toBeInTheDocument()
  })

  test('shows generating indicator when streaming and writing', () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: 'Partial content...',
        reportContentCategory: 'final_report',
        isStreaming: true,
        currentStatus: 'writing',
      }
      return selector ? selector(state) : state
    })

    render(<ReportTab />)

    // Check that MarkdownRenderer receives isStreaming prop and shows indicator
    expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument()
    expect(screen.getByText('Generating report...')).toBeInTheDocument()
  })

  test('renders children when provided', () => {
    render(
      <ReportTab>
        <div>Custom content</div>
      </ReportTab>
    )

    expect(screen.getByText('Custom content')).toBeInTheDocument()
    expect(screen.queryByTestId('markdown')).not.toBeInTheDocument()
  })

  test('always renders export footer', () => {
    render(<ReportTab />)

    expect(screen.getByTestId('export-footer')).toBeInTheDocument()
  })

  test('prepares Kokoro narration in the native audio element', async () => {
    vi.mocked(useChatStore).mockImplementation((selector?: (s: any) => any) => {
      const state = {
        reportContent: '# Report Title\n\nReport content here',
        reportContentCategory: 'final_report',
        isStreaming: false,
        currentStatus: null,
      }
      return selector ? selector(state) : state
    })
    const play = vi.fn().mockResolvedValue(undefined)
    vi.spyOn(window.HTMLMediaElement.prototype, 'play').mockImplementation(play)
    vi.spyOn(window.HTMLMediaElement.prototype, 'pause').mockImplementation(vi.fn())
    vi.spyOn(window.HTMLMediaElement.prototype, 'load').mockImplementation(vi.fn())
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new Blob(['audio']), { status: 200 })))
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:test-audio'),
      revokeObjectURL: vi.fn(),
    })

    render(<ReportTab />)

    await waitFor(() =>
      expect(screen.getByLabelText('Kokoro narration audio')).toHaveAttribute('src', 'blob:test-audio')
    )
    fireEvent.click(screen.getByLabelText('Prepare or play narration'))
    await waitFor(() => expect(play).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: 'Play Part 1/1' })).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledWith(
      '/api/tts/kokoro',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"voice":"onyx"'),
      })
    )
  })
})

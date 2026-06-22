// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ReportExportMenu Component
 *
 * Export menu for the report header:
 *  - Download .md   — backend report endpoint (?format=md) via the Next proxy
 *  - Download .docx — same endpoint with ?format=docx (clear message when the
 *    backend does not support DOCX yet / returns non-200)
 *  - Print / Save as PDF — print-friendly report-only printing via
 *    window.print() and @media print rules in globals.css
 *
 * Files are named after the research (conversation) title.
 */

'use client'

import { type FC, useCallback, useState } from 'react'
import { Banner, Button, Flex, Popover, Text } from '@/adapters/ui'
import { ChevronDown, Document, Download } from '@/adapters/ui/icons'
import { useChatStore } from '@/features/chat'
import { sanitizeFilename } from '@/utils/sanitize-filename'
import { downloadAsMarkdown } from '@/utils/download-as-markdown'

type ExportFormat = 'md' | 'docx'

const FORMAT_LABELS: Record<ExportFormat, string> = {
  md: 'Markdown',
  docx: 'DOCX',
}

/** CSS class toggled on <body> while printing so @media print rules apply */
export const REPORT_PRINTING_CLASS = 'report-printing'

const triggerBlobDownload = (blob: Blob, filename: string): void => {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.style.display = 'none'
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  setTimeout(() => {
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }, 100)
}

/**
 * Export menu with backend-format downloads and print support.
 */
export const ReportExportMenu: FC = () => {
  const reportContent = useChatStore((s) => s.reportContent)
  const jobId = useChatStore((s) => s.deepResearchJobId)
  const conversationTitle = useChatStore((s) => s.currentConversation?.title)

  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingFormat, setPendingFormat] = useState<ExportFormat | null>(null)

  const reportContentStr = typeof reportContent === 'string' ? reportContent : ''
  const hasContent = reportContentStr.trim().length > 0
  const baseName = sanitizeFilename(conversationTitle || 'research-report')

  const handleDownload = useCallback(
    async (format: ExportFormat) => {
      setIsMenuOpen(false)
      setError(null)

      // Markdown fallback: no job id (e.g. restored session) — export store content directly
      if (!jobId && format === 'md' && hasContent) {
        const result = downloadAsMarkdown(reportContentStr, conversationTitle ?? undefined)
        if (!result.success && result.error) setError(result.error)
        return
      }

      if (!jobId) {
        setError(`Cannot export ${FORMAT_LABELS[format]}: no research job is associated with this report.`)
        return
      }

      setPendingFormat(format)
      try {
        const response = await fetch(`/api/jobs/async/job/${jobId}/report?format=${format}`)
        const contentType = response.headers.get('content-type') || ''

        if (!response.ok) {
          throw new Error(
            `${FORMAT_LABELS[format]} export failed (${response.status}). ` +
              (format === 'docx'
                ? 'DOCX export may not be available on this server yet.'
                : 'Please try again.')
          )
        }

        // The backend returns JSON (job progress payload) instead of file bytes
        // when the report is not ready or the format is not supported.
        if (contentType.includes('application/json')) {
          if (format === 'docx') {
            throw new Error('DOCX export is not available on this server yet. Try Markdown or PDF instead.')
          }
          throw new Error('The report is not ready to export yet.')
        }

        const blob = await response.blob()
        triggerBlobDownload(blob, `${baseName}.${format}`)
      } catch (downloadError) {
        setError(
          downloadError instanceof Error
            ? downloadError.message
            : `Failed to export ${FORMAT_LABELS[format]}.`
        )
      } finally {
        setPendingFormat(null)
      }
    },
    [jobId, hasContent, reportContentStr, conversationTitle, baseName]
  )

  const handlePrint = useCallback(() => {
    setIsMenuOpen(false)
    setError(null)

    const cleanup = (): void => {
      document.body.classList.remove(REPORT_PRINTING_CLASS)
    }

    document.body.classList.add(REPORT_PRINTING_CLASS)
    window.addEventListener('afterprint', cleanup, { once: true })

    // Give the browser one frame to apply print classes before opening the dialog
    setTimeout(() => {
      window.print()
      // Safety: afterprint is unreliable in some webviews
      setTimeout(cleanup, 1000)
    }, 50)
  }, [])

  return (
    <Flex direction="col" className="min-w-0">
      {error && (
        <Banner kind="inline" status="error" onClose={() => setError(null)} className="mb-2">
          {error}
        </Banner>
      )}
      <Flex justify="end">
        <Popover
          open={isMenuOpen}
          onOpenChange={setIsMenuOpen}
          side="bottom"
          align="end"
          slotContent={
            <Flex direction="col" gap="1" className="min-w-48 p-1" role="menu" aria-label="Export report options">
              <Button
                kind="tertiary"
                size="small"
                role="menuitem"
                onClick={() => void handleDownload('md')}
                disabled={!hasContent || pendingFormat !== null}
                aria-label="Download report as Markdown"
                data-testid="export-download-md"
              >
                <Download className="h-4 w-4 mr-2" aria-hidden="true" />
                <Text kind="body/regular/sm">Download .md</Text>
              </Button>
              <Button
                kind="tertiary"
                size="small"
                role="menuitem"
                onClick={() => void handleDownload('docx')}
                disabled={!hasContent || pendingFormat !== null}
                aria-label="Download report as Word document"
                data-testid="export-download-docx"
              >
                <Download className="h-4 w-4 mr-2" aria-hidden="true" />
                <Text kind="body/regular/sm">
                  {pendingFormat === 'docx' ? 'Preparing .docx...' : 'Download .docx'}
                </Text>
              </Button>
              <Button
                kind="tertiary"
                size="small"
                role="menuitem"
                onClick={handlePrint}
                disabled={!hasContent}
                aria-label="Print or save report as PDF"
                data-testid="export-print"
              >
                <Document className="h-4 w-4 mr-2" aria-hidden="true" />
                <Text kind="body/regular/sm">Print / Save as PDF</Text>
              </Button>
            </Flex>
          }
        >
          <Button
            kind="tertiary"
            size="small"
            disabled={!hasContent}
            aria-label="Export report"
            aria-haspopup="menu"
            aria-expanded={isMenuOpen}
            title={hasContent ? 'Export report' : 'No report to export'}
            data-testid="report-export-menu-trigger"
          >
            <Download className="h-4 w-4 sm:mr-2" aria-hidden="true" />
            <span className="hidden sm:inline">Export</span>
            <ChevronDown className="ml-1 h-3 w-3" aria-hidden="true" />
          </Button>
        </Popover>
      </Flex>
    </Flex>
  )
}

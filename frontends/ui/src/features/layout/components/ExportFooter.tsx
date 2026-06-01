// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ExportFooter Component
 *
 * Footer with export actions for reports.
 * Provides buttons to export content as Markdown or PDF.
 */

'use client'

import { type FC, useCallback, useState } from 'react'
import { Banner, Flex, Button } from '@/adapters/ui'
import { useChatStore, useIsCurrentSessionBusy } from '@/features/chat'
import { downloadAsMarkdown } from '@/utils/download-as-markdown'
import { useDownloadPdfRoute } from '@/hooks/use-download-pdf'
import { Copy, Download } from '@/adapters/ui/icons'

interface ExportFooterProps {
  /** Whether to disable export buttons (e.g., when no content) */
  disabled?: boolean
}

const copyText = async (value: string): Promise<void> => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value)
    return
  }

  const textarea = document.createElement('textarea')
  textarea.value = value
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand('copy')
  document.body.removeChild(textarea)
}

/**
 * Export footer with Markdown and PDF export buttons.
 * Only renders when there's content to export.
 */
export const ExportFooter: FC<ExportFooterProps> = ({ disabled }) => {
  const reportContent = useChatStore((state) => state.reportContent)
  const conversationTitle = useChatStore((state) => state.currentConversation?.title)
  const { downloadPdf, isLoading: isPdfLoading, error: pdfError, clearError: clearPdfError } = useDownloadPdfRoute()
  const [mdError, setMdError] = useState<string | null>(null)
  const [copyNotice, setCopyNotice] = useState<string | null>(null)

  // Defensive check: ensure reportContent is a string before calling trim()
  const reportContentStr = typeof reportContent === 'string' ? reportContent : ''
  const hasContent = reportContentStr.trim().length > 0

  // Uses centralized hook that checks BOTH ephemeral AND persisted state.
  // This survives page refresh: even if SSE ephemeral flags are lost,
  // the hook derives busy state from persisted message history.
  const isDeepResearchInProgress = useIsCurrentSessionBusy()

  const isExportDisabled = disabled || !hasContent || isDeepResearchInProgress

  const tooltipContent = isDeepResearchInProgress
    ? 'Export will be available when research is complete'
    : hasContent
      ? 'Export report'
      : 'No content to export'

  const handleExportMarkdown = useCallback(() => {
    if (isExportDisabled) return
    setMdError(null)
    const result = downloadAsMarkdown(reportContentStr, conversationTitle ?? undefined)
    if (!result.success && result.error) {
      setMdError(result.error)
    }
  }, [isExportDisabled, reportContentStr, conversationTitle])

  const handleExportPDF = useCallback(() => {
    if (isExportDisabled || isPdfLoading) return
    downloadPdf(reportContentStr, conversationTitle ?? undefined)
  }, [isExportDisabled, isPdfLoading, reportContentStr, downloadPdf, conversationTitle])

  const handleCopyReport = useCallback(async () => {
    if (isExportDisabled) return
    setMdError(null)
    setCopyNotice(null)
    try {
      await copyText(reportContentStr)
      setCopyNotice('Report copied.')
      window.setTimeout(() => setCopyNotice(null), 1800)
    } catch (error) {
      setMdError(error instanceof Error ? error.message : 'Unable to copy report')
    }
  }, [isExportDisabled, reportContentStr])

  const exportError = mdError || pdfError
  const clearExportError = useCallback(() => {
    setMdError(null)
    clearPdfError()
  }, [clearPdfError])

  return (
    <Flex direction="col" className="border-base shrink-0 border-t">
      {exportError && (
        <Banner kind="inline" status="error" onClose={clearExportError} className="mx-4 mt-3">
          {exportError}
        </Banner>
      )}
      {copyNotice && (
        <Banner kind="inline" status="success" onClose={() => setCopyNotice(null)} className="mx-4 mt-3">
          {copyNotice}
        </Banner>
      )}
      <Flex align="center" justify="end" gap="2" className="px-4 py-3">
        <Button
          kind="tertiary"
          size="small"
          onClick={handleCopyReport}
          disabled={isExportDisabled}
          aria-label={isExportDisabled ? `Copy report (${tooltipContent})` : 'Copy report'}
          title={isExportDisabled ? tooltipContent : 'Copy report'}
        >
          <Copy />
          Copy
        </Button>
        <Button
          kind="tertiary"
          size="small"
          onClick={handleExportMarkdown}
          disabled={isExportDisabled}
          aria-label={isExportDisabled ? `Export as Markdown (${tooltipContent})` : 'Export as Markdown'}
          title={tooltipContent}
        >
          <Download />
          Markdown
        </Button>
        <Button
          kind="primary"
          color="brand"
          size="small"
          onClick={handleExportPDF}
          disabled={isExportDisabled || isPdfLoading}
          aria-label={
            isPdfLoading
              ? 'Generating PDF...'
              : isExportDisabled
                ? `Export as PDF (${tooltipContent})`
                : 'Export as PDF'
          }
          title={isPdfLoading ? 'Generating PDF...' : tooltipContent}
        >
          <Download />
          {isPdfLoading ? 'Generating...' : 'PDF'}
        </Button>
      </Flex>
    </Flex>
  )
}

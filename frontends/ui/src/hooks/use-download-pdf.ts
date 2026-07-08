// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useState } from 'react'
import { sanitizeFilename } from '@/utils/sanitize-filename'

/** Matches markdown images pointing at same-origin API assets: ![alt](/api/... or /v1/...) */
const SAME_ORIGIN_IMAGE_RE = /!\[([^\]]*)\]\((\/(?:api|v1)\/[^)\s]+)\)/g

const blobToDataUrl = (blob: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read image blob'))
    reader.readAsDataURL(blob)
  })

/**
 * Replace same-origin markdown image URLs with data URLs so the server-side
 * PDF renderer (which has neither the session cookies nor the origin) can
 * embed them. Images that fail to fetch are left untouched.
 */
export const inlineSameOriginImages = async (markdown: string): Promise<string> => {
  const matches = [...markdown.matchAll(SAME_ORIGIN_IMAGE_RE)]
  if (matches.length === 0) return markdown

  const uniqueUrls = [...new Set(matches.map((m) => m[2]))]
  const dataUrls = new Map<string, string>()

  await Promise.all(
    uniqueUrls.map(async (url) => {
      try {
        const response = await fetch(url, { credentials: 'same-origin' })
        if (!response.ok) return
        const blob = await response.blob()
        if (!blob.type.startsWith('image/')) return
        dataUrls.set(url, await blobToDataUrl(blob))
      } catch {
        // Leave the original URL in place; the PDF simply omits this image
      }
    })
  )

  if (dataUrls.size === 0) return markdown
  return markdown.replace(SAME_ORIGIN_IMAGE_RE, (full, alt: string, url: string) => {
    const dataUrl = dataUrls.get(url)
    return dataUrl ? `![${alt}](${dataUrl})` : full
  })
}

export const useDownloadPdfRoute = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const downloadPdf = async (markdown: string, filename?: string) => {
    setIsLoading(true)
    setError(null)

    try {
      // Inline same-origin report images so they survive server-side rendering
      const markdownWithImages = await inlineSameOriginImages(markdown)

      const response = await fetch('/api/generate-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ markdown: markdownWithImages }),
      })

      if (!response.ok) {
        let message = response.statusText
        try {
          const body = (await response.json()) as { error?: unknown; details?: unknown }
          const serverMessage = [body.error, body.details]
            .filter((value): value is string => typeof value === 'string' && value.length > 0)
            .join(': ')
          if (serverMessage) message = serverMessage
        } catch {
          // Fall back to the HTTP status text when the server does not return JSON.
        }
        throw new Error(`Failed to generate PDF: ${message}`)
      }

      const blob = await response.blob()

      const baseName = filename
        ? sanitizeFilename(filename)
        : `report-${new Date().toISOString().slice(0, 10)}`

      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${baseName}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)

      URL.revokeObjectURL(url)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to download PDF'
      setError(errorMessage)
      console.error('PDF download error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return {
    downloadPdf,
    isLoading,
    error,
    clearError,
  }
}

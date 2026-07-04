// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CitationCard Component
 *
 * Non-collapsible card displaying a single citation/source as a clickable link.
 * Shows title/domain and full URL.
 *
 * SSE Events:
 * - artifact.update type: "citation_source" - Referenced (discovered during search)
 * - artifact.update type: "citation_use" - Cited (actually used in report)
 */

'use client'

import { type FC } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { Check } from '@/adapters/ui/icons'
import type { CitationSource } from '@/features/chat/types'

interface CitationCardProps {
  /** Citation information */
  citation: CitationSource
}

/**
 * Format timestamp for display
 */
const formatTime = (date: Date | string): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date
  return dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/**
 * Extract domain from URL for display
 */
const getDomain = (url: string): string => {
  try {
    const urlObj = new URL(url)
    return urlObj.hostname.replace('www.', '')
  } catch {
    return url.substring(0, 30)
  }
}

/**
 * Compact source card: favicon, mono-caps domain, clamped title/content,
 * hover lift with blue border.
 */
export const CitationCard: FC<CitationCardProps> = ({ citation }) => {
  const domain = getDomain(citation.url)
  return (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="ar-card-lift block h-full rounded-xl border border-base bg-surface-raised p-3 transition-colors hover:border-[#F43F5E]"
    >
      <Flex direction="col" gap="2" className="h-full min-w-0">
        {/* Favicon + domain + timestamp */}
        <Flex align="center" gap="2" className="min-w-0">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=32`}
            alt=""
            width={16}
            height={16}
            loading="lazy"
            className="h-4 w-4 shrink-0 rounded-sm"
            onError={(event) => {
              event.currentTarget.style.display = 'none'
            }}
          />
          <span className="gx-mono min-w-0 flex-1 truncate uppercase">{domain}</span>
          {citation.isCited && (
            <span
              className="shrink-0"
              style={{ color: 'var(--text-color-brand)' }}
              aria-label="Cited in report"
            >
              <Check className="h-3.5 w-3.5" />
            </span>
          )}
          <Text kind="body/regular/xs" className="text-subtle shrink-0">
            {formatTime(citation.timestamp)}
          </Text>
        </Flex>

        {/* Title / content, 2-line clamp */}
        <Text kind="body/regular/sm" className="text-secondary line-clamp-2 break-all">
          {citation.content?.trim() || citation.url}
        </Text>
      </Flex>
    </a>
  )
}

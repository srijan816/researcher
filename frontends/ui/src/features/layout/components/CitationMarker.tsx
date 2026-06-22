// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CitationMarker Component
 *
 * Interactive inline [n] citation marker rendered inside the report markdown.
 * Hover (desktop) or tap (mobile) opens an evidence popover with the source
 * title, clickable URL, source-class badge, published date, and a supporting
 * extract when available. Degrades gracefully to title + URL only.
 *
 * Accessibility: the marker is a real button (focusable), the popover opens on
 * focus, and Escape closes it. The popover is linked via aria-describedby.
 */

'use client'

import { type FC, type KeyboardEvent, useCallback, useEffect, useId, useRef, useState } from 'react'
import { Text } from '@/adapters/ui'
import {
  formatPublishedDate,
  formatSourceClassLabel,
  getCitationDomain,
  type ReportCitationDetail,
} from '../lib/report-citations'

interface CitationMarkerProps {
  /** Merged citation detail for this marker */
  detail: ReportCitationDetail
}

const HOVER_CLOSE_DELAY_MS = 150

/**
 * Inline citation marker with an evidence popover.
 */
export const CitationMarker: FC<CitationMarkerProps> = ({ detail }) => {
  const [isOpen, setIsOpen] = useState(false)
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const containerRef = useRef<HTMLSpanElement | null>(null)
  const popoverId = useId()

  const cancelClose = useCallback(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current)
      closeTimerRef.current = null
    }
  }, [])

  const scheduleClose = useCallback(() => {
    cancelClose()
    closeTimerRef.current = setTimeout(() => setIsOpen(false), HOVER_CLOSE_DELAY_MS)
  }, [cancelClose])

  useEffect(() => () => cancelClose(), [cancelClose])

  // Close on outside tap/click (mobile tap-to-open support)
  useEffect(() => {
    if (!isOpen) return
    const handlePointerDown = (event: PointerEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('pointerdown', handlePointerDown)
    return () => document.removeEventListener('pointerdown', handlePointerDown)
  }, [isOpen])

  const handleKeyDown = useCallback((event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') {
      event.stopPropagation()
      setIsOpen(false)
    }
  }, [])

  const publishedLabel = formatPublishedDate(detail.publishedDate)

  return (
    <span
      ref={containerRef}
      className="relative inline-block"
      onMouseEnter={() => {
        cancelClose()
        setIsOpen(true)
      }}
      onMouseLeave={scheduleClose}
      onKeyDown={handleKeyDown}
    >
      <button
        type="button"
        className="citation-marker mx-0.5 inline-flex cursor-pointer items-center rounded border border-base bg-surface-raised px-1 align-super font-mono text-[0.7em] leading-snug text-accent-primary transition-colors hover:border-brand focus-visible:border-brand"
        aria-label={`Citation ${detail.number}: ${detail.title}`}
        aria-expanded={isOpen}
        aria-describedby={isOpen ? popoverId : undefined}
        onClick={() => setIsOpen((open) => !open)}
        onFocus={() => {
          cancelClose()
          setIsOpen(true)
        }}
        onBlur={(event) => {
          // Keep open while focus moves into the popover (e.g. to the source link)
          if (!containerRef.current?.contains(event.relatedTarget as Node)) {
            setIsOpen(false)
          }
        }}
        data-testid={`citation-marker-${detail.number}`}
      >
        [{detail.number}]
      </button>

      {isOpen && (
        <span
          id={popoverId}
          role="tooltip"
          className="absolute bottom-full left-1/2 z-50 mb-1.5 block w-72 max-w-[80vw] -translate-x-1/2 rounded-lg border border-base bg-surface-raised p-3 text-left shadow-lg"
          onMouseEnter={cancelClose}
          onMouseLeave={scheduleClose}
          data-testid={`citation-popover-${detail.number}`}
        >
          <span className="mb-1 flex items-start justify-between gap-2">
            <Text kind="label/semibold/sm" className="block min-w-0 text-primary">
              {detail.title}
            </Text>
            {detail.sourceClass && (
              <Text
                kind="body/regular/xs"
                className="block shrink-0 rounded border border-base bg-surface-sunken px-1.5 py-0.5 text-subtle"
              >
                {formatSourceClassLabel(detail.sourceClass)}
              </Text>
            )}
          </span>

          {publishedLabel && (
            <Text kind="body/regular/xs" className="mb-1 block text-subtle">
              Published {publishedLabel}
            </Text>
          )}

          {detail.quote && (
            <Text
              kind="body/regular/xs"
              className="mb-2 block border-l-2 border-base pl-2 italic text-secondary [display:-webkit-box] [-webkit-box-orient:vertical] [-webkit-line-clamp:4] overflow-hidden"
            >
              {detail.quote}
            </Text>
          )}

          {detail.url && (
            <a
              href={detail.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block truncate text-xs text-accent-primary underline-offset-2 hover:underline"
              title={detail.url}
            >
              {getCitationDomain(detail.url)} ↗
            </a>
          )}
        </span>
      )}
    </span>
  )
}

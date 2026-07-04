// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * ReportTab Component
 *
 * Displays research output in two visual modes:
 *   1. Research Notes (intermediate) -- preview styling with a header badge
 *   2. Final Report -- full-width rendered markdown with export footer
 *
 * Shows streaming indicator when report is being generated.
 * Includes export footer for Markdown and PDF export (final report only).
 */

'use client'

import { type FC, type MouseEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Flex, Text, Button, Spinner } from '@/adapters/ui'
import { useShallow } from 'zustand/react/shallow'
import { getJobReport } from '@/adapters/api'
import { useAuth } from '@/adapters/auth'
import { ArrowLeft, ArrowRight, Document, Pause, Play, Stop, Volume } from '@/adapters/ui/icons'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import { useChatStore } from '@/features/chat'
import { ExportFooter } from './ExportFooter'
import { CitationMarker } from './CitationMarker'
import { ReportExportMenu } from './ReportExportMenu'
import { VerificationStatsStrip } from './VerificationStatsStrip'
import { buildCitationDetails, linkifyCitationMarkers, type ReportCitationDetail } from '../lib/report-citations'

interface ReportTabProps {
  /** Optional custom content to display instead of store content */
  children?: ReactNode
}

interface PlaylistItem {
  id: string
  title: string
  url: string
  blob: Blob
}

const markdownToSpeechText = (content: string): string =>
  content
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/[*_~>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

const formatAudioTime = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const DEFAULT_NARRATION_VOICE = 'onyx'
const NARRATION_CHUNK_TARGET_CHARS = 800
const NARRATION_CHUNK_MAX_CHARS = 1000
// Number of narration chunks to synthesise concurrently. Matches the Kokoro worker pool
// size so parallel requests are served in parallel rather than queued.
const NARRATION_FETCH_CONCURRENCY = 3

const splitLongSpeechPart = (value: string, maxChars: number): string[] => {
  const chunks: string[] = []
  let current = ''

  for (const word of value.split(/\s+/).filter(Boolean)) {
    if (!current) {
      current = word
      continue
    }
    if (current.length + word.length + 1 <= maxChars) {
      current = `${current} ${word}`
      continue
    }
    chunks.push(current)
    current = word
  }

  if (current) chunks.push(current)
  return chunks
}

const splitSpeechIntoNarrationChunks = (text: string): string[] => {
  const parts = text
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter(Boolean)
  const chunks: string[] = []
  let current = ''

  const pushCurrent = () => {
    if (!current.trim()) return
    chunks.push(current.trim())
    current = ''
  }

  for (const part of parts.length ? parts : [text.trim()]) {
    if (part.length > NARRATION_CHUNK_MAX_CHARS) {
      pushCurrent()
      chunks.push(...splitLongSpeechPart(part, NARRATION_CHUNK_TARGET_CHARS))
      continue
    }

    if (!current) {
      current = part
    } else if (current.length + part.length + 1 <= NARRATION_CHUNK_TARGET_CHARS) {
      current = `${current} ${part}`
    } else {
      pushCurrent()
      current = part
    }
  }

  pushCurrent()
  return chunks
}

/**
 * Report tab content - displays research output.
 * Subscribes to chat store for report content, category, and streaming state.
 * Renders research notes with a subtle preview treatment and the final report at full prominence.
 */
export const ReportTab: FC<ReportTabProps> = ({ children }) => {
  const { reportContent, reportContentCategory, isStreaming, currentStatus, deepResearchCitations, deepResearchJobId } =
    useChatStore(useShallow((s) => ({
      reportContent: s.reportContent,
      reportContentCategory: s.reportContentCategory,
      isStreaming: s.isStreaming,
      currentStatus: s.currentStatus,
      deepResearchCitations: s.deepResearchCitations,
      deepResearchJobId: s.deepResearchJobId,
    })))
  const setReportContent = useChatStore((s) => s.setReportContent)
  const { idToken } = useAuth()
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [isPreparingAudio, setIsPreparingAudio] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [audioNotice, setAudioNotice] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playlist, setPlaylist] = useState<PlaylistItem[]>([])
  const [narrationEnabled, setNarrationEnabled] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioUrlRef = useRef<string | null>(null)
  const playlistRef = useRef<PlaylistItem[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const currentChunkIndexRef = useRef(0)
  const isPreparingAudioRef = useRef(false)
  const waitingForNextChunkRef = useRef(false)
  const narrationSignatureRef = useRef<string | null>(null)
  const refreshedReportJobRef = useRef<string | null>(null)

  const reportContentStr = typeof reportContent === 'string' ? reportContent : ''
  const isEmpty = !reportContentStr.trim()
  const isGeneratingReport = isStreaming && currentStatus === 'writing'
  const isResearchNotes = reportContentCategory === 'research_notes'
  const canReadAloud = !isEmpty && !isResearchNotes
  const isFinalReport = !isEmpty && !isResearchNotes

  // Evidence-linked citations: parse the references section, merge with
  // citation event metadata, and rewrite inline [n] markers into popover links.
  const citationDetails = useMemo<Map<number, ReportCitationDetail>>(
    () =>
      isFinalReport
        ? buildCitationDetails(reportContentStr, deepResearchCitations ?? [])
        : new Map<number, ReportCitationDetail>(),
    [isFinalReport, reportContentStr, deepResearchCitations]
  )

  const linkedReportContent = useMemo(() => {
    if (!isFinalReport || citationDetails.size === 0 || isGeneratingReport) return reportContentStr
    return linkifyCitationMarkers(reportContentStr, new Set(citationDetails.keys()))
  }, [isFinalReport, citationDetails, isGeneratingReport, reportContentStr])

  const renderCitationLink = useCallback(
    (citationNumber: number, children: ReactNode): ReactNode => {
      const detail = citationDetails.get(citationNumber)
      if (!detail) return <>{children}</>
      return <CitationMarker detail={detail} />
    },
    [citationDetails]
  )

  // Refresh the report from the backend when the tab shows a finished report.
  // The store copy usually comes from the SSE stream, which is emitted before
  // the backend embeds report images — the fresh GET .../report response is
  // the authoritative copy (it includes ![caption](/api/...) image markdown).
  useEffect(() => {
    if (!isFinalReport || isStreaming || !deepResearchJobId) return
    if (refreshedReportJobRef.current === deepResearchJobId) return
    refreshedReportJobRef.current = deepResearchJobId

    let cancelled = false
    getJobReport(deepResearchJobId, idToken || undefined)
      .then((response) => {
        if (cancelled || !response.has_report || !response.report) return
        const current = useChatStore.getState().reportContent
        if (response.report !== current) {
          setReportContent(response.report, 'final_report')
        }
      })
      .catch((error) => {
        // Allow a retry on the next open if the refresh failed
        if (refreshedReportJobRef.current === deepResearchJobId) {
          refreshedReportJobRef.current = null
        }
        console.warn('Failed to refresh report content:', error)
      })

    return () => {
      cancelled = true
    }
  }, [isFinalReport, isStreaming, deepResearchJobId, idToken, setReportContent])

  useEffect(() => {
    audioUrlRef.current = audioUrl
  }, [audioUrl])

  useEffect(() => {
    playlistRef.current = playlist
  }, [playlist])

  useEffect(() => {
    isPreparingAudioRef.current = isPreparingAudio
  }, [isPreparingAudio])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      audioRef.current?.pause()
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
      playlistRef.current.forEach((item) => URL.revokeObjectURL(item.url))
    }
  }, [])

  useEffect(() => {
    setNarrationEnabled(false)
    setAudioError(null)
    setAudioNotice(null)
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      setAudioUrl(null)
    }
    playlistRef.current.forEach((item) => URL.revokeObjectURL(item.url))
    playlistRef.current = []
    setPlaylist([])
    setCurrentTime(0)
    setDuration(0)
    currentChunkIndexRef.current = 0
    waitingForNextChunkRef.current = false
    narrationSignatureRef.current = null
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.removeAttribute('src')
      audioRef.current.load()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- reset cached audio only when the report text changes
  }, [reportContentStr])

  const setAudioElement = useCallback((audio: HTMLAudioElement | null) => {
    audioRef.current = audio
  }, [])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const handleLoadedMetadata = () => {
      setDuration(Number.isFinite(audio.duration) ? audio.duration : 0)
    }
    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime || 0)
    }
    const handleEnded = () => {
      const nextIndex = currentChunkIndexRef.current + 1
      const nextItem = playlistRef.current[nextIndex]

      if (!nextItem) {
        waitingForNextChunkRef.current = isPreparingAudioRef.current
        setCurrentTime(0)
        if (isPreparingAudioRef.current) setAudioNotice('Preparing the next narration part.')
        return
      }

      waitingForNextChunkRef.current = false
      currentChunkIndexRef.current = nextIndex
      setAudioUrl(nextItem.url)
      audio.src = nextItem.url
      audio.load()
      void audio.play().catch((error) => {
        const isPermissionBlock = error instanceof DOMException && error.name === 'NotAllowedError'
        if (isPermissionBlock) {
          setAudioNotice('Next narration part is ready. Use the playback controls.')
          return
        }
        setAudioError(error instanceof Error ? error.message : 'Unable to continue narration')
      })
    }

    audio.addEventListener('loadedmetadata', handleLoadedMetadata)
    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata)
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [audioUrl])

  const guardClick = useCallback(
    (handler: () => void | Promise<void>) =>
      (event: MouseEvent<HTMLElement>) => {
        event.preventDefault()
        event.stopPropagation()
        void handler()
      },
    []
  )

  const prepareNarration = useCallback(async ({ autoplay }: { autoplay: boolean }) => {
    if (!canReadAloud) return

    const speechText = markdownToSpeechText(reportContentStr)
    const speechChunks = splitSpeechIntoNarrationChunks(speechText)
    if (speechChunks.length === 0) return

    const signature = `${reportContentStr.length}:${reportContentStr.slice(0, 80)}`
    if (isPreparingAudioRef.current || narrationSignatureRef.current === signature) return

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    currentChunkIndexRef.current = 0
    waitingForNextChunkRef.current = false
    narrationSignatureRef.current = signature
    isPreparingAudioRef.current = true
    setIsPreparingAudio(true)
    setAudioError(null)
    setAudioNotice(`Preparing narration part 1 of ${speechChunks.length}.`)
    playlistRef.current.forEach((item) => URL.revokeObjectURL(item.url))
    playlistRef.current = []
    setPlaylist([])
    setAudioUrl(null)
    setCurrentTime(0)
    setDuration(0)

    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.removeAttribute('src')
      audioRef.current.load()
    }

    try {
      // Fan out chunk synthesis with bounded concurrency, committing results in index order
      // so playlist ordering + gapless playback stay unchanged. Chunk 0 still begins
      // playback the moment it arrives.
      const total = speechChunks.length
      const blobs: (Blob | null)[] = new Array(total).fill(null)
      const slotDone: (() => void)[] = []
      const slotPromise: Promise<void>[] = speechChunks.map(
        (_, i) =>
          new Promise<void>((resolve) => {
            slotDone[i] = resolve
          })
      )
      let firstError: Error | null = null
      let launched = 0

      const fetchAt = async (index: number) => {
        setAudioNotice(`Preparing narration part ${Math.min(index + 1, total)} of ${total}.`)
        try {
          const response = await fetch('/api/tts/kokoro', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: speechChunks[index], voice: DEFAULT_NARRATION_VOICE, format: 'mp3' }),
            signal: controller.signal,
          })

          if (!response.ok) {
            const detail = await response.json().catch(() => null)
            if (!firstError) {
              firstError = new Error(typeof detail?.error === 'string' ? detail.error : 'Kokoro is not available')
            }
          } else {
            blobs[index] = await response.blob()
          }
        } catch (error) {
          if (!controller.signal.aborted && !firstError) {
            firstError = error instanceof Error ? error : new Error('Narration generation failed')
          }
        } finally {
          slotDone[index]()
        }
      }

      const worker = async () => {
        while (true) {
          const index = launched++
          if (index >= total) return
          await fetchAt(index)
        }
      }

      const width = Math.min(NARRATION_FETCH_CONCURRENCY, total)
      const inflight: Promise<void>[] = []
      for (let w = 0; w < width; w++) inflight.push(worker())

      for (let index = 0; index < total; index++) {
        await slotPromise[index]
        if (firstError) throw firstError

        const blob = blobs[index]
        if (!blob) throw new Error('Narration generation failed')

        const nextUrl = URL.createObjectURL(blob)
        const nextItem: PlaylistItem = {
          id: `${signature}-${index}`,
          title: `Part ${index + 1}/${total}`,
          url: nextUrl,
          blob,
        }

        playlistRef.current = [...playlistRef.current, nextItem]
        setPlaylist(playlistRef.current)

        if (index === 0) {
          setAudioUrl(nextUrl)
          const audio = audioRef.current
          if (audio) {
            audio.src = nextUrl
            audio.load()
            if (autoplay) {
              await audio.play().catch((error) => {
                const isPermissionBlock = error instanceof DOMException && error.name === 'NotAllowedError'
                if (isPermissionBlock) {
                  setAudioNotice('Narration is ready. Use the playback controls.')
                  return
                }
                throw error
              })
            }
          }
        } else if (waitingForNextChunkRef.current && currentChunkIndexRef.current + 1 === index) {
          waitingForNextChunkRef.current = false
          currentChunkIndexRef.current = index
          setAudioUrl(nextUrl)
          const audio = audioRef.current
          if (audio) {
            audio.src = nextUrl
            audio.load()
            await audio.play().catch((error) => {
              const isPermissionBlock = error instanceof DOMException && error.name === 'NotAllowedError'
              if (isPermissionBlock) {
                setAudioNotice('Next narration part is ready. Use the playback controls.')
                return
              }
              throw error
            })
          }
        }
      }

      await Promise.all(inflight)
      if (firstError) throw firstError

      setAudioNotice(`Narration ready: ${total} parts.`)
    } catch (error) {
      narrationSignatureRef.current = null
      if (controller.signal.aborted) return
      setAudioError(error instanceof Error ? error.message : 'Unable to read report aloud')
    } finally {
      if (!controller.signal.aborted) {
        setIsPreparingAudio(false)
        isPreparingAudioRef.current = false
      }
    }
  }, [canReadAloud, reportContentStr])

  // Narration is opt-in (default off). It only starts when the user explicitly enables it.
  const handleEnableNarration = useCallback(() => {
    if (!canReadAloud) return
    setNarrationEnabled(true)
    void prepareNarration({ autoplay: false })
  }, [canReadAloud, prepareNarration])

  const handleReadAloud = useCallback(async () => {
    if (!canReadAloud) return

    if (audioUrl && audioRef.current) {
      setAudioError(null)
      setAudioNotice(null)
      await audioRef.current.play().catch((error) => {
        const isPermissionBlock = error instanceof DOMException && error.name === 'NotAllowedError'
        if (isPermissionBlock) {
          setAudioNotice('Audio is ready. Use the playback controls.')
          return
        }
        throw error
      })
      return
    }

    await prepareNarration({ autoplay: true })
  }, [audioUrl, canReadAloud, prepareNarration])

  const handlePause = useCallback(() => {
    audioRef.current?.pause()
  }, [])

  const handleStop = useCallback(() => {
    abortRef.current?.abort()
    narrationSignatureRef.current = null
    isPreparingAudioRef.current = false
    setIsPreparingAudio(false)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    setCurrentTime(0)
  }, [])

  const handleSkip = useCallback((seconds: number) => {
    const audio = audioRef.current
    if (!audio) return
    const nextTime = Math.min(Math.max((audio.currentTime || 0) + seconds, 0), audio.duration || 0)
    audio.currentTime = nextTime
    setCurrentTime(nextTime)
  }, [])

  const handleSeek = useCallback((value: number) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = value
    setCurrentTime(value)
  }, [])

  const handlePlayPlaylistItem = useCallback((item: PlaylistItem, index: number) => {
    audioRef.current?.pause()
    currentChunkIndexRef.current = index
    setAudioUrl(item.url)
    const audio = audioRef.current
    if (!audio) return
    audio.src = item.url
    audio.load()
    void audio.play().catch((error) => {
      const isPermissionBlock = error instanceof DOMException && error.name === 'NotAllowedError'
      if (isPermissionBlock) {
        setAudioNotice('Audio is ready. Use the playback controls.')
        return
      }
      setAudioError(error instanceof Error ? error.message : 'Unable to play narration')
    })
  }, [])

  return (
    <Flex direction="col" className="h-full">
      {/* Report header: export menu (md / docx / print) */}
      {isFinalReport && !children && (
        <Flex align="center" justify="between" gap="2" className="mb-3 shrink-0">
          <Text kind="label/semibold/sm" className="text-subtle">
            Final report
          </Text>
          <ReportExportMenu />
        </Flex>
      )}

      {/* Claim verification summary — renders nothing when data is absent */}
      {isFinalReport && !children && <VerificationStatsStrip />}

      {canReadAloud && !narrationEnabled && (
        <Flex align="center" gap="2" className="mb-3 shrink-0">
          <Button
            type="button"
            kind="tertiary"
            size="tiny"
            onClick={guardClick(handleEnableNarration)}
            aria-label="Generate audio narration"
            title="Generate audio narration (opt-in)"
          >
            <Volume className="h-4 w-4" />
            <span className="ml-1">Generate audio narration</span>
          </Button>
        </Flex>
      )}

      {canReadAloud && narrationEnabled && (
        <Flex direction="col" gap="2" className="border-base mb-3 shrink-0 rounded-md border bg-surface-raised px-3 py-2">
          <Flex align="center" justify="between" gap="2">
            <Flex align="center" gap="2" className="min-w-0">
              <Volume className="h-4 w-4 shrink-0 text-accent-primary" />
              <Text kind="body/regular/sm" className="truncate text-subtle">
                {audioError || audioNotice || 'Kokoro audio narration'}
              </Text>
            </Flex>
            <Flex align="center" gap="1" className="shrink-0">
              <Button
                type="button"
                kind="tertiary"
                size="tiny"
                onClick={guardClick(handleReadAloud)}
                disabled={isPreparingAudio}
                aria-label="Prepare or play narration"
                title="Prepare or play narration"
              >
                {isPreparingAudio ? <Spinner size="small" aria-label="Preparing audio" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                type="button"
                kind="tertiary"
                size="tiny"
                onClick={guardClick(() => handleSkip(-10))}
                disabled={!audioUrl}
                aria-label="Back 10 seconds"
                title="Back 10 seconds"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                kind="tertiary"
                size="tiny"
                onClick={guardClick(handlePause)}
                disabled={!audioUrl}
                aria-label="Pause narration"
                title="Pause narration"
              >
                <Pause className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                kind="tertiary"
                size="tiny"
                onClick={guardClick(() => handleSkip(10))}
                disabled={!audioUrl}
                aria-label="Forward 10 seconds"
                title="Forward 10 seconds"
              >
                <ArrowRight className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                kind="tertiary"
                size="tiny"
                onClick={guardClick(handleStop)}
                disabled={!audioUrl && !isPreparingAudio}
                aria-label="Stop narration"
                title="Stop narration"
              >
                <Stop className="h-4 w-4" />
              </Button>
            </Flex>
          </Flex>
          <audio
            ref={setAudioElement}
            src={audioUrl ?? undefined}
            controls
            playsInline
            preload="metadata"
            className={audioUrl ? 'w-full' : 'hidden'}
            aria-label="Kokoro narration audio"
          />
          <Flex align="center" gap="2">
            <Text kind="body/regular/sm" className="w-10 text-right text-subtle">
              {formatAudioTime(currentTime)}
            </Text>
            <input
              type="range"
              min="0"
              max={Math.max(duration, 0)}
              step="0.1"
              value={Math.min(currentTime, duration || 0)}
              disabled={!audioUrl || !duration}
              onChange={(event) => handleSeek(Number(event.currentTarget.value))}
              aria-label="Narration position"
              className="h-2 min-w-0 flex-1 accent-[var(--nv-color-accent-primary)]"
            />
            <Text kind="body/regular/sm" className="w-10 text-subtle">
              {formatAudioTime(duration)}
            </Text>
          </Flex>
          {playlist.length > 0 && (
            <Flex align="center" gap="1" className="overflow-x-auto">
              {playlist.map((item, index) => (
                <Button
                  key={item.id}
                  type="button"
                  kind="tertiary"
                  size="tiny"
                  onClick={guardClick(() => handlePlayPlaylistItem(item, index))}
                  aria-label={`Play ${item.title}`}
                  title={`Play ${item.title}`}
                >
                  {item.title}
                </Button>
              ))}
            </Flex>
          )}
        </Flex>
      )}

      {/* Scrollable content area */}
      <Flex direction="col" gap="4" className="flex-1 overflow-y-auto">
        {children ? (
          children
        ) : isEmpty ? (
          <Flex direction="col" align="center" justify="center" className="flex-1 py-8 text-center">
            <Document className="text-subtle mb-3 h-8 w-8" />
            <Text kind="body/regular/md" className="text-subtle">
              Report content will appear here when available.
            </Text>
          </Flex>
        ) : isResearchNotes ? (
          /* Research notes: preview treatment */
          <Flex direction="col" gap="3" className="flex-1">
            <Flex
              align="center"
              gap="2"
              className="shrink-0 rounded-md border border-yellow-200 bg-yellow-50 px-3 py-2 dark:border-yellow-800 dark:bg-yellow-950"
            >
              <div className="h-2 w-2 animate-pulse rounded-full bg-yellow-500" />
              <Text kind="body/regular/sm" className="text-yellow-700 dark:text-yellow-300">
                Research notes from agents — final report is still being generated.
              </Text>
            </Flex>
            <div className="flex-1 opacity-80">
              <MarkdownRenderer
                content={reportContentStr}
                isStreaming={false}
                className="max-w-none"
              />
            </div>
          </Flex>
        ) : (
          /* Final report: full prominence, print-friendly, evidence-linked citations */
          <div className="report-print-area flex-1">
            <MarkdownRenderer
              content={linkedReportContent}
              isStreaming={isGeneratingReport}
              className="max-w-none"
              renderCitationLink={renderCitationLink}
            />
          </div>
        )}
      </Flex>

      {/* Export footer - only meaningful for the final report */}
      <ExportFooter />
    </Flex>
  )
}

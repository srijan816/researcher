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

import { type FC, type MouseEvent, type ReactNode, useCallback, useEffect, useRef, useState } from 'react'
import { Flex, Text, Button, Spinner } from '@/adapters/ui'
import { useShallow } from 'zustand/react/shallow'
import { ArrowLeft, ArrowRight, Document, Pause, Play, Stop, Volume } from '@/adapters/ui/icons'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import { useChatStore } from '@/features/chat'
import { ExportFooter } from './ExportFooter'

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
const NARRATION_CHUNK_TARGET_CHARS = 3200
const NARRATION_CHUNK_MAX_CHARS = 3800

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
  const { reportContent, reportContentCategory, isStreaming, currentStatus } =
    useChatStore(useShallow((s) => ({
      reportContent: s.reportContent,
      reportContentCategory: s.reportContentCategory,
      isStreaming: s.isStreaming,
      currentStatus: s.currentStatus,
    })))
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [isPreparingAudio, setIsPreparingAudio] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [audioNotice, setAudioNotice] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playlist, setPlaylist] = useState<PlaylistItem[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioUrlRef = useRef<string | null>(null)
  const playlistRef = useRef<PlaylistItem[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const currentChunkIndexRef = useRef(0)
  const isPreparingAudioRef = useRef(false)
  const waitingForNextChunkRef = useRef(false)
  const narrationSignatureRef = useRef<string | null>(null)

  const reportContentStr = typeof reportContent === 'string' ? reportContent : ''
  const isEmpty = !reportContentStr.trim()
  const isGeneratingReport = isStreaming && currentStatus === 'writing'
  const isResearchNotes = reportContentCategory === 'research_notes'
  const canReadAloud = !isEmpty && !isResearchNotes

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
      for (const [index, chunk] of speechChunks.entries()) {
        setAudioNotice(`Preparing narration part ${index + 1} of ${speechChunks.length}.`)
        const response = await fetch('/api/tts/kokoro', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: chunk, voice: DEFAULT_NARRATION_VOICE, format: 'mp3' }),
          signal: controller.signal,
        })

        if (!response.ok) {
          const detail = await response.json().catch(() => null)
          const message = typeof detail?.error === 'string' ? detail.error : 'Kokoro is not available'
          throw new Error(message)
        }

        const blob = await response.blob()
        const nextUrl = URL.createObjectURL(blob)
        const nextItem: PlaylistItem = {
          id: `${signature}-${index}`,
          title: `Part ${index + 1}/${speechChunks.length}`,
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

      setAudioNotice(`Narration ready: ${speechChunks.length} parts.`)
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

  useEffect(() => {
    if (!canReadAloud) return
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
      {canReadAloud && (
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
          /* Final report: full prominence */
          <div className="flex-1">
            <MarkdownRenderer
              content={reportContentStr}
              isStreaming={isGeneratingReport}
              className="max-w-none"
            />
          </div>
        )}
      </Flex>

      {/* Export footer - only meaningful for the final report */}
      <ExportFooter />
    </Flex>
  )
}

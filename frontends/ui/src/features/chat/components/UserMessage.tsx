// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * UserMessage Component
 *
 * User query rendered Perplexity-style: a display (Fraunces) heading at the
 * top of its turn instead of a right-aligned chat bubble.
 */

'use client'

import { type FC, useCallback, useState } from 'react'
import { Button, Flex, Text } from '@/adapters/ui'
import { Copy } from '@/adapters/ui/icons'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import { formatTime } from '@/shared/utils/format-time'

export interface UserMessageProps {
  content: string
  /** Timestamp of the message (Date or ISO string from persisted state) */
  timestamp?: Date | string
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
 * User query heading component
 */
export const UserMessage: FC<UserMessageProps> = ({ content, timestamp }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!content) return
    await copyText(content)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }, [content])

  return (
    <Flex justify="start" className="w-full">
      <Flex direction="col" align="start" className="w-full">
        <div className="gx-query-heading w-full pt-2 [&_p]:m-0">
          <MarkdownRenderer content={content} />
        </div>
        <Flex align="center" gap="2" className="mt-1 self-start">
          {content.trim() && (
            <Button
              type="button"
              kind="tertiary"
              size="tiny"
              onClick={handleCopy}
              aria-label={copied ? 'Query copied' : 'Copy query'}
              title={copied ? 'Copied' : 'Copy query'}
            >
              <Copy className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
          )}
          {timestamp && (
            <Text kind="body/regular/xs" className="text-subtle">
              {formatTime(timestamp)}
            </Text>
          )}
        </Flex>
      </Flex>
    </Flex>
  )
}

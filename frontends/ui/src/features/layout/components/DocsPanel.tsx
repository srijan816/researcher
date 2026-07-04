// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * In-app API documentation panel.
 */

'use client'

import { type FC, type ReactNode, useCallback } from 'react'
import { Flex, SidePanel, Text } from '@/adapters/ui'
import { Book } from '@/adapters/ui/icons'
import { useLayoutStore } from '../store'

const baseUrl = 'http://localhost:9000'

export const DocsPanel: FC = () => {
  const { rightPanel, closeRightPanel, openRightPanel } = useLayoutStore()
  const isOpen = rightPanel === 'docs'

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        openRightPanel('docs')
      } else {
        closeRightPanel()
      }
    },
    [openRightPanel, closeRightPanel]
  )

  return (
    <SidePanel
      className="bg-surface-base top-[var(--header-height)] h-[calc(100dvh-var(--header-height)-var(--mobile-nav-height))] sm:h-[calc(100dvh-var(--header-height))] w-screen max-w-none rounded-none sm:max-w-[520px] sm:rounded-l-2xl"
      open={isOpen}
      onOpenChange={handleOpenChange}
      side="right"
      bordered
      closeOnClickOutside={false}
      slotHeading={
        <Flex align="center" gap="2">
          <Book className="h-5 w-5" />
          API Docs
        </Flex>
      }
    >
      <Flex direction="col" gap="5">
        <DocSection title="API Keys">
          <Text kind="body/regular/sm" className="text-subtle">
            Generate a key in Settings, copy it once, then send it as a bearer token from external apps.
          </Text>
        </DocSection>

        <DocSection title="Start Research">
          <CodeBlock>
{`curl -X POST "${baseUrl}/v1/jobs/async/submit" \\
  -H "Authorization: Bearer $AIQ_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_type": "deep_researcher",
    "input": "Research the current AI chip market with citations",
    "data_sources": ["web_search"],
    "research_depth": "deeper"
  }'`}
          </CodeBlock>
        </DocSection>

        <DocSection title="Read Results">
          <CodeBlock>
{`curl -H "Authorization: Bearer $AIQ_API_KEY" \\
  "${baseUrl}/v1/jobs/async/job/$JOB_ID"

curl -H "Authorization: Bearer $AIQ_API_KEY" \\
  "${baseUrl}/v1/jobs/async/job/$JOB_ID/report"

curl -N -H "Authorization: Bearer $AIQ_API_KEY" \\
  "${baseUrl}/v1/jobs/async/job/$JOB_ID/stream"`}
          </CodeBlock>
        </DocSection>

        <DocSection title="Clarifying Questions">
          <Text kind="body/regular/sm" className="text-subtle">
            Browser chat uses the WebSocket path and may ask clarifying questions. If no answer arrives within five
            minutes, GenAlphAI Research now continues with a skip response. Headless async job submission starts directly from the
            provided input.
          </Text>
        </DocSection>

        <DocSection title="Current Search Stack">
          <Text kind="body/regular/sm" className="text-subtle">
            Web research is wired through local SearXNG for result discovery and Jina Reader for clean full-page
            extraction. Live market prices still need a finance quote source because search snippets can be stale.
          </Text>
        </DocSection>
      </Flex>
    </SidePanel>
  )
}

const DocSection: FC<{ title: string; children: ReactNode }> = ({ title, children }) => (
  <Flex direction="col" gap="2">
    <Text kind="label/semibold/xs" className="text-subtle uppercase">
      {title}
    </Text>
    {children}
  </Flex>
)

const CodeBlock: FC<{ children: string }> = ({ children }) => (
  <pre className="border-base bg-surface-raised text-primary overflow-x-auto rounded-md border p-3 text-xs leading-5">
    <code>{children}</code>
  </pre>
)

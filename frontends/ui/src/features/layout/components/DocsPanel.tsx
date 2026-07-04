// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * In-app documentation panel for GenAlphAI Research.
 */

'use client'

import { type FC, type ReactNode, useCallback } from 'react'
import { Flex, SidePanel, Text } from '@/adapters/ui'
import { Book } from '@/adapters/ui/icons'
import { useLayoutStore } from '../store'

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
          How It Works
        </Flex>
      }
    >
      <Flex direction="col" gap="5">
        <DocSection title="What GenAlphAI Research Does">
          <Text kind="body/regular/sm" className="text-subtle">
            Ask a question and GenAlphAI Research plans the work, fans out multi-engine research
            across the live web, adversarially verifies what it finds, and writes a fully cited
            report. Every claim traces back to a source you can open.
          </Text>
        </DocSection>

        <DocSection title="Research Depths">
          <Text kind="body/regular/sm" className="text-subtle">
            Pick a depth in the composer: Quick for a fast pass, Standard for everyday questions,
            Deeper for broad coverage, and Deep for exhaustive runs targeting 90&ndash;150+ sources.
            Deeper runs take longer but verify more.
          </Text>
        </DocSection>

        <DocSection title="Audio & Images">
          <Text kind="body/regular/sm" className="text-subtle">
            Finished reports can be narrated as audio directly from the report view. Toggle
            &ldquo;Images&rdquo; in the composer to blend up to 3 generated visuals into the report.
          </Text>
        </DocSection>

        <DocSection title="Following a Run">
          <Text kind="body/regular/sm" className="text-subtle">
            The research panel shows the Plan being executed, a live Activity feed of agents and
            tool calls, and Sources as citations are collected. You can step away &mdash; the run
            continues and the report lands in your session.
          </Text>
        </DocSection>

        <DocSection title="Batch Research">
          <Text kind="body/regular/sm" className="text-subtle">
            Queue several questions at once from the Batch entry in the left rail. Items run one
            by one and each produces its own cited report in its session.
          </Text>
        </DocSection>

        <DocSection title="Public API">
          <Text kind="body/regular/sm" className="text-subtle">
            A public research API is available. See the overview and usage details at:
          </Text>
          <CodeBlock>{`GET https://app2.sniperip.com/about`}</CodeBlock>
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

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * BatchQueuePanel Component
 *
 * Right-side overlay panel hosting the batch research queue. Opened from the
 * NavRail "Batch" entry or the /batch page; replaces the old Batch tab that
 * lived inside the Research panel.
 */

'use client'

import { type FC, memo, useCallback } from 'react'
import { Flex, SidePanel } from '@/adapters/ui'
import { Clock } from '@/adapters/ui/icons'
import { useLayoutStore } from '../store'
import { BatchResearchQueue } from './BatchResearchQueue'

/**
 * Overlay panel for queueing and monitoring batch research runs.
 */
export const BatchQueuePanel: FC = memo(function BatchQueuePanel() {
  const isOpen = useLayoutStore((s) => s.rightPanel === 'batch-queue')
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        openRightPanel('batch-queue')
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
          <Clock className="h-5 w-5" />
          Batch Research
        </Flex>
      }
    >
      <BatchResearchQueue />
    </SidePanel>
  )
})

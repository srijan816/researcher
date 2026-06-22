// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * VerificationStatsStrip Component
 *
 * Compact strip above the final report summarizing adversarial claim
 * verification results from /shared/verification_report.json (delivered to
 * the UI as a file artifact event). Renders nothing when the data is absent.
 */

'use client'

import { type FC, useMemo } from 'react'
import { Flex, Text } from '@/adapters/ui'
import { Check, Warning } from '@/adapters/ui/icons'
import { useChatStore } from '@/features/chat'
import { findVerificationSummary } from '../lib/report-verification'

/**
 * "Claims checked: X supported · Y partial · Z contradicted · W unverified"
 * Contradicted count is styled as a warning when non-zero.
 */
export const VerificationStatsStrip: FC = () => {
  const deepResearchFiles = useChatStore((s) => s.deepResearchFiles)

  const summary = useMemo(
    () => findVerificationSummary(deepResearchFiles),
    [deepResearchFiles]
  )

  if (!summary) return null

  const hasContradictions = summary.contradicted > 0

  return (
    <Flex
      align="center"
      gap="2"
      className={`border-base mb-3 shrink-0 flex-wrap rounded-md border bg-surface-raised px-3 py-2 ${
        hasContradictions ? 'border-warning' : ''
      }`}
      data-testid="verification-stats-strip"
      role="status"
      aria-label={`Claims checked: ${summary.supported} supported, ${summary.partiallySupported} partially supported, ${summary.contradicted} contradicted, ${summary.notAddressed} unverified`}
    >
      {hasContradictions ? (
        <Warning className="h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
      ) : (
        <Check className="h-4 w-4 shrink-0 text-success" aria-hidden="true" />
      )}
      <Text kind="body/regular/sm" className="text-subtle">
        Claims checked:
      </Text>
      <Text kind="label/semibold/sm" className="text-success">
        {summary.supported} supported
      </Text>
      <Text kind="body/regular/sm" className="text-subtle" aria-hidden="true">
        ·
      </Text>
      <Text kind="label/semibold/sm" className="text-subtle">
        {summary.partiallySupported} partial
      </Text>
      <Text kind="body/regular/sm" className="text-subtle" aria-hidden="true">
        ·
      </Text>
      <Text kind="label/semibold/sm" className={hasContradictions ? 'text-warning' : 'text-subtle'}>
        {summary.contradicted} contradicted
      </Text>
      <Text kind="body/regular/sm" className="text-subtle" aria-hidden="true">
        ·
      </Text>
      <Text kind="label/semibold/sm" className="text-subtle">
        {summary.notAddressed} unverified
      </Text>
    </Flex>
  )
}

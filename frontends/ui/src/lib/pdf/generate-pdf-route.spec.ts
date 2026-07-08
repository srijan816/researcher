// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import { config } from '@/pages/api/generate-pdf'

describe('/api/generate-pdf config', () => {
  test('allows reports with inlined images in the request body', () => {
    expect(config.api.bodyParser.sizeLimit).toBe('16mb')
  })
})

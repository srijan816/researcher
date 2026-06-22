// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { SettingsPanel } from './SettingsPanel'

const mocks = vi.hoisted(() => ({
  closeRightPanel: vi.fn(),
  openRightPanel: vi.fn(),
  setTheme: vi.fn(),
  listAPIKeys: vi.fn(),
  createAPIKey: vi.fn(),
  revokeAPIKey: vi.fn(),
}))

vi.mock('../store', () => ({
  useLayoutStore: vi.fn((selector?: (state: any) => any) => {
    const state = {
      rightPanel: 'settings',
      closeRightPanel: mocks.closeRightPanel,
      openRightPanel: mocks.openRightPanel,
      theme: 'system',
      setTheme: mocks.setTheme,
    }
    return selector ? selector(state) : state
  }),
}))

vi.mock('@/adapters/api', () => ({
  listAPIKeys: mocks.listAPIKeys,
  createAPIKey: mocks.createAPIKey,
  revokeAPIKey: mocks.revokeAPIKey,
}))

vi.mock('@/adapters/auth', () => ({
  useAuth: vi.fn(() => ({
    authRequired: true,
    mustChangePassword: false,
  })),
}))

import { useLayoutStore } from '../store'

describe('SettingsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listAPIKeys.mockResolvedValue({ api_keys: [] })
    mocks.createAPIKey.mockResolvedValue({
      id: 'key-1',
      name: 'External App',
      prefix: 'aiq_test',
      key: 'aiq_test_secret',
      created_at: '2026-04-27T00:00:00Z',
      last_used_at: null,
    })
    // Reset mock to default open state
    vi.mocked(useLayoutStore).mockImplementation((selector?: (state: any) => any) => {
      const state = {
        rightPanel: 'settings',
        closeRightPanel: mocks.closeRightPanel,
        openRightPanel: mocks.openRightPanel,
        theme: 'system',
        setTheme: mocks.setTheme,
      }
      return selector ? selector(state) : state
    })
  })

  test('renders panel heading when open', () => {
    render(<SettingsPanel />)

    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  test('renders theme options section with static Dark message', () => {
    render(<SettingsPanel />)

    expect(screen.getByText('UI Theme')).toBeInTheDocument()
    expect(screen.getByText('Dark')).toBeInTheDocument()
    expect(screen.getByText('Dark is the fixed interface theme for this deployment.')).toBeInTheDocument()
  })

  test('does not render when panel is closed', () => {
    vi.mocked(useLayoutStore).mockImplementation((selector?: (state: any) => any) => {
      const state = {
        rightPanel: null,
        closeRightPanel: mocks.closeRightPanel,
        openRightPanel: mocks.openRightPanel,
        theme: 'system',
        setTheme: mocks.setTheme,
      }
      return selector ? selector(state) : state
    })

    render(<SettingsPanel />)

    // Panel should not be visible (SidePanel handles this)
    // The heading won't be rendered in closed state
    // This tests the isOpen logic
    expect(screen.queryByText('Settings')).not.toBeInTheDocument()
  })

  test('renders footer text', () => {
    render(<SettingsPanel />)

    expect(screen.getByText(/settings are saved automatically/i)).toBeInTheDocument()
  })
})

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, test, expect, beforeEach } from 'vitest'
import { useLayoutStore } from './store'

describe('useLayoutStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useLayoutStore.setState({
      isSessionsPanelOpen: false,
      rightPanel: null,
      researchPanelTab: 'plan',
      dataSourcesPanelTab: 'connections',
      researchDepth: 'deeper',
      researchEngine: 'aiq',
      theme: 'dark',
      imageCount: 3,
    })
    window.localStorage.clear()
  })

  describe('initial state', () => {
    test('has correct default values', () => {
      const state = useLayoutStore.getState()

      expect(state.isSessionsPanelOpen).toBe(false)
      expect(state.rightPanel).toBeNull()
      expect(state.researchPanelTab).toBe('plan')
      expect(state.dataSourcesPanelTab).toBe('connections')
      expect(state.researchDepth).toBe('deeper')
      expect(state.researchEngine).toBe('aiq')
    })
  })

  describe('toggleSessionsPanel', () => {
    test('opens sessions panel when closed', () => {
      useLayoutStore.getState().toggleSessionsPanel()

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)
    })

    test('closes sessions panel when open', () => {
      useLayoutStore.setState({ isSessionsPanelOpen: true })

      useLayoutStore.getState().toggleSessionsPanel()

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(false)
    })

    test('toggles multiple times correctly', () => {
      const { toggleSessionsPanel } = useLayoutStore.getState()

      toggleSessionsPanel()
      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)

      toggleSessionsPanel()
      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(false)

      toggleSessionsPanel()
      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)
    })
  })

  describe('setSessionsPanelOpen', () => {
    test('sets sessions panel to open', () => {
      useLayoutStore.getState().setSessionsPanelOpen(true)

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(true)
    })

    test('sets sessions panel to closed', () => {
      useLayoutStore.setState({ isSessionsPanelOpen: true })

      useLayoutStore.getState().setSessionsPanelOpen(false)

      expect(useLayoutStore.getState().isSessionsPanelOpen).toBe(false)
    })
  })

  describe('openRightPanel', () => {
    test('opens research panel', () => {
      useLayoutStore.getState().openRightPanel('research')

      expect(useLayoutStore.getState().rightPanel).toBe('research')
    })

    test('opens data-sources panel', () => {
      useLayoutStore.getState().openRightPanel('data-sources')

      expect(useLayoutStore.getState().rightPanel).toBe('data-sources')
    })

    test('opens settings panel', () => {
      useLayoutStore.getState().openRightPanel('settings')

      expect(useLayoutStore.getState().rightPanel).toBe('settings')
    })

    test('replaces existing panel', () => {
      useLayoutStore.setState({ rightPanel: 'research' })

      useLayoutStore.getState().openRightPanel('settings')

      expect(useLayoutStore.getState().rightPanel).toBe('settings')
    })
  })

  describe('closeRightPanel', () => {
    test('closes open panel', () => {
      useLayoutStore.setState({ rightPanel: 'research' })

      useLayoutStore.getState().closeRightPanel()

      expect(useLayoutStore.getState().rightPanel).toBeNull()
    })

    test('handles closing when already closed', () => {
      useLayoutStore.getState().closeRightPanel()

      expect(useLayoutStore.getState().rightPanel).toBeNull()
    })
  })

  describe('setResearchPanelTab', () => {
    test('sets thinking tab', () => {
      useLayoutStore.getState().setResearchPanelTab('thinking')

      expect(useLayoutStore.getState().researchPanelTab).toBe('thinking')
    })

    test('sets citations tab', () => {
      useLayoutStore.getState().setResearchPanelTab('citations')

      expect(useLayoutStore.getState().researchPanelTab).toBe('citations')
    })

    test('sets report tab', () => {
      useLayoutStore.setState({ researchPanelTab: 'thinking' })

      useLayoutStore.getState().setResearchPanelTab('report')

      expect(useLayoutStore.getState().researchPanelTab).toBe('report')
    })
  })

  describe('setDataSourcesPanelTab', () => {
    test('sets connections tab', () => {
      useLayoutStore.setState({ dataSourcesPanelTab: 'files' })

      useLayoutStore.getState().setDataSourcesPanelTab('connections')

      expect(useLayoutStore.getState().dataSourcesPanelTab).toBe('connections')
    })

    test('sets files tab', () => {
      useLayoutStore.getState().setDataSourcesPanelTab('files')

      expect(useLayoutStore.getState().dataSourcesPanelTab).toBe('files')
    })
  })

  describe('setTheme', () => {
    test('ignores legacy light requests and persists dark', () => {
      useLayoutStore.getState().setTheme('light' as never)

      expect(useLayoutStore.getState().theme).toBe('dark')
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
      expect(window.localStorage.getItem('gx-theme')).toBe('dark')
    })

    test('sets dark theme', () => {
      useLayoutStore.getState().setTheme('dark')

      expect(useLayoutStore.getState().theme).toBe('dark')
      expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
    })

    test('resolves system to a concrete theme', () => {
      useLayoutStore.getState().setTheme('system')

      expect(useLayoutStore.getState().theme).toBe('dark')
    })
  })

  describe('setImageCount', () => {
    test('sets a valid image count', () => {
      useLayoutStore.getState().setImageCount(2)

      expect(useLayoutStore.getState().imageCount).toBe(2)
    })

    test('clamps the image count to the 1-4 range', () => {
      useLayoutStore.getState().setImageCount(9)
      expect(useLayoutStore.getState().imageCount).toBe(4)

      useLayoutStore.getState().setImageCount(0)
      expect(useLayoutStore.getState().imageCount).toBe(1)
    })

    test('persists research defaults to localStorage', () => {
      useLayoutStore.getState().setImageCount(2)

      const raw = window.localStorage.getItem('gx-research-defaults')
      expect(raw).not.toBeNull()
      expect(JSON.parse(raw as string)).toMatchObject({ imageCount: 2 })
    })
  })

  describe('setResearchDepth', () => {
    test('sets research depth tier', () => {
      useLayoutStore.getState().setResearchDepth('deep')

      expect(useLayoutStore.getState().researchDepth).toBe('deep')
    })

    test('sets medium research depth tier', () => {
      useLayoutStore.getState().setResearchDepth('medium')

      expect(useLayoutStore.getState().researchDepth).toBe('medium')
    })
  })

  describe('setIncludeImages', () => {
    test('toggles generated images flag', () => {
      useLayoutStore.getState().setIncludeImages(true)

      expect(useLayoutStore.getState().includeImages).toBe(true)
    })
  })

  describe('setComposerDraft', () => {
    test('sets and clears composer draft text', () => {
      useLayoutStore.getState().setComposerDraft('Compare vector databases')
      expect(useLayoutStore.getState().composerDraft).toBe('Compare vector databases')

      useLayoutStore.getState().setComposerDraft(null)
      expect(useLayoutStore.getState().composerDraft).toBeNull()
    })
  })
})

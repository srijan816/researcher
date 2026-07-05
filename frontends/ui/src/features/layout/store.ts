// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout Store
 *
 * Zustand store for managing the main app layout state.
 * Controls sidebar visibility and panel states.
 */

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type {
  LayoutState,
  LayoutStore,
  RightPanelType,
  ResearchPanelTab,
  DataSourcesPanelTab,
  ThemeMode,
  ResearchDepth,
} from './types'
import { createDataSourcesClient, type DataSourceFromAPI } from '@/adapters/api'
import { applyTheme, getInitialTheme, resolveTheme } from './theme'

const isEnabledByDefault = (source: DataSourceFromAPI): boolean =>
  !source.requires_auth && (source.default_enabled ?? source.id === 'web_search')

/** localStorage key for persisted research defaults (depth, images) */
export const RESEARCH_DEFAULTS_STORAGE_KEY = 'gx-research-defaults'

const IMAGE_COUNT_MIN = 1
const IMAGE_COUNT_MAX = 4
const IMAGE_COUNT_DEFAULT = 3
const RESEARCH_DEPTH_VALUES: readonly string[] = ['shallow', 'medium', 'deeper', 'deep']

const clampImageCount = (count: number): number => {
  const rounded = Math.round(count)
  if (!Number.isFinite(rounded)) return IMAGE_COUNT_DEFAULT
  return Math.min(IMAGE_COUNT_MAX, Math.max(IMAGE_COUNT_MIN, rounded))
}

interface ResearchDefaults {
  researchDepth?: ResearchDepth
  includeImages?: boolean
  imageCount?: number
}

const loadResearchDefaults = (): ResearchDefaults => {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(RESEARCH_DEFAULTS_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as ResearchDefaults
    return {
      researchDepth: RESEARCH_DEPTH_VALUES.includes(parsed.researchDepth as string)
        ? parsed.researchDepth
        : undefined,
      includeImages: typeof parsed.includeImages === 'boolean' ? parsed.includeImages : undefined,
      imageCount:
        typeof parsed.imageCount === 'number' ? clampImageCount(parsed.imageCount) : undefined,
    }
  } catch {
    return {}
  }
}

const saveResearchDefaults = (state: {
  researchDepth: ResearchDepth
  includeImages: boolean
  imageCount: number
}): void => {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(
      RESEARCH_DEFAULTS_STORAGE_KEY,
      JSON.stringify({
        researchDepth: state.researchDepth,
        includeImages: state.includeImages,
        imageCount: state.imageCount,
      })
    )
  } catch {
    // Storage unavailable — defaults apply for this session only
  }
}

const persistedDefaults = loadResearchDefaults()

const initialState: LayoutState = {
  isSessionsPanelOpen: false,
  rightPanel: null,
  researchPanelTab: 'plan',
  dataSourcesPanelTab: 'connections',
  enabledDataSourceIds: [], // Start empty, populated when data sources are fetched
  researchDepth: persistedDefaults.researchDepth ?? 'deeper',
  researchEngine: 'aiq',
  includeImages: persistedDefaults.includeImages ?? false,
  imageCount: persistedDefaults.imageCount ?? IMAGE_COUNT_DEFAULT,
  composerDraft: null,
  theme: getInitialTheme(),
  availableDataSources: null,
  knowledgeLayerAvailable: false, // Default to false until API confirms availability
  dataSourcesLoading: false,
  dataSourcesError: null,
  // Deprecated aliases for backwards compatibility
  detailsPanelTab: 'report',
  dataSourcePanelTab: 'connections',
}

export const useLayoutStore = create<LayoutStore>()(
  devtools(
    (set) => ({
      ...initialState,

      toggleSessionsPanel: () =>
        set(
          (state) => ({ isSessionsPanelOpen: !state.isSessionsPanelOpen }),
          false,
          'toggleSessionsPanel'
        ),

      setSessionsPanelOpen: (open: boolean) =>
        set({ isSessionsPanelOpen: open }, false, 'setSessionsPanelOpen'),

      openRightPanel: (panel: RightPanelType) =>
        set({ rightPanel: panel }, false, 'openRightPanel'),

      closeRightPanel: () => set({ rightPanel: null }, false, 'closeRightPanel'),

      setResearchPanelTab: (tab: ResearchPanelTab) =>
        set({ researchPanelTab: tab }, false, 'setResearchPanelTab'),

      setDataSourcesPanelTab: (tab: DataSourcesPanelTab) =>
        set({ dataSourcesPanelTab: tab }, false, 'setDataSourcesPanelTab'),

      toggleDataSource: (id: string) =>
        set(
          (state) => {
            const isEnabled = state.enabledDataSourceIds.includes(id)
            return {
              enabledDataSourceIds: isEnabled
                ? state.enabledDataSourceIds.filter((sourceId) => sourceId !== id)
                : [...state.enabledDataSourceIds, id],
            }
          },
          false,
          'toggleDataSource'
        ),

      setEnabledDataSources: (ids: string[]) =>
        set({ enabledDataSourceIds: ids }, false, 'setEnabledDataSources'),

      setResearchDepth: (depth: ResearchDepth) =>
        set(
          (state) => {
            saveResearchDefaults({
              researchDepth: depth,
              includeImages: state.includeImages,
              imageCount: state.imageCount,
            })
            return { researchDepth: depth }
          },
          false,
          'setResearchDepth'
        ),

      setIncludeImages: (include: boolean) =>
        set(
          (state) => {
            saveResearchDefaults({
              researchDepth: state.researchDepth,
              includeImages: include,
              imageCount: state.imageCount,
            })
            return { includeImages: include }
          },
          false,
          'setIncludeImages'
        ),

      setImageCount: (count: number) =>
        set(
          (state) => {
            const imageCount = clampImageCount(count)
            saveResearchDefaults({
              researchDepth: state.researchDepth,
              includeImages: state.includeImages,
              imageCount,
            })
            return { imageCount }
          },
          false,
          'setImageCount'
        ),

      setComposerDraft: (draft: string | null) =>
        set({ composerDraft: draft }, false, 'setComposerDraft'),

      setTheme: (theme: ThemeMode) => {
        const resolved = resolveTheme(theme)
        applyTheme(resolved)
        set({ theme: resolved }, false, 'setTheme')
      },

      fetchDataSources: async (authToken?: string) => {
        set({ dataSourcesLoading: true, dataSourcesError: null }, false, 'fetchDataSources/start')

        try {
          const client = createDataSourcesClient({ authToken })
          const response = await client.getDataSources()

          // Enable only sources marked as default by the backend. This keeps local
          // corpora opt-in unless a user selects them for a session.
          const enabledIds = response.data_sources
            .filter(isEnabledByDefault)
            .map((source) => source.id)

          set(
            {
              availableDataSources: response.data_sources,
              knowledgeLayerAvailable: response.knowledge_layer,
              enabledDataSourceIds: enabledIds,
              dataSourcesLoading: false,
              dataSourcesError: null,
            },
            false,
            'fetchDataSources/success'
          )
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch data sources'
          set(
            {
              dataSourcesLoading: false,
              dataSourcesError: errorMessage,
            },
            false,
            'fetchDataSources/error'
          )
        }
      },

      disableAuthRequiredSources: () =>
        set(
          (state) => ({
            enabledDataSourceIds: state.enabledDataSourceIds.filter((id) => {
              const source = state.availableDataSources?.find((s) => s.id === id)
              return !source?.requires_auth
            }),
          }),
          false,
          'disableAuthRequiredSources'
        ),

      setAvailableDataSources: (sources: DataSourceFromAPI[]) =>
        set({ availableDataSources: sources }, false, 'setAvailableDataSources'),

      setKnowledgeLayerAvailable: (available: boolean) =>
        set({ knowledgeLayerAvailable: available }, false, 'setKnowledgeLayerAvailable'),

      // Deprecated actions - delegate to new ones
      setDetailsPanelTab: (tab: ResearchPanelTab) =>
        set(
          { researchPanelTab: tab, detailsPanelTab: tab },
          false,
          'setDetailsPanelTab'
        ),

      setDataSourcePanelTab: (tab: DataSourcesPanelTab) =>
        set(
          { dataSourcesPanelTab: tab, dataSourcePanelTab: tab },
          false,
          'setDataSourcePanelTab'
        ),
    }),
    { name: 'LayoutStore' }
  )
)

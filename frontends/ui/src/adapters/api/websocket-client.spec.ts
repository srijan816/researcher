// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { NATMessageType, NATWebSocketClient } from './websocket-client'

class MockWebSocket {
  static readonly OPEN = 1
  static instances: MockWebSocket[] = []

  readonly url: string
  readyState = MockWebSocket.OPEN
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  send = vi.fn()
  close = vi.fn()
}

describe('NATWebSocketClient auth observability', () => {
  beforeEach(() => {
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    delete (window as unknown as Record<string, unknown>).DD_RUM
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  test('emits RUM error for websocket auth_error payloads', async () => {
    const addError = vi.fn()
    ;(window as unknown as Record<string, unknown>).DD_RUM = { addError }
    const onError = vi.fn()
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: { onError },
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))
    ws.onmessage?.(
      {
        data: JSON.stringify({
          type: NATMessageType.ERROR,
          content: {
            code: 'UNKNOWN_ERROR',
            message: 'auth_error',
            details: 'Token expired',
          },
          status: 'error',
        }),
      } as MessageEvent
    )

    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        code: 'UNKNOWN_ERROR',
        message: 'auth_error',
        details: 'Token expired',
      })
    )
    expect(addError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        source: 'websocket',
        auth_error_code: 'auth_error',
        details: 'Token expired',
      })
    )
  })

  test('does not emit RUM error for non-auth websocket errors', async () => {
    const addError = vi.fn()
    ;(window as unknown as Record<string, unknown>).DD_RUM = { addError }
    const onError = vi.fn()
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: { onError },
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))
    ws.onmessage?.(
      {
        data: JSON.stringify({
          type: NATMessageType.ERROR,
          content: {
            code: 'UNKNOWN_ERROR',
            message: 'workflow_error',
            details: 'Unexpected failure',
          },
          status: 'error',
        }),
      } as MessageEvent
    )

    expect(onError).toHaveBeenCalled()
    expect(addError).not.toHaveBeenCalled()
  })

  test('sends shallow research mode without force-deep override', async () => {
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: {},
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))

    client.sendMessage('University education', ['web_search'], 'shallow')

    const envelope = JSON.parse(ws.send.mock.calls[0][0])
    const payload = JSON.parse(envelope.content.messages[0].content[0].text)
    expect(payload).toMatchObject({
      query: 'University education',
      data_sources: ['web_search'],
      research_depth: 'shallow',
      agent_type: 'deep_researcher',
      force_deep_research: false,
    })
  })

  test('sends deeper and deep modes as explicit deep-research requests', async () => {
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: {},
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))

    client.sendMessage('Best way to live in the age of AI', ['web_search'], 'deeper')
    client.sendMessage('Monetize takeabreak.life', ['web_search'], 'deep')

    const deeperEnvelope = JSON.parse(ws.send.mock.calls[0][0])
    const deepEnvelope = JSON.parse(ws.send.mock.calls[1][0])
    const deeperPayload = JSON.parse(deeperEnvelope.content.messages[0].content[0].text)
    const deepPayload = JSON.parse(deepEnvelope.content.messages[0].content[0].text)

    expect(deeperPayload).toMatchObject({
      research_depth: 'deeper',
      agent_type: 'deep_researcher',
      force_deep_research: true,
    })
    expect(deepPayload).toMatchObject({
      research_depth: 'deep',
      agent_type: 'deep_researcher',
      force_deep_research: true,
    })
  })

  test('sends include_images flag with the default engine agent type', async () => {
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: {},
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))

    client.sendMessage('Audit the research workflow', ['web_search'], 'deeper', true)

    const envelope = JSON.parse(ws.send.mock.calls[0][0])
    const payload = JSON.parse(envelope.content.messages[0].content[0].text)

    expect(payload).toMatchObject({
      query: 'Audit the research workflow',
      research_depth: 'deeper',
      agent_type: 'deep_researcher',
      force_deep_research: true,
      include_images: true,
      image_count: 3,
    })
  })

  test('sends the requested image_count when images are enabled', async () => {
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: {},
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))

    client.sendMessage('Visual research', ['web_search'], 'deeper', true, 2)

    const envelope = JSON.parse(ws.send.mock.calls[0][0])
    const payload = JSON.parse(envelope.content.messages[0].content[0].text)
    expect(payload).toMatchObject({ include_images: true, image_count: 2 })
  })

  test('omits image_count when images are disabled', async () => {
    const client = new NATWebSocketClient({
      conversationId: 'conv-1',
      websocketUrl: 'ws://localhost/websocket',
      callbacks: {},
    })

    await client.connect()
    const ws = MockWebSocket.instances[0]
    ws.onopen?.(new Event('open'))

    client.sendMessage('No visuals', ['web_search'], 'deeper', false, 2)

    const envelope = JSON.parse(ws.send.mock.calls[0][0])
    const payload = JSON.parse(envelope.content.messages[0].content[0].text)
    expect(payload.include_images).toBe(false)
    expect('image_count' in payload).toBe(false)
  })
})

// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Gateway Server with WebSocket Proxy
 *
 * Architecture:
 * - Runs on port 3000 as the main entry point
 * - Proxies to Next.js server (dev on 3001)
 * - Proxies /websocket to backend WebSocket endpoint
 */

const http = require('http')
const httpProxy = require('http-proxy')
const { parse } = require('url')

const dev = process.env.NODE_ENV !== 'production'
const hostname = process.env.BIND_HOST || process.env.HOST || '0.0.0.0'
const port = parseInt(process.env.PORT || '3000', 10)

const getBackendUrl = () => {
  const url = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:9000'
  return url.replace(/\/$/, '')
}

const getBackendWsUrl = () => {
  const baseUrl = getBackendUrl()
  return baseUrl.replace(/^http/, 'ws')
}

const BACKEND_HTTP_URL = getBackendUrl()
const BACKEND_WS_URL = getBackendWsUrl()
const NEXT_INTERNAL_URL = process.env.NEXT_INTERNAL_URL || 'http://localhost:3001'

// In production, we run Next.js in the same process
let nextApp = null
let nextHandle = null

if (!dev) {
  const next = require('next')
  nextApp = next({ dev: false, hostname, port: 3001 })
  nextHandle = nextApp.getRequestHandler()
}

// Proxy options
const proxyOpts = {
  changeOrigin: true,
  ws: true,
  xfwd: true,
  preserveHeaderKeyCase: true,
}

const nextProxy = httpProxy.createProxyServer({
  ...proxyOpts,
  target: NEXT_INTERNAL_URL,
})

const backendProxy = httpProxy.createProxyServer({
  ...proxyOpts,
})

// Robust error handler for proxy -> writes 502 and destroys socket safely
function safeProxyError(res, targetName, err) {
  console.error(`[${targetName} Error]:`, err?.message || err)
  try {
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: `${targetName} unavailable` }))
    } else {
      res.end()
    }
  } catch (e) {
    // Socket may already be destroyed - that's fine
  }
}

// Register proxy error handlers
nextProxy.on('error', (err, req, res) => safeProxyError(res, 'Next.js', err))
nextProxy.on('econnreset', (err, req, res) => safeProxyError(res, 'Next.js', err))
backendProxy.on('error', (err, req, res) => safeProxyError(res, 'Backend', err))
backendProxy.on('econnreset', (err, req, res) => safeProxyError(res, 'Backend', err))

// Suppress "Listening to the same EventEmitter" warnings from http-proxy
nextProxy.on('error', () => {})
backendProxy.on('error', () => {})

// WebSocket backend keep-alive
backendProxy.on('open', (proxySocket) => {
  try {
    proxySocket.setKeepAlive?.(true, 15000)
  } catch {}
  proxySocket.on('error', (e) => {
    // Swallow - handled by error events above
  })
})

// Forward cookies for backend WebSocket
backendProxy.on('proxyReqWs', (proxyReq, req) => {
  if (req.headers.cookie) {
    proxyReq.setHeader('Cookie', req.headers.cookie)
  }
})

const startServer = async () => {
  if (!dev && nextApp) {
    await nextApp.prepare()
  }

  const server = http.createServer()

  // Handle every incoming connection gracefully
  server.on('connection', (socket) => {
    socket.setKeepAlive(true, 15000)
  })

  // Main HTTP request handler
  server.on('request', async (req, res) => {
    let parsedUrl
    try {
      parsedUrl = parse(req.url, true)
    } catch {
      res.writeHead(400, { 'Content-Type': 'text/plain' })
      res.end('Bad Request')
      return
    }

    if (dev) {
      const pathname = parsedUrl.pathname || ''

      if (pathname.startsWith('/api/')) {
        nextProxy.web(req, res, { target: NEXT_INTERNAL_URL })
        return
      }

      // Backend routes
      if (
        pathname.startsWith('/v1/') ||
        pathname.startsWith('/chat') ||
        pathname.startsWith('/generate') ||
        pathname.startsWith('/executions') ||
        pathname.startsWith('/health') ||
        pathname.startsWith('/monitor') ||
        pathname.startsWith('/static')
      ) {
        backendProxy.web(req, res, { target: BACKEND_HTTP_URL })
        return
      }

      nextProxy.web(req, res, { target: NEXT_INTERNAL_URL })
    } else {
      try {
        await nextHandle(req, res, parsedUrl)
      } catch (err) {
        console.error('[Production Handler Error]:', err?.message || err)
        if (!res.headersSent) {
          res.writeHead(500, { 'Content-Type': 'application/json' })
        }
        res.end('Internal Server Error')
      }
    }
  })

  // WebSocket upgrade handler
  server.on('upgrade', (req, socket, head) => {
    let parsedUrl
    try {
      parsedUrl = parse(req.url, true)
    } catch {
      socket.destroy()
      return
    }
    const pathname = parsedUrl.pathname || '/'

    if (pathname === '/websocket' || pathname.startsWith('/websocket')) {
      req.url = '/websocket' + (parsedUrl.search || '')
      backendProxy.ws(req, socket, head, { target: BACKEND_WS_URL }, (err) => {
        if (err) {
          console.error('[WS Error]:', err.message)
        }
        socket.destroy()
      })
    } else if (dev) {
      nextProxy.ws(req, socket, head, { target: NEXT_INTERNAL_URL }, (err) => {
        if (err) {
          console.error('[Next.js WS Error]:', err.message)
        }
        socket.destroy()
      })
    } else {
      const upgradeHandler = nextApp.getUpgradeHandler()
      upgradeHandler(req, socket, head)
    }
  })

  // Prevent crash on uncaught errors
  process.on('uncaughtException', (err) => {
    console.error('[Uncaught Exception]:', err?.message || err)
    if (err?.code === 'EADDRINUSE') {
      process.exit(1)
    }
  })

  process.on('unhandledRejection', (reason) => {
    console.error('[Unhandled Rejection]:', reason)
  })

  // Server configuration for long-running connections
  server.keepAliveTimeout = 60000
  server.headersTimeout = 65000
  server.requestTimeout = 0

  server.on('error', (err) => {
    console.error('[Server Error]:', err?.message || err)
    if (err?.code === 'EADDRINUSE') {
      process.exit(1)
    }
  })

  // Handle client-level errors without crashing
  server.on('clientError', (err, socket) => {
    if (socket?.destroyed) return
    try {
      socket.end('HTTP/1.1 400 Bad Request\r\n\r\n')
    } catch {}
  })

  server.listen(port, hostname, () => {
    console.log(`\nGateway running:\n  Frontend: http://localhost:${port}\n  Backend:  ${BACKEND_HTTP_URL}\n`)
  })

  // Graceful shutdown
  const cleanExit = (signal) => {
    console.log(`\nShutting down...`)
    server.close(() => process.exit(0))
    setTimeout(() => process.exit(0), 2000)
  }

  process.on('SIGTERM', () => cleanExit('SIGTERM'))
  process.on('SIGINT', () => cleanExit('SIGINT'))
}

startServer().catch((err) => {
  console.error('[Startup Error]:', err)
  process.exit(1)
})

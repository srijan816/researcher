// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client'

import { type FC, type ReactNode, memo, useMemo } from 'react'
import ReactMarkdown, { type Components, type ExtraProps } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Text, CodeSnippet, Anchor } from '@/adapters/ui'
import type { MarkdownRendererProps } from './types'
import { getLanguageFromClassName } from './utils'

function getTextFromChildren(node: ReactNode): string {
  if (typeof node === 'string') return node
  if (typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(getTextFromChildren).join('')
  if (node && typeof node === 'object' && 'props' in node) {
    return getTextFromChildren((node as React.ReactElement).props.children)
  }
  return ''
}

/**
 * Only render images from same-origin API routes (`/api/`, `/v1/`) or https.
 * Blocks data:, javascript:, http:, protocol-relative and other schemes.
 */
export function isAllowedImageSrc(src: string | undefined | null): src is string {
  if (!src) return false
  if (src.startsWith('/api/') || src.startsWith('/v1/')) return true
  return src.startsWith('https://')
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/**
 * MarkdownRenderer - Renders markdown content with KUI styling
 *
 * @param content - Markdown string to render
 * @param isStreaming - Whether content is still streaming (disables memoization)
 * @param className - Additional CSS classes
 * @param compact - Use smaller text sizes for chat bubbles
 */
export const MarkdownRenderer: FC<MarkdownRendererProps> = memo(
  ({ content, className = '', compact = false, renderCitationLink }) => {
    // Custom component mappings
    const components: Components = useMemo(
      () => ({
        code: ({
          children,
          className: codeClassName,
          ...props
        }: React.ComponentPropsWithoutRef<'code'> & ExtraProps) => {
          // Check if this is a block code (has language class) vs inline
          const isBlock = codeClassName?.startsWith('language-')
          const codeContent = String(children).replace(/\n$/, '')

          if (isBlock) {
            const language = getLanguageFromClassName(codeClassName)
            const lineCount = codeContent.split('\n').length

            return (
              <CodeSnippet
                value={codeContent}
                language={language}
                kind="block"
                collapsible={lineCount > 15}
                rows={15}
              />
            )
          }

          // Inline code
          return (
            <code
              className="bg-surface-raised text-primary rounded px-1.5 py-0.5 font-mono text-sm"
              {...props}
            >
              {children}
            </code>
          )
        },

        // Skip default pre rendering since CodeSnippet handles it
        pre: ({ children }) => <>{children}</>,

        // Headings — include id for in-page anchor navigation
        h1: ({ children }) => {
          const id = slugify(getTextFromChildren(children))
          return (
            <Text asChild kind="title/xl" className="text-primary mb-3 mt-6 block scroll-mt-4">
              <h1 id={id}>{children}</h1>
            </Text>
          )
        },
        h2: ({ children }) => {
          const id = slugify(getTextFromChildren(children))
          return (
            <Text asChild kind="title/lg" className="text-primary mb-2 mt-5 block scroll-mt-4">
              <h2 id={id}>{children}</h2>
            </Text>
          )
        },
        h3: ({ children }) => {
          const id = slugify(getTextFromChildren(children))
          return (
            <Text asChild kind="title/md" className="text-primary mb-2 mt-4 block scroll-mt-4">
              <h3 id={id}>{children}</h3>
            </Text>
          )
        },
        h4: ({ children }) => {
          const id = slugify(getTextFromChildren(children))
          return (
            <Text asChild kind="title/sm" className="text-primary mb-1 mt-3 block scroll-mt-4">
              <h4 id={id}>{children}</h4>
            </Text>
          )
        },

        // Paragraphs
        p: ({ children }) => (
          <Text
            asChild
            kind={compact ? 'body/regular/sm' : 'body/regular/md'}
            className="text-primary mb-3 block leading-relaxed"
          >
            <p>{children}</p>
          </Text>
        ),

        // Lists
        ul: ({ children }) => (
          <ul className="text-primary mb-3 list-outside list-disc space-y-1 pl-5">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="text-primary mb-3 list-outside list-decimal space-y-1 pl-5">{children}</ol>
        ),
        li: ({ children }) => (
          <Text asChild kind={compact ? 'body/regular/sm' : 'body/regular/md'}>
            <li className="text-primary">{children}</li>
          </Text>
        ),

        // Links — anchor hrefs scroll in-page; external hrefs open new tabs.
        // `#citation-N` hrefs (from linkifyCitationMarkers) render via the
        // caller-provided citation renderer when available.
        a: ({ href, children }) => {
          if (href?.startsWith('#citation-') && renderCitationLink) {
            const citationNumber = Number(href.slice('#citation-'.length))
            if (Number.isFinite(citationNumber) && citationNumber > 0) {
              return <>{renderCitationLink(citationNumber, children)}</>
            }
          }
          if (href?.startsWith('#')) {
            return (
              <Anchor
                href={href}
                kind="inline"
                onClick={(e: React.MouseEvent) => {
                  e.preventDefault()
                  const el = document.getElementById(href.slice(1))
                  el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }}
              >
                {children}
              </Anchor>
            )
          }
          return (
            <Anchor href={href ?? '#'} target="_blank" rel="noopener noreferrer" kind="inline">
              {children}
            </Anchor>
          )
        },

        // Images — figure-quality presentation with caption from alt text.
        // Only same-origin API paths (/api/, /v1/) and https sources render;
        // everything else (data:, http:, javascript:, …) is dropped.
        // Rendered with <span>s (block-styled) because images arrive inside <p>.
        img: ({ src, alt }) => {
          const srcStr = typeof src === 'string' ? src : undefined
          if (!isAllowedImageSrc(srcStr)) return null
          return (
            <span className="my-4 block text-center">
              {/* eslint-disable-next-line @next/next/no-img-element -- report images are same-origin authed API assets */}
              <img
                src={srcStr}
                alt={alt ?? ''}
                loading="lazy"
                className="mx-auto block h-auto max-w-full rounded-[14px] border border-[var(--app-border)]"
              />
              {alt ? (
                <span className="text-subtle mt-2 block font-mono text-xs">{alt}</span>
              ) : null}
            </span>
          )
        },

        // Emphasis
        strong: ({ children }) => (
          <strong className="text-primary font-semibold">{children}</strong>
        ),
        em: ({ children }) => <em className="text-primary italic">{children}</em>,

        // Blockquotes
        blockquote: ({ children }) => (
          <blockquote className="border-info text-subtle my-3 border-l-4 pl-4 italic">
            {children}
          </blockquote>
        ),

        // Horizontal rule
        hr: () => <hr className="border-base my-4" />,

        // Tables (GFM)
        table: ({ children }) => (
          <div className="my-4 overflow-x-auto">
            <table className="border-base min-w-full rounded border">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-surface-raised">{children}</thead>,
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => <tr className="border-base border-b">{children}</tr>,
        th: ({ children }) => (
          <Text asChild kind="label/semibold/sm">
            <th className="text-primary px-3 py-2 text-left">{children}</th>
          </Text>
        ),
        td: ({ children }) => (
          <Text asChild kind="body/regular/sm">
            <td className="text-primary px-3 py-2">{children}</td>
          </Text>
        ),
      }),
      [compact, renderCitationLink]
    )

    return (
      <div className={`markdown-content break-words [overflow-wrap:anywhere] [&>*:last-child]:mb-0 ${className}`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {content}
        </ReactMarkdown>
      </div>
    )
  }
)

MarkdownRenderer.displayName = 'MarkdownRenderer'

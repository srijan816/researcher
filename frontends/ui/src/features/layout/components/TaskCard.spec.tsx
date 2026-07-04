// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import { vi, describe, test, expect, beforeEach } from 'vitest'
import { TaskCard } from './TaskCard'
import type { DeepResearchTodo } from '@/features/chat/types'

describe('TaskCard', () => {
  const createTodo = (overrides: Partial<DeepResearchTodo> = {}): DeepResearchTodo => ({
    id: 'todo-1',
    content: 'Research market trends',
    status: 'pending',
    ...overrides,
  })

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('basic rendering', () => {
    test('renders task content', () => {
      render(<TaskCard todo={createTodo({ content: 'Analyze competitors' })} />)

      expect(screen.getByText('Analyze competitors')).toBeInTheDocument()
    })

    test('renders a timeline dot', () => {
      const { container } = render(<TaskCard todo={createTodo()} />)

      expect(container.querySelector('.gx-timeline-dot')).toBeInTheDocument()
    })
  })

  describe('status badges', () => {
    test('shows "pending" badge for pending status', () => {
      render(<TaskCard todo={createTodo({ status: 'pending' })} />)

      expect(screen.getByText('pending')).toBeInTheDocument()
    })

    test('shows "in progress" badge for in_progress status', () => {
      render(<TaskCard todo={createTodo({ status: 'in_progress' })} />)

      expect(screen.getByText('in progress')).toBeInTheDocument()
    })

    test('shows "complete" badge for completed status', () => {
      render(<TaskCard todo={createTodo({ status: 'completed' })} />)

      expect(screen.getByText('complete')).toBeInTheDocument()
    })

    test('shows "stopped" badge for stopped status', () => {
      render(<TaskCard todo={createTodo({ status: 'stopped' })} />)

      expect(screen.getByText('stopped')).toBeInTheDocument()
    })
  })

  describe('timeline dot state', () => {
    test('dot is filled when task is completed', () => {
      const { container } = render(<TaskCard todo={createTodo({ status: 'completed' })} />)

      expect(container.querySelector('.gx-timeline-dot--done')).toBeInTheDocument()
    })

    test('dot pulses when task is in progress', () => {
      const { container } = render(<TaskCard todo={createTodo({ status: 'in_progress' })} />)

      expect(container.querySelector('.gx-timeline-dot--active')).toBeInTheDocument()
    })

    test('dot is faint when task is pending', () => {
      const { container } = render(<TaskCard todo={createTodo({ status: 'pending' })} />)

      expect(container.querySelector('.gx-timeline-dot--done')).not.toBeInTheDocument()
      expect(container.querySelector('.gx-timeline-dot--active')).not.toBeInTheDocument()
      expect(container.querySelector('.gx-timeline-dot')).toBeInTheDocument()
    })
  })

  describe('completed task styling', () => {
    test('completed task text is muted', () => {
      render(<TaskCard todo={createTodo({ status: 'completed', content: 'Done task' })} />)

      const taskText = screen.getByText('Done task')
      expect(taskText).toHaveClass('text-subtle')
    })

    test('non-completed task text is not muted', () => {
      render(<TaskCard todo={createTodo({ status: 'pending', content: 'Pending task' })} />)

      const taskText = screen.getByText('Pending task')
      expect(taskText).not.toHaveClass('text-subtle')
    })
  })
})

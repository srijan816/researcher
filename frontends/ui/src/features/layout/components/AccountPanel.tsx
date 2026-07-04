// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AccountPanel Component
 *
 * Right-side overlay panel for account management, reachable from the
 * NavRail user menu. Shows profile details, a change-password form, and
 * a sign-out action.
 */

'use client'

import { type FC, memo, useCallback, useState } from 'react'
import { Avatar, Banner, Button, Flex, SidePanel, Text } from '@/adapters/ui'
import { Lock, Logout } from '@/adapters/ui/icons'
import { changePassword } from '@/adapters/api'
import { useAuth } from '@/adapters/auth'
import { useLayoutStore } from '../store'

const MIN_PASSWORD_LENGTH = 8

/** HTTP status returned when the backend endpoint is not deployed */
const NOT_FOUND_STATUS = '404'

type PasswordStatus = { kind: 'success' | 'error'; message: string } | null

/**
 * Account panel — profile info, change password, sign out.
 */
export const AccountPanel: FC = memo(function AccountPanel() {
  const isOpen = useLayoutStore((s) => s.rightPanel === 'account')
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const { user, authRequired, signOut } = useAuth()

  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordStatus, setPasswordStatus] = useState<PasswordStatus>(null)
  const [isChangingPassword, setIsChangingPassword] = useState(false)

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        openRightPanel('account')
      } else {
        closeRightPanel()
      }
    },
    [openRightPanel, closeRightPanel]
  )

  const handleChangePassword = useCallback(async () => {
    setPasswordStatus(null)
    if (newPassword !== confirmPassword) {
      setPasswordStatus({ kind: 'error', message: 'New passwords do not match.' })
      return
    }
    if (newPassword.length < MIN_PASSWORD_LENGTH) {
      setPasswordStatus({
        kind: 'error',
        message: `New password must be at least ${MIN_PASSWORD_LENGTH} characters.`,
      })
      return
    }

    setIsChangingPassword(true)
    try {
      await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      setPasswordStatus({ kind: 'success', message: 'Password changed.' })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Password change failed.'
      setPasswordStatus({
        kind: 'error',
        message: message.includes(NOT_FOUND_STATUS)
          ? 'Password changes are temporarily unavailable. Please try again later.'
          : message,
      })
    } finally {
      setIsChangingPassword(false)
    }
  }, [confirmPassword, currentPassword, newPassword])

  const handleSignOut = useCallback(() => {
    closeRightPanel()
    void signOut()
  }, [closeRightPanel, signOut])

  const displayName = user?.name || 'User'
  const username = user?.id

  return (
    <SidePanel
      className="bg-surface-base top-[var(--header-height)] h-[calc(100dvh-var(--header-height)-var(--mobile-nav-height))] sm:h-[calc(100dvh-var(--header-height))] w-screen max-w-none rounded-none sm:max-w-[400px] sm:rounded-l-2xl"
      open={isOpen}
      onOpenChange={handleOpenChange}
      side="right"
      bordered
      closeOnClickOutside={false}
      slotHeading={
        <Flex align="center" gap="2">
          <Lock className="h-5 w-5" />
          Account
        </Flex>
      }
    >
      <Flex direction="col" gap="6">
        {/* Profile */}
        <Flex direction="col" gap="3">
          <Text kind="label/semibold/xs" className="text-subtle uppercase">
            Profile
          </Text>
          <Flex align="center" gap="3" className="rounded-md border border-base bg-surface-raised-30 px-3 py-3">
            <Avatar
              size="medium"
              src={user?.image || undefined}
              fallback={displayName.charAt(0).toUpperCase()}
            />
            <Flex direction="col" gap="1" className="min-w-0">
              <Text kind="label/bold/md" className="truncate text-primary">
                {displayName}
              </Text>
              {user?.email && (
                <Text kind="body/regular/sm" className="truncate text-subtle">
                  {user.email}
                </Text>
              )}
              {username && (
                <Text kind="body/regular/xs" className="truncate text-subtle">
                  Username: {username}
                </Text>
              )}
            </Flex>
          </Flex>
        </Flex>

        {/* Change password */}
        {authRequired && (
          <Flex direction="col" gap="3">
            <Flex align="center" gap="2">
              <Lock className="h-4 w-4" />
              <Text kind="label/semibold/xs" className="text-subtle uppercase">
                Change Password
              </Text>
            </Flex>

            <Flex direction="col" gap="2">
              <AccountPasswordInput
                id="account-current-password"
                label="Current password"
                value={currentPassword}
                onChange={setCurrentPassword}
                autoComplete="current-password"
              />
              <AccountPasswordInput
                id="account-new-password"
                label="New password"
                value={newPassword}
                onChange={setNewPassword}
                autoComplete="new-password"
              />
              <AccountPasswordInput
                id="account-confirm-password"
                label="Confirm new password"
                value={confirmPassword}
                onChange={setConfirmPassword}
                autoComplete="new-password"
              />
              <Button
                kind="secondary"
                size="small"
                onClick={() => void handleChangePassword()}
                disabled={isChangingPassword || !currentPassword || !newPassword || !confirmPassword}
                aria-label="Change password"
              >
                {isChangingPassword ? 'Changing…' : 'Change Password'}
              </Button>
            </Flex>

            {passwordStatus && (
              <Banner kind="inline" status={passwordStatus.kind}>
                <Text kind="body/regular/sm">{passwordStatus.message}</Text>
              </Banner>
            )}
          </Flex>
        )}

        {/* Sign out */}
        {authRequired && (
          <Flex direction="col" gap="3">
            <Text kind="label/semibold/xs" className="text-subtle uppercase">
              Session
            </Text>
            <Button
              kind="secondary"
              size="small"
              onClick={handleSignOut}
              aria-label="Sign out"
            >
              <Flex align="center" justify="center" gap="2">
                <Logout className="h-4 w-4" />
                Sign Out
              </Flex>
            </Button>
          </Flex>
        )}
      </Flex>
    </SidePanel>
  )
})

interface AccountPasswordInputProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  autoComplete: string
}

const AccountPasswordInput: FC<AccountPasswordInputProps> = ({ id, label, value, onChange, autoComplete }) => (
  <label className="flex flex-col gap-1" htmlFor={id}>
    <Text kind="label/regular/sm" className="text-subtle">
      {label}
    </Text>
    <input
      id={id}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      type="password"
      autoComplete={autoComplete}
      className="border-base bg-surface-raised text-primary h-9 rounded-md border px-3 text-sm outline-none focus:border-accent"
    />
  </label>
)

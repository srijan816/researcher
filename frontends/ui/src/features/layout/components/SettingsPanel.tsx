// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * SettingsPanel Component
 *
 * Right-side panel for application settings.
 * Contains appearance settings and service API key management.
 */

'use client'

import { type FC, memo, useCallback, useEffect, useState } from 'react'
import { Button, Flex, Text, SidePanel, Banner } from '@/adapters/ui'
import { Copy, Lock, Plus, Settings, Trash } from '@/adapters/ui/icons'
import { changePassword, createAPIKey, listAPIKeys, revokeAPIKey, type APIKeyMetadata } from '@/adapters/api'
import { useAuth } from '@/adapters/auth'
import { useLayoutStore } from '../store'

/**
 * Settings panel for application preferences.
 * Opens from the right side of the screen.
 */
export const SettingsPanel: FC = memo(function SettingsPanel() {
  const isOpen = useLayoutStore((s) => s.rightPanel === 'settings')
  const closeRightPanel = useLayoutStore((s) => s.closeRightPanel)
  const openRightPanel = useLayoutStore((s) => s.openRightPanel)
  const [apiKeys, setApiKeys] = useState<APIKeyMetadata[]>([])
  const [apiKeyName, setApiKeyName] = useState('External App')
  const [newApiKey, setNewApiKey] = useState<string | null>(null)
  const [apiKeyError, setApiKeyError] = useState<string | null>(null)
  const [isLoadingKeys, setIsLoadingKeys] = useState(false)
  const [isCreatingKey, setIsCreatingKey] = useState(false)
  const { authRequired, mustChangePassword } = useAuth()
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordStatus, setPasswordStatus] = useState<{ kind: 'success' | 'error'; message: string } | null>(null)
  const [isChangingPassword, setIsChangingPassword] = useState(false)

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) {
        openRightPanel('settings')
      } else {
        closeRightPanel()
      }
    },
    [openRightPanel, closeRightPanel]
  )

  const refreshApiKeys = useCallback(async () => {
    if (!isOpen) return
    setIsLoadingKeys(true)
    setApiKeyError(null)
    try {
      const response = await listAPIKeys()
      setApiKeys(response.api_keys)
    } catch (error) {
      setApiKeyError(error instanceof Error ? error.message : 'Failed to load API keys')
    } finally {
      setIsLoadingKeys(false)
    }
  }, [isOpen])

  useEffect(() => {
    void refreshApiKeys()
  }, [refreshApiKeys])

  const handleCreateApiKey = useCallback(async () => {
    setIsCreatingKey(true)
    setApiKeyError(null)
    try {
      const response = await createAPIKey(apiKeyName.trim() || 'API Key')
      setNewApiKey(response.key)
      setApiKeys((current) => [response, ...current])
    } catch (error) {
      setApiKeyError(error instanceof Error ? error.message : 'Failed to generate API key')
    } finally {
      setIsCreatingKey(false)
    }
  }, [apiKeyName])

  const handleRevokeApiKey = useCallback(async (id: string) => {
    setApiKeyError(null)
    try {
      await revokeAPIKey(id)
      setApiKeys((current) => current.filter((key) => key.id !== id))
    } catch (error) {
      setApiKeyError(error instanceof Error ? error.message : 'Failed to revoke API key')
    }
  }, [])

  const handleCopyApiKey = useCallback(async () => {
    if (!newApiKey) return
    await navigator.clipboard.writeText(newApiKey)
  }, [newApiKey])

  const handleChangePassword = useCallback(async () => {
    setPasswordStatus(null)
    if (newPassword !== confirmPassword) {
      setPasswordStatus({ kind: 'error', message: 'New passwords do not match.' })
      return
    }
    if (newPassword.length < 8) {
      setPasswordStatus({ kind: 'error', message: 'New password must be at least 8 characters.' })
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
      setPasswordStatus({
        kind: 'error',
        message: error instanceof Error ? error.message : 'Password change failed.',
      })
    } finally {
      setIsChangingPassword(false)
    }
  }, [confirmPassword, currentPassword, newPassword])

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
          <Settings className="h-5 w-5" />
          Settings
        </Flex>
      }
      slotFooter={
        <Text kind="body/regular/xs" className="text-subtle">
          Settings are saved automatically.
        </Text>
      }
    >
      <Flex direction="col" gap="6">
        {/* Appearance Section */}
        <Flex direction="col" gap="3">
          <Text kind="label/semibold/xs" className="text-subtle uppercase">
            UI Theme
          </Text>

          <Flex direction="col" gap="1" className="rounded-md border border-base bg-surface-raised-30 px-3 py-2">
            <Text kind="label/semibold/sm" className="text-primary">
              Dark
            </Text>
            <Text kind="body/regular/xs" className="text-subtle">
              Dark is the fixed interface theme for this deployment.
            </Text>
          </Flex>
        </Flex>

        <DividerLine />

        {authRequired && (
          <>
            <Flex direction="col" gap="3">
              <Flex align="center" gap="2">
                <Lock className="h-4 w-4" />
                <Text kind="label/semibold/xs" className="text-subtle uppercase">
                  Password
                </Text>
              </Flex>

              {mustChangePassword && (
                <Banner kind="inline" status="warning">
                  <Text kind="body/regular/sm">
                    This account is using its default password. Change it before sharing access.
                  </Text>
                </Banner>
              )}

              <Flex direction="col" gap="2">
                <PasswordInput
                  id="current-password"
                  label="Current password"
                  value={currentPassword}
                  onChange={setCurrentPassword}
                  autoComplete="current-password"
                />
                <PasswordInput
                  id="new-password"
                  label="New password"
                  value={newPassword}
                  onChange={setNewPassword}
                  autoComplete="new-password"
                />
                <PasswordInput
                  id="confirm-password"
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
                  Change Password
                </Button>
              </Flex>

              {passwordStatus && (
                <Banner kind="inline" status={passwordStatus.kind}>
                  <Text kind="body/regular/sm">{passwordStatus.message}</Text>
                </Banner>
              )}
            </Flex>

            <DividerLine />
          </>
        )}

        <Flex direction="col" gap="3">
          <Flex align="center" gap="2">
            <Lock className="h-4 w-4" />
            <Text kind="label/semibold/xs" className="text-subtle uppercase">
              API Keys
            </Text>
          </Flex>

          <Flex direction="col" gap="2">
            <label className="sr-only" htmlFor="api-key-name">API key name</label>
            <input
              id="api-key-name"
              value={apiKeyName}
              onChange={(event) => setApiKeyName(event.target.value)}
              className="border-base bg-surface-raised text-primary h-9 rounded-md border px-3 text-sm outline-none focus:border-accent"
              maxLength={120}
            />
            <Button
              kind="secondary"
              size="small"
              onClick={handleCreateApiKey}
              disabled={isCreatingKey}
              aria-label="Generate API key"
            >
              <Flex align="center" gap="1">
                <Plus className="h-4 w-4" />
                Generate Key
              </Flex>
            </Button>
          </Flex>

          {newApiKey && (
            <Banner kind="inline" status="success">
              <Flex direction="col" gap="2">
                <Text kind="body/regular/sm">
                  Copy this key now. It will not be shown again.
                </Text>
                <Flex align="center" gap="2" className="min-w-0">
                  <code className="border-base bg-surface-base min-w-0 flex-1 overflow-x-auto rounded border px-2 py-1 text-xs">
                    {newApiKey}
                  </code>
                  <Button kind="tertiary" size="small" onClick={handleCopyApiKey} aria-label="Copy API key">
                    <Copy className="h-4 w-4" />
                  </Button>
                </Flex>
              </Flex>
            </Banner>
          )}

          {apiKeyError && (
            <Banner kind="inline" status="error">
              <Text kind="body/regular/sm">{apiKeyError}</Text>
            </Banner>
          )}

          <Flex direction="col" gap="2">
            {isLoadingKeys ? (
              <Text kind="body/regular/sm" className="text-subtle">Loading keys...</Text>
            ) : apiKeys.length === 0 ? (
              <Text kind="body/regular/sm" className="text-subtle">No API keys yet.</Text>
            ) : (
              apiKeys.map((key) => (
                <Flex
                  key={key.id}
                  align="center"
                  justify="between"
                  gap="3"
                  className="border-base rounded-md border px-3 py-2"
                >
                  <Flex direction="col" gap="1" className="min-w-0">
                    <Text kind="body/semibold/sm" className="truncate">{key.name}</Text>
                    <Text kind="body/regular/xs" className="text-subtle truncate">
                      {key.prefix}... - last used {key.last_used_at || 'never'}
                    </Text>
                  </Flex>
                  <Button
                    kind="tertiary"
                    size="small"
                    onClick={() => void handleRevokeApiKey(key.id)}
                    aria-label={`Revoke API key ${key.name}`}
                    title="Revoke API key"
                  >
                    <Trash className="h-4 w-4" />
                  </Button>
                </Flex>
              ))
            )}
          </Flex>
        </Flex>
      </Flex>
    </SidePanel>
  )
})

const DividerLine: FC = () => (
  <div className="border-base border-t" />
)

interface PasswordInputProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  autoComplete: string
}

const PasswordInput: FC<PasswordInputProps> = ({ id, label, value, onChange, autoComplete }) => (
  <label className="flex flex-col gap-1" htmlFor={id}>
    <Text kind="label/regular/sm" className="text-subtle">{label}</Text>
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

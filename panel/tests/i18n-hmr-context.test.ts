import { describe, expect, it, vi } from 'vitest'

describe('i18n HMR context identity', () => {
  it('reuses one context object across module re-evaluation', async () => {
    const first = await import('../src/renderer/src/i18n/context')

    vi.resetModules()
    const second = await import('../src/renderer/src/i18n/context')

    expect(second.I18nContext).toBe(first.I18nContext)
  })

  it('uses readable English instead of raw keys without a mounted provider', async () => {
    const { I18nContext } = await import('../src/renderer/src/i18n/context')
    const fallback = (I18nContext as any)._currentValue

    expect(fallback.t('app.mode.chat')).toBe('Chat')
    expect(fallback.t('layout.sidebarHeader.chats')).toBe('Chats')
  })
})

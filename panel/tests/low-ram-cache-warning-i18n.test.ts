import { describe, it, expect } from 'vitest'
import en from '../src/renderer/src/i18n/locales/en.json'
import es from '../src/renderer/src/i18n/locales/es.json'
import ja from '../src/renderer/src/i18n/locales/ja.json'
import ko from '../src/renderer/src/i18n/locales/ko.json'
import zh from '../src/renderer/src/i18n/locales/zh.json'

/**
 * The low-RAM cache recommendation is the one piece of cache guidance a user
 * on a 16-32GB Mac most needs to see, and it is shown in the app rather than
 * only in the engine startup log. A missing translation would silently fall
 * back to the key or to English for that user, so every locale must carry it
 * and every locale must keep the {{totalGB}} placeholder.
 */
const LOCALES: Record<string, any> = { en, es, ja, ko, zh }

describe('low-RAM paged-cache warning', () => {
  it.each(Object.keys(LOCALES))('%s has the warning string', (name) => {
    const s = LOCALES[name]?.sessions?.config?.lowRamPagedCacheWarning
    expect(typeof s, `${name} is missing sessions.config.lowRamPagedCacheWarning`).toBe('string')
    expect(s.length).toBeGreaterThan(40)
  })

  it.each(Object.keys(LOCALES))('%s keeps the {{totalGB}} placeholder', (name) => {
    expect(LOCALES[name].sessions.config.lowRamPagedCacheWarning).toContain('{{totalGB}}')
  })

  it('says it is a recommendation, not an enforced limit, in every locale', () => {
    // An invented RAM guard once refused to load big models for six releases.
    // This string must never read as though something was disabled.
    const forbidden = /disabled automatically|has been disabled|blocked/i
    for (const [name, cat] of Object.entries(LOCALES)) {
      const s = (cat as any).sessions.config.lowRamPagedCacheWarning
      const claimsEnforcement = /\bwe (disabled|turned off)\b/i.test(s)
      expect(claimsEnforcement, `${name} implies enforcement`).toBe(false)
      expect(forbidden.test(s) && !/Nothing is disabled|No se desactiva nada|自動的に無効化されるものはありません|자동으로 비활성화되는 것은 없으며|不会自动禁用/.test(s)).toBe(false)
    }
  })
})

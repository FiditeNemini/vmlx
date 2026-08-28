import { describe, it, expect } from 'vitest'
import en from '../src/renderer/src/i18n/locales/en.json'
import es from '../src/renderer/src/i18n/locales/es.json'
import ja from '../src/renderer/src/i18n/locales/ja.json'
import ko from '../src/renderer/src/i18n/locales/ko.json'
import zh from '../src/renderer/src/i18n/locales/zh.json'

/**
 * Every user sees this notice at the top of Session Settings, in their own
 * language. The product policy moved from "RAM cache is a 2% trade-off you
 * may toggle" to the SSD-only tier: In-Memory Paged Cache and Media
 * Preprocess RAM Cache are locked OFF for every model, prompts persist on
 * SSD, and payloads restore transiently. The notice must describe THAT
 * policy — the old 2% claim would misdescribe the shipping behavior. A
 * missing translation silently falls back to English, and this is exactly
 * the audience that would be left behind.
 */
const LOCALES: Record<string, any> = { en, es, ja, ko, zh }

describe('RAM cache trade-off notice', () => {
  it.each(Object.keys(LOCALES))('%s has the notice', (name) => {
    const s = LOCALES[name]?.sessions?.config?.ramCacheTradeoffNotice
    expect(typeof s, `${name} is missing sessions.config.ramCacheTradeoffNotice`).toBe('string')
    expect(s.length).toBeGreaterThan(60)
  })

  it.each(Object.keys(LOCALES))('%s points at the actual setting', (name) => {
    // Telling someone to change a setting without naming it is useless.
    const s = LOCALES[name].sessions.config.ramCacheTradeoffNotice
    expect(/RAM/.test(s) || /메모리|メモリ|内存|Memoria/.test(s)).toBe(true)
    expect(/SSD/.test(s)).toBe(true)
  })

  it.each(Object.keys(LOCALES))('%s no longer carries the superseded 2%% trade-off claim', (name) => {
    // The SSD-only tier locked paged RAM off for every model; a "costs about
    // 2%" toggle description would misdescribe the shipping product.
    expect(LOCALES[name].sessions.config.ramCacheTradeoffNotice).not.toMatch(/2\s*%/)
  })

  it('never claims the app changed anything on the user behalf', () => {
    // An invented RAM guard once refused to load big models for six releases.
    for (const [name, cat] of Object.entries(LOCALES)) {
      const s = (cat as any).sessions.config.ramCacheTradeoffNotice
      expect(/\bwe (disabled|turned off|switched)\b/i.test(s), `${name} implies enforcement`).toBe(false)
    }
  })

  it('the superseded low-RAM-only string is gone from every locale', () => {
    for (const [name, cat] of Object.entries(LOCALES)) {
      expect((cat as any).sessions.config.lowRamPagedCacheWarning, `${name} still has the old key`).toBeUndefined()
    }
  })
})

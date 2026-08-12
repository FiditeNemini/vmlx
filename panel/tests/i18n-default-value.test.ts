import { describe, expect, it } from 'vitest'
import { translateFromCatalog } from '../src/renderer/src/i18n/translate'

/**
 * Load progress arrives from the MAIN process as `{ labelKey, label }`, and
 * SessionCard/SessionView pass the English `label` as `defaultValue` so a key
 * the renderer's catalog does not carry still renders words.
 *
 * `translateFromCatalog` ignored it and fell back to the raw key, so any
 * main/renderer skew — a new load-progress pattern shipped without its key, or
 * an updated main against an older renderer bundle — put a dotted key straight
 * into the progress bar. The defaultValue existed for exactly that case and was
 * doing nothing.
 */
describe('translateFromCatalog defaultValue', () => {
  const en = { sessions: { loadProgress: { ready: 'Ready' } } }

  it('prefers the catalog when the key exists', () => {
    expect(
      translateFromCatalog(en, en, 'sessions.loadProgress.ready', {
        defaultValue: 'Ready (from main)',
      }),
    ).toBe('Ready')
  })

  it('uses defaultValue for a key neither catalog carries', () => {
    expect(
      translateFromCatalog(en, en, 'sessions.loadProgress.brandNew', {
        defaultValue: 'Materializing weights',
      }),
    ).toBe('Materializing weights')
  })

  it('still falls back to the key when no defaultValue is supplied', () => {
    expect(translateFromCatalog(en, en, 'sessions.loadProgress.brandNew')).toBe(
      'sessions.loadProgress.brandNew',
    )
  })

  it('ignores an empty or non-string defaultValue', () => {
    expect(
      translateFromCatalog(en, en, 'a.missing.key', { defaultValue: '' }),
    ).toBe('a.missing.key')
    expect(
      translateFromCatalog(en, en, 'a.missing.key', { defaultValue: 42 }),
    ).toBe('a.missing.key')
  })

  it('interpolates a defaultValue too', () => {
    expect(
      translateFromCatalog(en, en, 'a.missing.key', {
        defaultValue: 'Loading {{name}}',
        name: 'DSV4',
      }),
    ).toBe('Loading DSV4')
  })

  it('falls back through the secondary catalog before defaultValue', () => {
    expect(
      translateFromCatalog({}, en, 'sessions.loadProgress.ready', {
        defaultValue: 'from main',
      }),
    ).toBe('Ready')
  })
})

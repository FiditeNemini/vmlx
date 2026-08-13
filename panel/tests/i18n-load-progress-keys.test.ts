import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The main process cannot translate — it has no locale catalog — so each load
 * progress entry ships an i18n `labelKey` alongside the English `label`, and
 * the renderer resolves the key with `label` as `defaultValue`.
 *
 * That defaultValue is why this test has to exist. A key the catalogs do not
 * carry does not throw, does not warn, and does not render a dotted key — it
 * silently serves English to every user in every language. The whole
 * `sessions.loadProgress.*` namespace was emitted for exactly that long while
 * the catalogs only ever defined `main.loadProgress.*`, so all 31 keys were
 * unresolved in all five locales and nobody could see it.
 *
 * Asserting the keys resolve is therefore the only thing that can catch a
 * namespace or rename drift between the emitter and the catalogs.
 */

const LOCALES = ['en', 'es', 'ja', 'ko', 'zh'] as const
const ROOT = join(__dirname, '..')

function loadCatalog(locale: string): Record<string, unknown> {
  return JSON.parse(
    readFileSync(
      join(ROOT, 'src/renderer/src/i18n/locales', `${locale}.json`),
      'utf-8',
    ),
  )
}

/** Resolve a dotted key against a nested catalog, like the renderer does. */
function resolve(catalog: Record<string, unknown>, key: string): string | undefined {
  let node: unknown = catalog
  for (const part of key.split('.')) {
    if (typeof node !== 'object' || node === null || !(part in node)) return undefined
    node = (node as Record<string, unknown>)[part]
  }
  return typeof node === 'string' ? node : undefined
}

/** Every labelKey the main process actually emits for load progress. */
function emittedLabelKeys(): string[] {
  const src = readFileSync(join(ROOT, 'src/main/sessions.ts'), 'utf-8')
  return [...src.matchAll(/labelKey:\s*'([^']+)'/g)].map(m => m[1])
}

describe('load-progress i18n keys', () => {
  const keys = emittedLabelKeys()

  // Guards the guard: if the emitter is refactored so this regex stops
  // matching, every assertion below would pass over an empty list.
  it('finds the load progress table in the main process', () => {
    expect(keys.length).toBeGreaterThan(20)
    expect(new Set(keys).size).toBeGreaterThan(20)
  })

  it.each(LOCALES)('resolves every emitted labelKey in %s', locale => {
    const catalog = loadCatalog(locale)
    const unresolved = [...new Set(keys)].filter(k => resolve(catalog, k) === undefined)
    expect(unresolved).toEqual([])
  })

  it('keeps the key set identical across every locale', () => {
    const flatten = (node: unknown, prefix = ''): string[] =>
      typeof node === 'object' && node !== null
        ? Object.entries(node as Record<string, unknown>).flatMap(([k, v]) =>
            flatten(v, `${prefix}${k}.`),
          )
        : [prefix.slice(0, -1)]

    const base = flatten(loadCatalog('en')).sort()
    for (const locale of LOCALES.filter(l => l !== 'en')) {
      expect({ locale, keys: flatten(loadCatalog(locale)).sort() }).toEqual({
        locale,
        keys: base,
      })
    }
  })
})

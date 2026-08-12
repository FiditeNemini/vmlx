import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The Cache panel's reuse explainer used to promise, flatly, that reuse "matches
 * the longest continuous causal token prefix from token 0". That overpromises
 * for path-dependent families: MEASURED, a prompt shortened by ~130 tokens —
 * well inside a single 256-token block — reused NOTHING, while a same-length
 * divergent tail correctly reused 1792/1878.
 *
 * It was also the only untranslated string on the panel, so four of five locales
 * showed English regardless of the user's language.
 */
const ROOT = join(__dirname, '..')
const PANEL = join(ROOT, 'src/renderer/src/components/sessions/CachePanel.tsx')
const LOCALES = join(ROOT, 'src/renderer/src/i18n/locales')
const KEY = 'sessions.cachePanel.reuseExplainer'

function resolve(catalog: Record<string, unknown>, dotted: string): unknown {
  let node: unknown = catalog
  for (const part of dotted.split('.')) {
    if (typeof node !== 'object' || node === null || !(part in node)) return undefined
    node = (node as Record<string, unknown>)[part]
  }
  return node
}

describe('cache panel reuse explainer', () => {
  it('renders through i18n rather than a hardcoded string', () => {
    const source = readFileSync(PANEL, 'utf8')
    expect(source).toContain(`t('${KEY}')`)
    expect(source).not.toContain('Reuse matches the longest continuous causal token prefix')
  })

  it('every locale defines it as a non-empty string', () => {
    const files = readdirSync(LOCALES).filter((f) => f.endsWith('.json'))
    expect(files.length).toBeGreaterThan(1)
    for (const file of files) {
      const value = resolve(JSON.parse(readFileSync(join(LOCALES, file), 'utf8')), KEY)
      expect(typeof value, `${file} is missing ${KEY}`).toBe('string')
      expect((value as string).trim().length, `${file} has an empty ${KEY}`).toBeGreaterThan(0)
    }
  })

  it('the English copy states the path-dependent caveat', () => {
    const en = resolve(
      JSON.parse(readFileSync(join(LOCALES, 'en.json'), 'utf8')),
      KEY,
    ) as string
    // The caveat is the whole point — a shorter prompt can reuse nothing.
    expect(en.toLowerCase()).toContain('shorter')
    expect(en).toMatch(/DeepSeek V4|ZAYA|rotating/i)
  })
})

describe('hybrid SSM session timeout', () => {
  /**
   * FOUND LIVE: a 101,502-token prompt to Qwen3.6-27B rendered "Message failed -
   * Request timed out after 300s" while the engine served it in ~230s. A script
   * passes its own long timeout and never sees this; only a real user hits the
   * session default.
   *
   * The rule was written out SEVEN times and four copies had diverged. It now
   * lives once, in panel/src/shared/slowFamilyTimeouts.ts, so assert THAT and
   * assert every consumer imports it rather than re-declaring a local table.
   */
  const SHARED = join(ROOT, 'src/shared/slowFamilyTimeouts.ts')

  it('the shared table covers every slow family, by registry name', () => {
    const src = readFileSync(SHARED, 'utf8')
    expect(src).toContain('SLOW_FAMILY_TIMEOUT_SECONDS = 900')
    // Registry names, NOT engine family_name — two earlier fixes were silent
    // no-ops because they used the engine spellings (qwen3_5, nemotron_h).
    for (const family of [
      'deepseek-v4',
      'minimax_m3',
      'openpangu_v2',
      'qwen3.5',
      'qwen3.5-moe',
      'qwen3-next',
      'nemotron-h',
    ]) {
      expect(src, `shared table is missing ${family}`).toContain(family)
    }
  })

  it('every timeout consumer reads the shared table', () => {
    for (const rel of [
      'src/main/sessions.ts',
      'src/main/ipc/chat.ts',
      'src/main/api-gateway.ts',
      'src/renderer/src/components/sessions/SessionSettings.tsx',
    ]) {
      const src = readFileSync(join(ROOT, rel), 'utf8')
      expect(src, `${rel} does not import the shared timeout table`).toMatch(
        /slowFamilyTimeouts/,
      )
      // and must not have grown a local copy back
      expect(src, `${rel} re-declared a local slow-family table`).not.toMatch(
        /const\s+\w*SLOW_FAMILY\w*\s*:\s*Record/,
      )
    }
  })
})

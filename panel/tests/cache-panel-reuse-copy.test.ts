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
   * FOUND LIVE IN THE APP, not by API testing: a 101,502-token prompt to
   * Qwen3.6-27B rendered "Message failed - Request timed out after 300s" in the
   * chat, while the engine served that exact prompt in ~230s of prefill plus
   * decode. A script passes its own long timeout and never sees this; only a
   * real user hits the session default.
   *
   * The engine-side slow-family bump cannot help here: the panel ALWAYS emits
   * --timeout, so the engine sees it as explicitly set and leaves it alone.
   * The default has to be right on the panel side.
   */
  it('gives hybrid SSM families the 900s default', () => {
    const src = readFileSync(join(ROOT, 'src/main/sessions.ts'), 'utf8')
    expect(src).toContain('HYBRID_SSM_DEFAULT_TIMEOUT_SECONDS = 900')
    const setMatch = src.match(/HYBRID_SSM_TIMEOUT_FAMILIES = new Set\(\[([^\]]*)\]/)
    expect(setMatch, 'hybrid timeout family set is gone').toBeTruthy()
    // PANEL registry names — the registry maps the engine's qwen3_5/qwen3_next/
    // nemotron_h to these. Asserting the engine spellings here is what let the
    // original bug through: the set looked populated but matched nothing.
    for (const family of ['qwen3.5', 'qwen3-next', 'nemotron-h']) {
      expect(setMatch![1]).toContain(family)
    }
    // and it must be consulted by the resolver, not just declared
    expect(src).toContain('HYBRID_SSM_TIMEOUT_FAMILIES.has(normalizedFamily)')
  })
})

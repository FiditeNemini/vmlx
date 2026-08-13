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
  // Quoted `main.loadProgress.*` rather than `labelKey: '...'`, because one
  // emit picks its key with a ternary and would otherwise go unchecked.
  return [...src.matchAll(/'(main\.loadProgress\.[A-Za-z0-9_]+)'/g)].map(m => m[1])
}

/** Body of every `session:loadProgress` emit in the main process. */
function loadProgressEmitBodies(): { line: number; body: string }[] {
  const src = readFileSync(join(ROOT, 'src/main/sessions.ts'), 'utf-8')
  const out: { line: number; body: string }[] = []
  const re = /\.emit\(\s*'session:loadProgress'\s*,\s*\{/g
  let m: RegExpExecArray | null
  while ((m = re.exec(src))) {
    let depth = 0
    const start = re.lastIndex - 1
    let j = start
    for (; j < src.length; j++) {
      if (src[j] === '{') depth++
      else if (src[j] === '}') {
        depth--
        if (depth === 0) break
      }
    }
    out.push({ line: src.slice(0, m.index).split('\n').length, body: src.slice(start, j + 1) })
  }
  return out
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

  it('EVERY session:loadProgress emit ships a labelKey', () => {
    // The pattern table was fixed first and six other emits were missed —
    // 'Connected', 'Model ready', 'Waking from sleep...', 'Scanning model
    // files...', 'Model runtime still loading...' and the resident-RAM phase.
    // Three of those already HAD catalog entries, so translations existed and
    // sat unreachable purely because the emit never sent a key. Checking only
    // the keys that are sent can never catch an emit that sends none.
    const emits = loadProgressEmitBodies()
    expect(emits.length).toBeGreaterThan(5)
    const missing = emits.filter(e => !e.body.includes('labelKey')).map(e => e.line)
    expect(missing).toEqual([])
  })

  it('each key maps to the English text it is emitted beside', () => {
    // Resolving is not enough — a key can resolve to the WRONG entry. The
    // repoint mapped by exact English text precisely because eight keys had
    // been renamed, and two of them differ only by a word:
    // `loadingJangVl` is "Loading JANG VL model..." while `loadingJangVlShort`
    // is "Loading JANG VL...". Swapping them resolves fine and silently shows
    // the wrong phase in every language. Pinning en[labelKey] === label is what
    // makes a mis-mapping fail.
    const src = readFileSync(join(ROOT, 'src/main/sessions.ts'), 'utf-8')
    const pairs = [...src.matchAll(/label:\s*'([^']+)',\s*labelKey:\s*'([^']+)'/g)]
    expect(pairs.length).toBeGreaterThan(20)

    const en = loadCatalog('en')
    const mismatched = pairs
      .map(([, label, key]) => ({ key, label, en: resolve(en, key) }))
      .filter(r => r.en !== r.label)
    expect(mismatched).toEqual([])
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

/**
 * The generalisation of the same defect.
 *
 * Catalog-vs-catalog parity checks — "all five locales carry the same 1958
 * keys" — CANNOT catch this class, because the broken keys were missing from
 * every catalog at once and so parity held perfectly the whole time. A
 * renderer-side `t('...')` sweep cannot catch it either: the key is a bare
 * string literal produced in the MAIN process, which has no catalog to check
 * it against, and only becomes an i18n key once the renderer feeds it to t().
 *
 * So sweep every `labelKey:` literal the main process emits, wherever it lives,
 * and require it to resolve. Keyed on `labelKey:` rather than on
 * "dotted string in a known namespace", which false-positives on hostnames
 * (`api.openai.com`) and filenames (`image.png`).
 */
describe('every main-process labelKey resolves', () => {
  function mainProcessLabelKeys(): { key: string; file: string }[] {
    const files = [
      'src/main/sessions.ts',
      'src/main/server.ts',
      'src/main/ipc/chat.ts',
      'src/main/ipc/image.ts',
    ]
    const found: { key: string; file: string }[] = []
    for (const file of files) {
      let src: string
      try {
        src = readFileSync(join(ROOT, file), 'utf-8')
      } catch {
        continue // file may be reorganised; the sweep below still guards the rest
      }
      for (const m of src.matchAll(/labelKey:\s*['"]([A-Za-z0-9_.]+)['"]/g)) {
        found.push({ key: m[1], file })
      }
    }
    return found
  }

  it('finds labelKey emitters to check', () => {
    // Positive control: if this hits zero the sweep below is vacuous.
    expect(mainProcessLabelKeys().length).toBeGreaterThan(20)
  })

  it('resolves every emitted labelKey in every locale', () => {
    const emitted = mainProcessLabelKeys()
    const unresolved: string[] = []
    for (const locale of LOCALES) {
      const catalog = loadCatalog(locale)
      for (const { key, file } of emitted) {
        if (resolve(catalog, key) === undefined) {
          unresolved.push(`${locale}: ${key} (${file})`)
        }
      }
    }
    expect(unresolved).toEqual([])
  })
})

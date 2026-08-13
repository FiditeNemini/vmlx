import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { normalizeHfTokenSetting } from '../src/shared/hfSettings'

/**
 * `hf_api_key` is read in several main-process places and pushed into
 * `HF_TOKEN` for spawned engines and download workers. A shared
 * `normalizeHfTokenSetting` exists for exactly that, but only some readers
 * used it: the session spawn path, the developer-console env builder and the
 * image-download token check all read `db.getSetting` raw.
 *
 * Both UI save paths trim before storing, so this was latent rather than
 * active. It becomes real for a token written by an older build, a migration,
 * or a direct DB edit: a trailing newline reaches the engine as
 * `Authorization: Bearer hf_xxx\n`, which fails as an AUTH error rather than as
 * the formatting problem it actually is — and a whitespace-only value passes a
 * bare truthiness check, suppressing the "no token set" warning.
 *
 * One rule, several enforcement points: pin that every reader normalizes.
 */

const ROOT = join(__dirname, '..')

function sourceOf(rel: string): string {
  return readFileSync(join(ROOT, rel), 'utf-8')
}

/** Strip comments so prose mentioning the raw call cannot satisfy a scan. */
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^[ \t]*\/\/.*$/gm, '')
}

const READERS = [
  'src/main/sessions.ts',
  'src/main/ipc/developer.ts',
  'src/main/ipc/models.ts',
]

describe('every main-process hf_api_key reader normalizes', () => {
  it('finds readers to check', () => {
    // Positive control: a rename that moves these files must fail loudly
    // rather than turn the sweep below into a no-op.
    const withReads = READERS.filter((f) =>
      stripComments(sourceOf(f)).includes('hf_api_key'),
    )
    expect(withReads.length).toBe(READERS.length)
  })

  it('wraps every getSetting("hf_api_key") in normalizeHfTokenSetting', () => {
    const offenders: string[] = []
    for (const file of READERS) {
      const src = stripComments(sourceOf(file))
      // Match the read expression and require the normalizer around it.
      const raw = [...src.matchAll(/([A-Za-z.]*getSetting\(\s*['"]hf_api_key['"]\s*\))/g)]
      for (const m of raw) {
        const before = src.slice(Math.max(0, m.index! - 60), m.index!)
        if (!before.includes('normalizeHfTokenSetting(')) {
          offenders.push(`${file}: ${m[1]}`)
        }
      }
    }
    expect(offenders).toEqual([])
  })
})

describe('normalizeHfTokenSetting contract', () => {
  it('drops a token that is only whitespace', () => {
    expect(normalizeHfTokenSetting('   ')).toBeNull()
    expect(normalizeHfTokenSetting('\n')).toBeNull()
  })

  it('strips the trailing newline a copy-paste leaves behind', () => {
    expect(normalizeHfTokenSetting('hf_abc123\n')).toBe('hf_abc123')
    expect(normalizeHfTokenSetting('  hf_abc123  ')).toBe('hf_abc123')
  })

  it('passes a clean token through unchanged', () => {
    expect(normalizeHfTokenSetting('hf_abc123')).toBe('hf_abc123')
  })

  it('treats absent settings as no token', () => {
    expect(normalizeHfTokenSetting(null)).toBeNull()
    expect(normalizeHfTokenSetting(undefined)).toBeNull()
  })
})

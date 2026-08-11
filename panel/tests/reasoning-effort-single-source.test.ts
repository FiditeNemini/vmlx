import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'fs'
import { join } from 'path'
import { REASONING_EFFORT_LEVELS } from '../src/shared/reasoningEffortPolicy'
import { normalizeReasoningEffort } from '../src/shared/reasoningEffortPolicy'
import { sanitizeChatOverrides } from '../src/main/chat-override-policy'

/**
 * Adding a reasoning level has repeatedly half-landed because the level list is
 * re-typed by hand in several places. Muse's `xhigh` rendered as a button and the
 * engine honoured it, yet:
 *   - remoteModelCapabilities filtered it out, so remote models never offered it;
 *   - chat-override-policy dropped it on save, so the choice did not persist.
 * Both were separate hand-rolled Sets. This test fails the moment a new one appears.
 */
const root = join(__dirname, '..', 'src')

function walk(dir: string, out: string[] = []): string[] {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (/\.(ts|tsx)$/.test(p)) out.push(p)
  }
  return out
}

describe('reasoning effort levels have a single source of truth', () => {
  it('xhigh is canonical', () => {
    expect(REASONING_EFFORT_LEVELS).toContain('xhigh')
    expect(normalizeReasoningEffort('xhigh')).toBe('xhigh')
  })

  it('a chat override of xhigh survives sanitisation (it used to be dropped)', () => {
    const out = sanitizeChatOverrides({ chatId: 'c', reasoningEffort: 'xhigh' } as any)
    expect(out.reasoningEffort).toBe('xhigh')
  })

  it('no source file hand-rolls the level list', () => {
    // A literal list of every level EXCEPT xhigh is the exact drift signature.
    const bad: string[] = []
    for (const f of walk(root)) {
      const src = readFileSync(f, 'utf8')
      for (const line of src.split('\n')) {
        if (/\.test\.|\.spec\./.test(f)) continue
        const hasAll = /['"]low['"]/.test(line) && /['"]medium['"]/.test(line) &&
                       /['"]high['"]/.test(line) && /['"]max['"]/.test(line)
        if (hasAll && !/xhigh/.test(line)) bad.push(`${f.replace(root, 'src')}: ${line.trim().slice(0, 90)}`)
      }
    }
    expect(bad, `hand-rolled effort list(s) missing xhigh:\n${bad.join('\n')}`).toEqual([])
  })
})

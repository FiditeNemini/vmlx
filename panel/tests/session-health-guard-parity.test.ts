import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * `session:health` is emitted from four places and consumed by two components.
 *
 * SessionDashboard keyed on `data.status === 'ok'`, which only ONE of the four
 * emits sends. The other three — remote healthy, remote busy, local busy on a
 * long prefill — carry `running: true` and no `status`, so the guard declined
 * every one of them and the dashboard could never reconcile a stale remote
 * session. SessionsContext had keyed on `running` the whole time, so the two
 * consumers of the same event disagreed about what "healthy" means.
 *
 * A guard is only as good as the field it reads: keying on a field that most
 * producers do not send fails silently and looks exactly like "no events".
 */

const ROOT = join(__dirname, '..')
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8')

/** Field names in each `session:health` payload, by brace matching. */
function healthEmitPayloads(): string[][] {
  const src = read('src/main/sessions.ts')
  const out: string[][] = []
  const re = /\.emit\(\s*'session:health'\s*,\s*\{/g
  let m: RegExpExecArray | null
  while ((m = re.exec(src))) {
    let depth = 0
    let j = m.end !== undefined ? m.index : m.index
    j = re.lastIndex - 1
    const start = j
    for (; j < src.length; j++) {
      if (src[j] === '{') depth++
      else if (src[j] === '}') {
        depth--
        if (depth === 0) break
      }
    }
    const body = src.slice(start, j + 1)
    out.push([...body.matchAll(/[{,]\s*([a-zA-Z_]\w*)\s*[,:}]/g)].map(x => x[1]))
  }
  return out
}

describe('session:health guard parity', () => {
  const payloads = healthEmitPayloads()

  it('finds every session:health emit', () => {
    // Positive control: if the emits move or are reshaped, fail loudly rather
    // than asserting over an empty list.
    expect(payloads.length).toBeGreaterThanOrEqual(4)
  })

  it('every emit carries `running`, so it is safe to key on', () => {
    for (const fields of payloads) expect(fields).toContain('running')
  })

  it('`status` is NOT on every emit, so keying on it drops events', () => {
    // This is the defect, pinned. If a future change adds `status` everywhere
    // this can be revisited — until then, `status` is the wrong field.
    const withStatus = payloads.filter(f => f.includes('status')).length
    expect(withStatus).toBeGreaterThan(0)
    expect(withStatus).toBeLessThan(payloads.length)
  })

  it('both consumers key on the same field', () => {
    for (const file of [
      'src/renderer/src/components/sessions/SessionDashboard.tsx',
      'src/renderer/src/contexts/SessionsContext.tsx',
    ]) {
      const src = read(file)
      const start = src.indexOf('onHealth')
      expect(start).toBeGreaterThan(-1)
      const handler = src.slice(start, start + 1600)
      expect(handler).toMatch(/data\??\.running/)
      expect(handler).not.toMatch(/data\??\.?status\s*===\s*'ok'/)
    }
  })
})

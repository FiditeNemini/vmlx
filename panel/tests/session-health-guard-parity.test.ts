import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * `session:health` is emitted from four places and consumed by two components.
 *
 * SessionDashboard keyed on `data.status === 'ok'`, which only ONE of the four
 * emits sends. The other three — remote healthy, remote busy, local busy on a
 * long prefill — send no `status`, so the guard declined every one of them and
 * the dashboard could never reconcile a stale remote session. SessionsContext
 * had keyed on `running` the whole time, so the two consumers of the same event
 * disagreed about what "healthy" means.
 *
 * A guard is only as good as the field it reads: keying on a field that most
 * producers do not send fails silently and looks exactly like "no events".
 *
 * Moving the guard onto `running` then exposed a second problem in the
 * producers. Both busy branches had hardcoded `running: true` — harmless while
 * nothing read it, wrong the moment something did, because those branches fire
 * for a still-loading server that has not opened /health yet. They now derive
 * it from session state. Fixing a consumer can promote a latent producer bug
 * into a live one.
 */

const ROOT = join(__dirname, '..')
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8')

/**
 * Strip comments before analysing an emit body. Without this the assertions
 * read the prose: a comment explaining why `running: true` is wrong matches a
 * /running:\s*true/ scan, and comment lines sitting between a comma and a
 * field hide that field from a key scan. Both produced false failures.
 */
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '')
}

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
    const body = stripComments(src.slice(start, j + 1))
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

  it('a busy emit never hardcodes running: true', () => {
    // Both busy branches sit in a `catch`. Liveness is not readiness: a
    // still-loading server has not opened /health yet, so the request throws
    // while the process is obviously alive, and the remote branch catches DNS
    // failures and refused connections too. Hardcoding `running: true` there
    // told the dashboard the model was usable mid-load — which only became
    // visible once the guard started reading `running`.
    //
    // Asserting the field merely EXISTS is not enough; that assertion stayed
    // green when the payload was flipped to `running: false`. The value has to
    // come from session state.
    const src = readFileSync(join(ROOT, 'src/main/sessions.ts'), 'utf8')
    const offenders: string[] = []
    for (const m of src.matchAll(/\.emit\(\s*'session:health'\s*,\s*\{/g)) {
      let depth = 0
      const start = m.index + m[0].length - 1
      let j = start
      for (; j < src.length; j++) {
        if (src[j] === '{') depth++
        else if (src[j] === '}') {
          depth--
          if (depth === 0) break
        }
      }
      const body = stripComments(src.slice(start, j + 1))
      if (/busy:\s*true/.test(body) && /running:\s*true/.test(body)) {
        offenders.push(`line ${src.slice(0, m.index).split('\n').length}`)
      }
    }
    expect(offenders).toEqual([])
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

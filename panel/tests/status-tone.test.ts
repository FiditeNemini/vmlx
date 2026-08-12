import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const source = readFileSync(
  resolve(__dirname, '../src/renderer/src/components/chat/ChatSettings.tsx'), 'utf-8')

/**
 * A screenshot showed "Status: Sleeping" in destructive red, directly above a
 * composer banner reading "Model sleeping — will auto-wake on your next
 * message". The rule was `status === 'running' ? primary : destructive`, so
 * loading, sleeping AND stopped all rendered in the colour reserved for
 * errors — the UI said something was broken when nothing was.
 */
describe('session status colour reflects meaning, not just "is it running"', () => {
  const body = (() => {
    const start = source.indexOf('function statusToneClass(')
    expect(start).toBeGreaterThan(-1)
    return source.slice(start, source.indexOf('\n}', start))
  })()

  it('reserves the destructive colour for actual errors', () => {
    expect(body).toMatch(/status === 'error'\)? return 'text-destructive'/)
    // the old catch-all is what painted three healthy states red
    expect(source).not.toContain("session.status === 'running' ? 'text-primary' : 'text-destructive'")
  })

  it('treats loading and sleeping as notable, not wrong', () => {
    expect(body).toMatch(/status === 'loading' \|\| status === 'standby'/)
    expect(body).toContain('text-amber-400')
  })

  it('running stays primary and anything else is muted', () => {
    expect(body).toMatch(/status === 'running'\)? return 'text-primary'/)
    expect(body).toContain("return 'text-muted-foreground'")
  })

  it('is what the status span actually uses', () => {
    expect(source).toContain('className={statusToneClass(session.status)}')
  })
})

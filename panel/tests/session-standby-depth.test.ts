import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const source = readFileSync(
  resolve(__dirname, '../src/renderer/src/components/sessions/SessionView.tsx'),
  'utf8',
)

describe('SessionView standby depth', () => {
  it('stores the emitted deep/soft depth instead of retaining a stale badge', () => {
    expect(source).toContain("standbyDepth: data.depth === 'deep' ? 'deep' : 'soft'")
    expect(source).toContain("session.standbyDepth === 'deep'")
    expect(source).not.toContain("(session as any).standbyDepth")
  })

  it('clears standby depth when the session leaves standby', () => {
    expect(source).toContain("status: 'loading', standbyDepth: null")
    expect(source).toMatch(/status: 'running',[\s\S]{0,80}standbyDepth: null/)
    expect(source).toMatch(/status: 'stopped',[\s\S]{0,120}standbyDepth: null/)
  })
})

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const read = (path: string): string => readFileSync(path, 'utf8')

describe('session PID lifecycle', () => {
  it('emits the spawned PID when a local engine becomes ready', () => {
    const source = read('src/main/sessions.ts')
    const readyStart = source.indexOf("this.emit('session:ready', {", source.indexOf('await this.waitForReady'))
    const readyBlock = source.slice(readyStart, readyStart + 240)

    expect(readyBlock).toContain('...(proc.pid ? { pid: proc.pid } : {})')
  })

  it('preserves a local PID when standby or monitored loading becomes ready', () => {
    const source = read('src/main/sessions.ts')

    expect(source.match(/\.\.\.\(session\.pid \? \{ pid: session\.pid \} : \{\}\)/g)).toHaveLength(2)
  })

  it('tracks ready PIDs in the shared chat context and clears them on stop', () => {
    const source = read('src/renderer/src/contexts/SessionsContext.tsx')

    expect(source).toContain('pid?: number')
    expect(source.match(/\.\.\.\(data\.pid \? \{ pid: data\.pid \} : \{\}\)/g)).toHaveLength(3)
    expect(source).toContain("status: 'stopped' as const, pid: undefined")
  })
})

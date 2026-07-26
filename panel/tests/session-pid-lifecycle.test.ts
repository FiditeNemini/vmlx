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

  it('uses the shared context PID as authoritative in chat settings', () => {
    const source = read('src/renderer/src/components/layout/ChatModeToolbar.tsx')

    expect(source).toContain('pid: contextSession ? contextSession.pid : sessionDetail.pid')
    expect(source).toContain('config: contextSession?.config ?? sessionDetail.config')
    expect(source).toContain('pid: displaySession.pid')
  })

  it('fences an intentional Stop from in-flight health-monitor failures', () => {
    const source = read('src/main/sessions.ts')

    expect(source).toContain('private intentionalStops = new Map<string, number>()')
    expect(source).toContain('(this.intentionalStops.get(sessionId) || 0) + 1')
    expect(source).toContain('if (remaining > 0) this.intentionalStops.set(sessionId, remaining)')
    expect(source).toContain('this.intentionalStops.delete(sessionId)')
    expect(source.match(/this\.intentionalStops\.has\(sessionId\)/g)?.length).toBeGreaterThanOrEqual(4)
    expect(source).toContain('Adoption probes are asynchronous')
  })
})

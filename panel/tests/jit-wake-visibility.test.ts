import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const sessions = readFileSync(new URL('../src/main/sessions.ts', import.meta.url), 'utf8')
const chat = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')

describe('message-triggered deep wake visibility', () => {
  it('publishes loading and starts the resident monitor before synchronous admin wake', () => {
    const wakeStart = sessions.indexOf('async wakeSession(sessionId: string)')
    const wakeEnd = sessions.indexOf('/** Check idle sessions', wakeStart)
    const body = sessions.slice(wakeStart, wakeEnd)
    const loading = body.indexOf("db.updateSession(sessionId, { status: 'loading', standbyDepth: null })")
    const monitor = body.indexOf('this.startLoadResidentMonitor(sessionId, session.pid, modelFileBytes)')
    const request = body.indexOf("fetch(`http://${host}:${session.port}/admin/wake`")

    expect(loading).toBeGreaterThan(0)
    expect(monitor).toBeGreaterThan(loading)
    expect(request).toBeGreaterThan(monitor)
    expect(body).toContain("progress: 1")
  })

  it('routes a sleeping in-app chat through SessionManager before health probing', () => {
    const wake = chat.indexOf('await sessionManager.wakeSession(resolvedSession.id)')
    const health = chat.indexOf('// Health check with retry')

    expect(wake).toBeGreaterThan(0)
    expect(health).toBeGreaterThan(wake)
    expect(chat).toContain('resolvedSession?.status === "standby"')
  })

  it('restores the original standby depth on wake failure', () => {
    expect(sessions).toContain("db.updateSession(sessionId, { status: 'standby', standbyDepth })")
    expect(sessions).toContain("this.emit('session:standby', { sessionId, depth: standbyDepth })")
  })

  it('does not let the health monitor erase loading while admin wake is in flight', () => {
    expect(sessions).toContain('private wakePending = new Set<string>()')
    expect(sessions).toContain('this.wakePending.add(sessionId)')
    expect(sessions).toContain('this.wakePending.has(session.id)')
    expect(sessions).toContain('this.wakePending.delete(sessionId)')
  })
})

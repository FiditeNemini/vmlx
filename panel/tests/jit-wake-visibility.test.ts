import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const sessions = readFileSync(new URL('../src/main/sessions.ts', import.meta.url), 'utf8')
const chat = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')

describe('message-triggered deep wake visibility', () => {
  it('publishes loading and starts the resident monitor before synchronous admin wake', () => {
    const wakeStart = sessions.indexOf('async wakeSession(sessionId: string)')
    const wakeEnd = sessions.indexOf('/** Check idle sessions', wakeStart)
    const body = sessions.slice(wakeStart, wakeEnd)
    // wakeSession delegates the presentation to the SHARED beginWakeProgress
    // helper (one implementation for panel-initiated, external-JIT, and
    // adopted-engine wakes) — but the ordering contract is unchanged:
    // loading is published and progress starts BEFORE the synchronous
    // admin/wake request.
    const loading = body.indexOf("db.updateSession(sessionId, { status: 'loading', standbyDepth: null })")
    const presentation = body.indexOf('this.beginWakeProgress(session,')
    const request = body.indexOf("fetch(`http://${host}:${session.port}/admin/wake`")

    expect(loading).toBeGreaterThan(0)
    expect(presentation).toBeGreaterThan(loading)
    expect(request).toBeGreaterThan(presentation)
    const helperStart = sessions.indexOf('private beginWakeProgress(session: Session')
    const helperBody = sessions.slice(helperStart, sessions.indexOf('/** Check a log line', helperStart))
    expect(helperBody).toContain('this.startLoadResidentMonitor(sessionId, session.pid, modelFileBytes)')
    // The wake start has no measured denominator — indeterminate, never an
    // invented number.
    expect(helperBody).toContain('indeterminate: true')
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

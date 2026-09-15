import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { SessionSsdPoolNotice } from '../src/renderer/src/components/sessions/SsdPoolNotice'

const session = {
  id: 'current', host: '127.0.0.1', port: 8019, pid: 321,
  status: 'running', type: 'local', config: '{}',
}

describe('SSD notice on active Chat and server surfaces', () => {
  it('binds the real local engine identity, not just a model name or port', () => {
    const element = SessionSsdPoolNotice({ session })!
    expect(element.props).toEqual({ sessionId: 'current', host: '127.0.0.1', port: 8019, pid: 321 })
    expect(element.key).toBe('current:321:127.0.0.1:8019')
    expect(SessionSsdPoolNotice({ session: { ...session, pid: 654 } })!.key).not.toBe(element.key)
  })
  it.each(['stopped', 'loading', 'standby', 'error'])('does not show an old pool on a %s engine', status => {
    expect(SessionSsdPoolNotice({ session: { ...session, status } })).toBeNull()
  })
  it('excludes remote, image, absent and unidentified engines', () => {
    expect(SessionSsdPoolNotice({})).toBeNull()
    expect(SessionSsdPoolNotice({ session: { ...session, type: 'remote' } })).toBeNull()
    expect(SessionSsdPoolNotice({ session: { ...session, config: '{"modelType":"image"}' } })).toBeNull()
    expect(SessionSsdPoolNotice({ session: { ...session, pid: undefined } })).toBeNull()
  })
  it('mounts the shared binding in both actual rendering paths', () => {
    const app = readFileSync(new URL('../src/renderer/src/App.tsx', import.meta.url), 'utf8')
    const server = readFileSync(new URL('../src/renderer/src/components/sessions/SessionView.tsx', import.meta.url), 'utf8')
    const chat = app.slice(app.indexOf('function ChatModeContent('), app.indexOf('function ChatEmptyState('))
    expect(app).toContain('ssdSession={activeSession}')
    expect(chat).toContain('<SessionSsdPoolNotice session={ssdSession} />')
    expect(server).toContain('<SessionSsdPoolNotice session={session} />')
  })
})

import { describe, expect, it } from 'vitest'
import { localApiUrl, summarizeTrayState, sessionShownByProcess } from '../src/shared/trayState'

describe('tray reports actual server state', () => {
  it('deduplicates memory and separates sleeping/loading from loaded models', () => {
    const processes = [{ port: 9001, status: 'running', gpuMemoryMB: 2048 }]
    const sessions = [
      { id: 'same', port: 9001, status: 'running' },
      { id: 'sleep', port: 9002, status: 'standby' },
      { id: 'load', port: 9003, status: 'loading' },
      { id: 'old', port: 9004, status: 'stopped' },
    ]
    expect(summarizeTrayState(processes, sessions, new Map([
      ['same', 2048], ['sleep', 50000], ['load', 1024], ['old', 40000],
    ]))).toEqual({ running: 1, loading: 1, standby: 1, memoryMB: 3072 })
  })
  it('does not let a dead process or another host hide a running session', () => {
    const session = { id: 'live', port: 9001, status: 'running', host: '127.0.0.1' }
    expect(sessionShownByProcess(session, [{ port: 9001, status: 'stopped' }])).toBe(false)
    expect(sessionShownByProcess({ ...session, host: 'other.local' }, [{ port: 9001, status: 'running' }])).toBe(false)
  })
  it.each([
    ['0.0.0.0', 'http://127.0.0.1:9123'], ['::', 'http://[::1]:9123'],
    ['::1', 'http://[::1]:9123'], ['[::1]', 'http://[::1]:9123'],
    ['192.168.1.3', 'http://192.168.1.3:9123'],
  ])('copies connectable root and OpenAI URLs for %s', (host, root) => {
    expect(localApiUrl(host, 9123)).toBe(root)
    expect(localApiUrl(host, 9123, true)).toBe(root + '/v1')
  })
})

import { describe, expect, it, vi } from 'vitest'
const { db } = vi.hoisted(() => ({ db: {
  getSession: vi.fn(() => ({ id: 'target', type: 'local', status: 'stopped',
    config: JSON.stringify({ additionalArgs: '--custom "unfinished' }) })),
  getSessions: vi.fn(() => []), getSetting: vi.fn(), updateSession: vi.fn(),
} }))
vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({
  app: { getAppPath: () => process.cwd(), getPath: () => '/tmp', isPackaged: false },
  BrowserWindow: { getFocusedWindow: () => ({}), getAllWindows: () => [] },
  dialog: { showMessageBox: vi.fn() },
  powerSaveBlocker: { isStarted: () => false, start: () => 1, stop: () => undefined },
}))
import { SessionManager } from '../src/main/sessions'
describe('malformed additional arguments lifecycle', () => {
  for (const method of ['startSession', 'restartSession', 'preflightSessionStart'] as const) {
    it(`${method} rejects before stopping or changing any session`, async () => {
      const manager = new SessionManager()
      const stop = vi.spyOn(manager, 'stopSession').mockResolvedValue()
      const detect = vi.spyOn(manager, 'detect').mockResolvedValue([])
      const emit = vi.spyOn(manager, 'emit')
      db.updateSession.mockClear()
      await expect(manager[method]('target')).rejects.toThrow('unclosed quote')
      expect(stop).not.toHaveBeenCalled()
      expect(detect).not.toHaveBeenCalled()
      expect(emit).not.toHaveBeenCalled()
      expect(db.updateSession).not.toHaveBeenCalled()
    })
  }
})

import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { db, state } = vi.hoisted(() => {
  const state = {
    sessions: [] as any[],
    settings: new Map<string, string>(),
  }
  const db = {
    getSessions: vi.fn(() => state.sessions),
    getSession: vi.fn((id: string) => state.sessions.find(session => session.id === id)),
    getSessionByModelPath: vi.fn((modelPath: string) =>
      state.sessions.find(session => session.modelPath === modelPath),
    ),
    getSetting: vi.fn((key: string) => state.settings.get(key)),
    createSession: vi.fn((session: any) => {
      state.sessions.push(session)
    }),
    updateSession: vi.fn((id: string, patch: Record<string, unknown>) => {
      const session = state.sessions.find(candidate => candidate.id === id)
      if (session) Object.assign(session, patch)
    }),
    deleteSession: vi.fn(),
  }
  return { db, state }
})

vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({
  app: {
    getAppPath: () => process.cwd(),
    getPath: () => '/tmp',
    isPackaged: false,
  },
  powerSaveBlocker: {
    isStarted: () => false,
    start: () => 1,
    stop: () => undefined,
  },
}))

import { SessionManager } from '../src/main/sessions'

const temporaryBundles: string[] = []

function modelBundle(): string {
  const directory = mkdtempSync(join(tmpdir(), 'vmlx-stop-cancels-start-'))
  temporaryBundles.push(directory)
  writeFileSync(join(directory, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }))
  return directory
}

afterEach(() => {
  for (const directory of temporaryBundles.splice(0)) {
    rmSync(directory, { recursive: true, force: true })
  }
  state.sessions.length = 0
  state.settings.clear()
  vi.restoreAllMocks()
})

describe('explicit Stop cancels a queued start', () => {
  /**
   * Save & Restart is renderer-orchestrated update -> stop -> start over
   * separate IPC calls. The session lock serializes those operations but
   * serialization is not cancellation: a user Stop that lands between the
   * restart's stop and its queued start used to execute FIRST and set the
   * session Stopped, after which the queued start spawned an engine the UI
   * no longer tracked. Observed live on the installed 1.6.44 smoke: UI and
   * DB said Stopped, port down, while a fresh ~98GB DSV4 engine (PID 98861)
   * stayed resident. Stop must defeat every queued start.
   */
  it('a start queued before an explicit stop aborts instead of spawning', async () => {
    const manager = new SessionManager()
    const sessionId = 'stop-beats-queued-start'
    state.sessions = [
      {
        id: sessionId,
        type: 'local',
        modelPath: modelBundle(),
        status: 'stopped',
        config: JSON.stringify({}),
      },
    ]

    const inner = vi
      .spyOn(manager as any, '_startSessionInner')
      .mockResolvedValue(undefined)

    // Hold the session lock so both operations queue in a known order.
    let releaseLock: () => void = () => {}
    const gate = new Promise<void>(resolve => {
      releaseLock = resolve
    })
    const lockHolder = (manager as any).withSessionLock(sessionId, () => gate)

    // 1. Start queues first (captures the pre-stop epoch)...
    const start = manager.startSession(sessionId)
    const startOutcome = start.then(
      () => ({ rejected: false as const }),
      (error: Error) => ({ rejected: true as const, message: error.message }),
    )
    // 2. ...then the explicit Stop arrives while the start is still queued.
    const stop = manager.stopSession(sessionId)

    releaseLock()
    await lockHolder
    await stop

    const outcome = await startOutcome
    expect(outcome.rejected).toBe(true)
    expect(outcome).toMatchObject({
      message: expect.stringContaining('stopped after this start was requested'),
    })
    expect(inner).not.toHaveBeenCalled()
    expect(state.sessions[0].status).toBe('stopped')
  })

  it('the renderer restart ordering (stop, then start) still starts', async () => {
    const manager = new SessionManager()
    const sessionId = 'restart-ordering-still-works'
    state.sessions = [
      {
        id: sessionId,
        type: 'local',
        modelPath: modelBundle(),
        status: 'stopped',
        config: JSON.stringify({}),
      },
    ]

    const inner = vi
      .spyOn(manager as any, '_startSessionInner')
      .mockResolvedValue(undefined)

    // Save & Restart awaits the stop BEFORE requesting the start, so the
    // start captures the post-stop epoch and must proceed.
    await manager.stopSession(sessionId)
    await expect(manager.startSession(sessionId)).resolves.toBeUndefined()
    expect(inner).toHaveBeenCalledTimes(1)
  })
})

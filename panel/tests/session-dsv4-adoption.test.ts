import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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

function modelBundle(modelType: string): string {
  const directory = mkdtempSync(join(tmpdir(), 'vmlx-adoption-policy-'))
  temporaryBundles.push(directory)
  writeFileSync(join(directory, 'config.json'), JSON.stringify({ model_type: modelType }))
  return directory
}

function malformedDsv4Bundle(): string {
  const directory = mkdtempSync(join(tmpdir(), 'deepseek-v4-adoption-policy-'))
  temporaryBundles.push(directory)
  writeFileSync(join(directory, 'config.json'), '{ malformed json')
  return directory
}

function localSession(
  id: string,
  modelPath: string,
  config: Record<string, unknown>,
  port = 8123,
): any {
  return {
    id,
    modelPath,
    modelName: id,
    host: '127.0.0.1',
    port,
    status: 'stopped',
    config: JSON.stringify(config),
    createdAt: 1,
    updatedAt: 1,
    type: 'local',
  }
}

function detected(modelPath: string, port = 8123, pid = 4242): any {
  return {
    pid,
    modelPath,
    modelName: 'live-model',
    port,
    healthy: true,
    standbyDepth: null,
  }
}

describe('DSV4 existing-engine adoption policy', () => {
  beforeEach(() => {
    state.sessions = []
    state.settings.clear()
    vi.clearAllMocks()
  })

  afterEach(() => {
    while (temporaryBundles.length > 0) {
      rmSync(temporaryBundles.pop()!, { recursive: true, force: true })
    }
  })

  it.each([
    {
      label: 'stale cache-on process',
      config: {
        dsv4PrefixCache: true,
        enablePrefixCache: true,
        usePagedCache: true,
        enableBlockDiskCache: true,
      },
    },
    {
      label: 'cache-off process with unattested executable provenance',
      config: {
        dsv4PrefixCache: false,
        enablePrefixCache: false,
        usePagedCache: false,
        enableBlockDiskCache: false,
      },
    },
  ])('terminates a same-path/same-port DSV4 $label before launch', async ({ config }) => {
    const modelPath = modelBundle('deepseek_v4')
    const session = localSession('dsv4', modelPath, config)
    state.sessions = [session]
    state.settings.set('gateway_single_model_mode', 'true')

    const manager = new SessionManager()
    const live = detected(modelPath, session.port)
    ;(manager as any).detect = vi.fn().mockResolvedValue([live])
    const terminate = vi.fn().mockResolvedValue(undefined)
    ;(manager as any).terminateDetectedLocalEngine = terminate

    await (manager as any).stopDetectedLocalEnginesForSingleModel(session.id)

    expect(terminate).toHaveBeenCalledTimes(1)
    expect(terminate).toHaveBeenCalledWith(live, state.sessions)
  })

  it('refuses direct and replacement DSV4 adoption even when path, port, and health match', async () => {
    const modelPath = modelBundle('deepseek_v4')
    const session = localSession('dsv4', modelPath, { enablePrefixCache: false })
    state.sessions = [session]

    const manager = new SessionManager()
    vi.clearAllMocks()
    const detectSpy = vi.fn().mockResolvedValue([detected(modelPath, session.port)])
    ;(manager as any).detect = detectSpy

    await expect((manager as any).adoptDetectedTargetProcessForStart(session.id)).resolves.toBe(false)
    await expect((manager as any).adoptHealthyReplacementForSession(session)).resolves.toBe(false)

    expect(detectSpy).not.toHaveBeenCalled()
    expect(db.updateSession).not.toHaveBeenCalled()
  })

  it('does not create or mark a startup DSV4 process as adopted', async () => {
    const modelPath = modelBundle('deepseek_v4')
    const manager = new SessionManager()
    ;(manager as any).detect = vi.fn().mockResolvedValue([detected(modelPath)])

    const adopted = await manager.detectAndAdoptAll()

    expect(adopted).toEqual([])
    expect(db.createSession).not.toHaveBeenCalled()
    expect(db.updateSession).not.toHaveBeenCalled()
    expect((manager as any).processes.size).toBe(0)
  })

  it('fails closed when a malformed DSV4 bundle makes full detection fall back', async () => {
    const modelPath = malformedDsv4Bundle()
    const manager = new SessionManager()
    ;(manager as any).detect = vi.fn().mockResolvedValue([detected(modelPath)])

    const adopted = await manager.detectAndAdoptAll()

    expect(adopted).toEqual([])
    expect(db.createSession).not.toHaveBeenCalled()
    expect((manager as any).processes.size).toBe(0)
  })

  it('terminates a hidden non-adoptable DSV4 engine before one-model pruning', async () => {
    const dsv4Path = modelBundle('deepseek_v4')
    const qwenPath = modelBundle('qwen3')
    const staleDsv4 = detected(dsv4Path, 8123, 4242)
    const qwen = detected(qwenPath, 8124, 4343)
    state.settings.set('gateway_single_model_mode', 'true')

    const manager = new SessionManager()
    ;(manager as any).detect = vi.fn().mockResolvedValue([staleDsv4, qwen])
    const terminate = vi.fn().mockResolvedValue(undefined)
    ;(manager as any).terminateDetectedLocalEngine = terminate

    const adopted = await manager.detectAndAdoptAll()

    expect(terminate).toHaveBeenCalledTimes(1)
    expect(terminate).toHaveBeenCalledWith(staleDsv4, state.sessions)
    expect(adopted).toHaveLength(1)
    expect(adopted[0].modelPath).toBe(qwenPath)
    expect((manager as any).processes.size).toBe(1)
    expect(Array.from((manager as any).processes.values())).toEqual([
      { process: null, adoptedPid: qwen.pid },
    ])
  })

  it('preserves same-target and startup adoption for a non-DSV4 family', async () => {
    const modelPath = modelBundle('qwen3')
    const session = localSession('qwen', modelPath, {})
    state.sessions = [session]

    const manager = new SessionManager()
    const live = detected(modelPath, session.port)
    ;(manager as any).detect = vi.fn().mockResolvedValue([live])
    const terminate = vi.fn().mockResolvedValue(undefined)
    ;(manager as any).terminateDetectedLocalEngine = terminate

    await (manager as any).stopDetectedLocalEnginesForSingleModel(session.id)
    expect(terminate).not.toHaveBeenCalled()
    await expect((manager as any).adoptDetectedTargetProcessForStart(session.id)).resolves.toBe(true)
    expect(db.updateSession).toHaveBeenCalledWith(
      session.id,
      expect.objectContaining({ status: 'running', pid: live.pid, port: live.port }),
    )

    state.sessions = []
    vi.clearAllMocks()
    const startupManager = new SessionManager()
    ;(startupManager as any).detect = vi.fn().mockResolvedValue([live])
    const adopted = await startupManager.detectAndAdoptAll()

    expect(adopted).toHaveLength(1)
    expect(db.createSession).toHaveBeenCalledTimes(1)
    expect((startupManager as any).processes.get(adopted[0].id)).toEqual({
      process: null,
      adoptedPid: live.pid,
    })
  })
})

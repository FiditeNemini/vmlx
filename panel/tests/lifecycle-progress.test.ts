import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  lifecycleDisplay,
  lifecyclePhaseLabel,
  parseLifecycleSnapshot,
} from '../src/main/lifecycleProgress'
import { localEngineReadyFromHealthBody } from '../src/main/engineReadiness'

const repo = join(__dirname, '..')
const read = (rel: string): string => readFileSync(join(repo, rel), 'utf8')

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
  const directory = mkdtempSync(join(tmpdir(), 'vmlx-lifecycle-progress-'))
  temporaryBundles.push(directory)
  writeFileSync(join(directory, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }))
  return directory
}

function localSession(id: string, status = 'loading'): any {
  return {
    id,
    type: 'local',
    modelPath: modelBundle(),
    status,
    config: JSON.stringify({}),
  }
}

function contractLine(snap: Record<string, unknown>): string {
  return `[10:00:00] INFO LOADPROGRESS ${JSON.stringify(snap)}\n`
}

afterEach(() => {
  for (const directory of temporaryBundles.splice(0)) {
    rmSync(directory, { recursive: true, force: true })
  }
  state.sessions.length = 0
  state.settings.clear()
  vi.restoreAllMocks()
})

describe('lifecycle phase mapping (engine owns the oracle)', () => {
  it('phases without a measured denominator are indeterminate — no invented numbers', () => {
    const base = { completed: 0, total: 0, model_loaded: false, ready: false, generation: 1 }
    expect(lifecycleDisplay({ ...base, phase: 'starting' })).toEqual({ indeterminate: true })
    expect(lifecycleDisplay({ ...base, phase: 'loading_weights' })).toEqual({ indeterminate: true })
    expect(lifecycleDisplay({ ...base, phase: 'initializing_engine' })).toEqual({ indeterminate: true })
    expect(lifecycleDisplay({ ...base, phase: 'restoring_acceleration' })).toEqual({ indeterminate: true })
    expect(lifecycleDisplay({ ...base, phase: 'ready', ready: true })).toEqual({ indeterminate: false, percent: 100 })
  })

  it('determinate percentages exist only for measured shard units, 100 only for ready', () => {
    const shard = (completed: number, total: number) =>
      lifecycleDisplay({ phase: 'loading_weights', completed, total, model_loaded: false, ready: false, generation: 1 })
    expect(shard(0, 17)).toEqual({ indeterminate: false, percent: 0 })
    expect(shard(9, 17)).toEqual({ indeterminate: false, percent: 53 })
    // All shards loaded is still not inference-ready: 100 is reserved.
    expect(shard(17, 17)).toEqual({ indeterminate: false, percent: 99 })
    expect(shard(30, 17)).toEqual({ indeterminate: false, percent: 99 })
  })

  it('labels shard progress with translatable params', () => {
    const label = lifecyclePhaseLabel({ phase: 'loading_weights', completed: 3, total: 17, model_loaded: false, ready: false, generation: 1 })
    expect(label.labelKey).toBe('main.loadProgress.loadingWeightShards')
    expect(label.labelParams).toEqual({ completed: 3, total: 17 })
  })

  it('parses only well-formed snapshots', () => {
    expect(parseLifecycleSnapshot('{"phase":"starting","generation":1}')).toMatchObject({ phase: 'starting', generation: 1 })
    expect(parseLifecycleSnapshot('not json')).toBeNull()
    expect(parseLifecycleSnapshot('{"no_phase":true}')).toBeNull()
  })

  it('every emitted labelKey resolves in every locale catalog', () => {
    for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
      const catalog = JSON.parse(read(`src/renderer/src/i18n/locales/${locale}.json`))
      for (const key of ['loadingWeightShards', 'initializingEngine', 'restoringAcceleration', 'preparing']) {
        expect(catalog.main.loadProgress[key], `${locale}:${key}`).toBeTruthy()
      }
    }
  })
})

describe('state machine: contract events drive the panel', () => {
  it('cold load: engine phases advance the bar to engine-owned 100', () => {
    const manager = new SessionManager()
    const sessionId = 'cold-load'
    state.sessions = [localSession(sessionId)]
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))

    manager.pushLog(sessionId, contractLine({ phase: 'starting', completed: 0, total: 0, model_loaded: false, ready: false, generation: 1 }))
    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 5, total: 10, model_loaded: false, ready: false, generation: 1 }))
    manager.pushLog(sessionId, contractLine({ phase: 'restoring_acceleration', completed: 10, total: 10, model_loaded: true, ready: false, generation: 1 }))
    manager.pushLog(sessionId, contractLine({ phase: 'ready', completed: 10, total: 10, model_loaded: true, ready: true, generation: 1 }))

    // starting = indeterminate; measured shards = determinate 50; the
    // acceleration phase has no denominator (indeterminate, keeps 50); the
    // engine's authoritative ready is the only 100.
    expect(events.map(e => [e.progress, e.indeterminate])).toEqual([
      [0, true],
      [50, false],
      [50, true],
      [100, false],
    ])
    expect(events.every(e => e.progressGeneration === 1)).toBe(true)
  })

  it('stale generation events are discarded after a new attempt begins', () => {
    const manager = new SessionManager()
    const sessionId = 'stale-generation'
    state.sessions = [localSession(sessionId)]
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))

    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 8, total: 10, model_loaded: false, ready: false, generation: 2 }))
    // A late line from the PREVIOUS attempt must not repaint anything.
    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 1, total: 10, model_loaded: false, ready: false, generation: 1 }))
    expect(events).toHaveLength(1)
    expect(events[0].progress).toBe(80)
  })

  it('a NEW generation resets the monotonic display guard', () => {
    const manager = new SessionManager()
    const sessionId = 'new-generation-resets'
    state.sessions = [localSession(sessionId)]
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))

    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 9, total: 10, model_loaded: false, ready: false, generation: 1 }))
    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 1, total: 10, model_loaded: false, ready: false, generation: 2 }))
    expect(events.map(e => e.progress)).toEqual([90, 10])
  })

  it('legacy log heuristics stay off once the engine speaks the contract', () => {
    const manager = new SessionManager()
    const sessionId = 'legacy-suppressed'
    state.sessions = [localSession(sessionId)]
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))

    manager.pushLog(sessionId, contractLine({ phase: 'starting', completed: 0, total: 0, model_loaded: false, ready: false, generation: 1 }))
    manager.pushLog(sessionId, 'INFO: Uvicorn running on http://127.0.0.1:8000\n')
    expect(events).toHaveLength(1)
    expect(events[0].labelKey).toBe('main.loadProgress.initializing')
  })

  it('multi-session isolation: events land only on their own session', () => {
    const manager = new SessionManager()
    state.sessions = [localSession('session-a'), localSession('session-b')]
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))

    manager.pushLog('session-a', contractLine({ phase: 'loading_weights', completed: 2, total: 4, model_loaded: false, ready: false, generation: 1 }))
    manager.pushLog('session-b', contractLine({ phase: 'starting', completed: 0, total: 0, model_loaded: false, ready: false, generation: 1 }))

    const snapshot = manager.getLoadProgressSnapshot()
    expect((snapshot['session-a'] as any).progress).toBe(50)
    expect((snapshot['session-b'] as any).indeterminate).toBe(true)
    expect(events.filter(e => e.sessionId === 'session-a')).toHaveLength(1)
  })

  it('Stop clears the hydration snapshot and generation state', async () => {
    const manager = new SessionManager()
    const sessionId = 'stop-clears'
    state.sessions = [localSession(sessionId)]
    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 1, total: 4, model_loaded: false, ready: false, generation: 3 }))
    expect(manager.getLoadProgressSnapshot()[sessionId]).toBeTruthy()

    await manager.stopSession(sessionId)
    expect(manager.getLoadProgressSnapshot()[sessionId]).toBeUndefined()
    // The replacement engine restarts at generation 1 — it must not be
    // discarded against the old high-water mark.
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))
    state.sessions[0].status = 'loading'
    manager.pushLog(sessionId, contractLine({ phase: 'starting', completed: 0, total: 0, model_loaded: false, ready: false, generation: 1 }))
    expect(events).toHaveLength(1)
  })

  it('navigation hydration: the snapshot serves pages opened mid-load', () => {
    const manager = new SessionManager()
    const sessionId = 'hydration'
    state.sessions = [localSession(sessionId)]
    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 3, total: 4, model_loaded: false, ready: false, generation: 1 }))
    manager.pushLog(sessionId, contractLine({ phase: 'initializing_engine', completed: 4, total: 4, model_loaded: true, ready: false, generation: 1 }))
    const entry = manager.getLoadProgressSnapshot()[sessionId] as any
    expect(entry.progress).toBe(75)
    expect(entry.indeterminate).toBe(true)
    expect(entry.progressGeneration).toBe(1)
    expect(entry.labelKey).toBe('main.loadProgress.initializingEngine')
  })

  it('local /health readiness requires the BODY to say so, never HTTP 200 alone', () => {
    expect(localEngineReadyFromHealthBody({ status: 'healthy', model_loaded: true })).toBe(true)
    expect(localEngineReadyFromHealthBody({ status: 'healthy', model_loaded: true, load_progress: { ready: true } })).toBe(true)
    // Every state /health happily serves with HTTP 200:
    expect(localEngineReadyFromHealthBody({ status: 'no_model' })).toBe(false)
    expect(localEngineReadyFromHealthBody({ status: 'standby_deep', model_loaded: false })).toBe(false)
    expect(localEngineReadyFromHealthBody({ status: 'healthy', model_loaded: false })).toBe(false)
    expect(localEngineReadyFromHealthBody({ status: 'healthy', model_loaded: true, wake_in_progress: true })).toBe(false)
    expect(localEngineReadyFromHealthBody({ status: 'healthy', model_loaded: true, load_progress: { ready: false } })).toBe(false)
    expect(localEngineReadyFromHealthBody(null)).toBe(false)
    expect(localEngineReadyFromHealthBody('ok')).toBe(false)
  })

  it('the queued wait aborts on PID replacement (process swap mid-wait)', async () => {
    const manager = new SessionManager()
    const sessionId = 'pid-replacement'
    state.sessions = [{ ...localSession(sessionId, 'loading'), pid: 1111 }]

    const wait = manager.waitForSessionLifecycleSettled(sessionId, { timeoutMs: 5000 })
    setTimeout(() => { state.sessions[0].pid = 2222 }, 80)
    await expect(wait).resolves.toBe('replaced')
  })

  it('the queued wait returns standby/error transitions instead of hanging', async () => {
    const manager = new SessionManager()
    const sessionId = 'standby-transition'
    state.sessions = [localSession(sessionId, 'loading')]
    const wait = manager.waitForSessionLifecycleSettled(sessionId, { timeoutMs: 5000 })
    setTimeout(() => { state.sessions[0].status = 'standby' }, 80)
    await expect(wait).resolves.toBe('standby')
  })

  it('running alone is not enough for a contract engine — engine ready is required', async () => {
    const manager = new SessionManager()
    const sessionId = 'contract-ready-required'
    state.sessions = [localSession(sessionId, 'loading')]
    // The engine spoke the contract but has not delivered ready yet.
    manager.pushLog(sessionId, contractLine({ phase: 'loading_weights', completed: 1, total: 4, model_loaded: false, ready: false, generation: 1 }))
    state.sessions[0].status = 'running'

    const wait = manager.waitForSessionLifecycleSettled(sessionId, { timeoutMs: 4000 })
    let settledEarly = false
    wait.then(() => { settledEarly = true })
    await new Promise(resolve => setTimeout(resolve, 900))
    expect(settledEarly).toBe(false)
    // The engine's authoritative ready arrives — the wait resolves.
    manager.pushLog(sessionId, contractLine({ phase: 'ready', completed: 4, total: 4, model_loaded: true, ready: true, generation: 1 }))
    await expect(wait).resolves.toBe('running')
  })

  it('waitForSessionLifecycleSettled resolves on ready and aborts on Stop', async () => {
    const manager = new SessionManager()
    const sessionId = 'queued-message'
    state.sessions = [localSession(sessionId, 'loading')]

    const settled = manager.waitForSessionLifecycleSettled(sessionId, { timeoutMs: 5000 })
    setTimeout(() => { state.sessions[0].status = 'running' }, 120)
    await expect(settled).resolves.toBe('running')

    state.sessions[0].status = 'loading'
    const controller = new AbortController()
    const aborted = manager.waitForSessionLifecycleSettled(sessionId, { timeoutMs: 5000, signal: controller.signal })
    setTimeout(() => controller.abort(), 60)
    await expect(aborted).rejects.toThrow('Request canceled')
  })

  it('wake presentation still publishes the waking event', () => {
    const manager = new SessionManager()
    const sessionId = 'wake-presentation'
    const session = localSession(sessionId)
    state.sessions = [session]
    const events: any[] = []
    manager.on('session:loadProgress', (data: any) => events.push(data))
    ;(manager as any).beginWakeProgress(session, '[Wake] test wake')
    ;(manager as any).stopWakeHealthPoller(sessionId)
    expect(events[0]).toMatchObject({ sessionId, labelKey: 'main.loadProgress.wakingFromSleep', indeterminate: true })
  })
})

describe('contract wiring pins', () => {
  it('the engine owns the snapshot and /health exposes it', () => {
    const server = readFileSync(join(repo, '..', 'vmlx_engine', 'server.py'), 'utf8')
    expect(server).toContain('result["load_progress"] = _lifecycle_progress.snapshot()')
    expect(server.match(/result\["load_progress"\]/g)!.length).toBeGreaterThanOrEqual(2)
    expect(server).toContain('_lifecycle_progress.begin_attempt(_lifecycle_progress.PHASE_STARTING)')
    const module = readFileSync(join(repo, '..', 'vmlx_engine', 'load_progress.py'), 'utf8')
    expect(module).toContain('LOADPROGRESS')
    expect(module).toContain('"generation"')
  })

  it('RSS is a diagnostic, never the percentage oracle', () => {
    const source = read('src/main/sessions.ts')
    const tickStart = source.indexOf('// RSS is a DIAGNOSTIC readout, never the percentage oracle')
    expect(tickStart).toBeGreaterThan(-1)
    const tick = source.slice(tickStart, source.indexOf('tick()', tickStart))
    expect(tick).not.toContain('loadProgressState.set')
    expect(tick).not.toMatch(/progress:\s*residentProgress/)
  })

  it('SessionView health handler never promotes running from a non-running health event', () => {
    const view = read('src/renderer/src/components/sessions/SessionView.tsx')
    expect(view).toContain("data.running === true ? { status: 'running' as const")
    expect(view).not.toMatch(/handleHealth[\s\S]{0,400}\n\s+status: 'running',/)
  })

  it('a chat message sent mid-load queues until the lifecycle settles', () => {
    const chat = read('src/main/ipc/chat.ts')
    expect(chat).toContain('waitForSessionLifecycleSettled')
    expect(chat).toContain('signal: abortController.signal')
  })

  it('renderer hydrates progress on mount and guards stale generations', () => {
    const context = read('src/renderer/src/contexts/SessionsContext.tsx')
    expect(context).toContain('window.api.sessions.getLoadProgress?.()')
    expect(context).toContain('data.progressGeneration < existing.progressGeneration')
    const preload = read('src/preload/index.ts')
    expect(preload).toContain("ipcRenderer.invoke('sessions:getLoadProgress')")
  })

  it('every load surface renders the target-session progress', () => {
    const card = read('src/renderer/src/components/sessions/SessionCard.tsx')
    const view = read('src/renderer/src/components/sessions/SessionView.tsx')
    const chat = read('src/renderer/src/components/chat/ChatInterface.tsx')
    const create = read('src/renderer/src/components/sessions/CreateSession.tsx')
    expect(card).toContain('session.status === "running" && progress && progress.progress < 100')
    expect(view).toContain("session.status === 'loading' || session.status === 'running'")
    expect(chat).toContain('useSessionsContext()')
    expect(chat).toContain('sessionLoadProgress.progress < 100')
    expect(create).toContain('loadProgress.get(launchSessionId)')
  })
})

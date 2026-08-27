import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
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

// Track the shipping value instead of hardcoding it: every version bump
// otherwise breaks 28 unrelated assertions that only care that the session
// was stamped current.
const CURRENT_CACHE_DEFAULTS_VERSION = Number(
  /const CACHE_STACK_STARTUP_DEFAULTS_VERSION = (\d+)/.exec(
    readFileSync('src/main/sessions.ts', 'utf8'),
  )?.[1],
)

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

function untouchedV12Dsv4CacheConfig(): Record<string, unknown> {
  return {
    cacheStackStartupDefaultsVersion: 12,
    continuousBatching: true,
    maxNumSeqs: 1,
    prefillBatchSize: 512,
    prefillStepSize: 2048,
    completionBatchSize: 512,
    dsv4PrefixCache: true,
    enablePrefixCache: true,
    prefixCacheSize: 100,
    prefixCacheMaxBytes: 0,
    cacheMemoryMb: 0,
    cacheMemoryPercent: 15,
    cacheTtlMinutes: 0,
    noMemoryAwareCache: false,
    usePagedCache: false,
    enableDiskCache: false,
    diskCacheMaxGb: 10,
    diskCacheDir: '',
    enableBlockDiskCache: true,
    blockDiskCacheMaxGb: 10,
    blockDiskCacheDir: '',
    pagedCacheBlockSize: 256,
    maxCacheBlocks: 4097,
    kvCacheQuantization: 'auto',
    kvCacheGroupSize: 64,
  }
}

function untouchedV11Dsv4CacheConfig(): Record<string, unknown> {
  return {
    ...untouchedV12Dsv4CacheConfig(),
    cacheStackStartupDefaultsVersion: 11,
    dsv4PrefixCache: false,
    enablePrefixCache: false,
    enableBlockDiskCache: false,
    maxCacheBlocks: 1000,
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

  // v17 (2026-08-21) makes SSD-only the default for every family that has an
  // SSD-only lane, DSV4 included -- proven live on DeepSeek-V4-Flash-0731, where
  // the block disk-only backend served 3325/3328 cached tokens with correct
  // answers across changed prefixes, effort changes, tool toggles and malformed
  // tool calls. The legacy tuple rewrites below still run and still own the
  // native block geometry; the paged-RAM tier is what changed.
  it('migrates only the exact v11 DSV4 fail-closed cache tuple to native hot/warm/cold defaults', () => {
    const modelPath = modelBundle('deepseek_v4')
    state.sessions = [localSession('dsv4-v11', modelPath, untouchedV11Dsv4CacheConfig())]

    new SessionManager()

    const migrated = JSON.parse(state.sessions[0].config)
    expect(migrated.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
    expect(migrated.dsv4PrefixCache).toBe(true)
    expect(migrated.enablePrefixCache).toBe(true)
    // v17: SSD-first. The native composite index still owns block geometry.
    expect(migrated.usePagedCache).toBe(false)
    expect(migrated.enableDiskCache).toBe(false)
    expect(migrated.enableBlockDiskCache).toBe(true)
    expect(migrated.pagedCacheBlockSize).toBe(256)
    expect(migrated.maxCacheBlocks).toBe(4097)
    expect(migrated.kvCacheQuantization).toBe('auto')
  })

  it.each([
    { label: 'prefix already enabled', patch: { enablePrefixCache: true } },
    { label: 'paged RAM explicitly enabled', patch: { usePagedCache: true } },
    { label: 'SSD block cache already enabled', patch: { enableBlockDiskCache: true } },
    { label: 'cache block capacity customized', patch: { maxCacheBlocks: 2048 } },
    { label: 'RAM percentage customized', patch: { cacheMemoryPercent: 37 } },
    { label: 'block-L2 directory customized', patch: { blockDiskCacheDir: '/tmp/custom-block-cache' } },
    { label: 'prefill batch customized', patch: { prefillBatchSize: 256 } },
  ])('preserves a near-miss DSV4 v11 cache tuple: $label', ({ patch }) => {
    const modelPath = modelBundle('deepseek_v4')
    const original = {
      ...untouchedV11Dsv4CacheConfig(),
      ...patch,
    }
    state.sessions = [localSession('dsv4-near-miss', modelPath, original)]

    new SessionManager()

    const preserved = JSON.parse(state.sessions[0].config)
    // What "preserved" protects is that the legacy v11 REWRITE did not fire:
    // a near-miss keeps its own prefix policy and block capacity rather than
    // being rebuilt into the native hot/warm/cold tuple.
    expect(preserved.enablePrefixCache).toBe(original.enablePrefixCache)
    expect(preserved.maxCacheBlocks).toBe(original.maxCacheBlocks)
    expect(preserved.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
    // v17 is a separate, universal default change: the in-RAM mirror ships off
    // for updated users too, otherwise "it ships off" is true only of fresh
    // installs. It runs as a POST-pass precisely so it cannot rewrite a
    // near-miss tuple into an exact match and re-fire the migration above.
    expect(preserved.usePagedCache).toBe(false)
    expect(preserved.enableBlockDiskCache).toBe(true)
  })

  it('leaves a session already stamped at the current version completely alone', () => {
    // The version stamp is what makes every migration run exactly once. A
    // session already at the current version must not be touched by ANY pass,
    // including the v17 SSD-first post-pass -- otherwise a user who turned the
    // RAM cache back on after upgrading would silently lose that choice on the
    // next launch.
    const modelPath = modelBundle('deepseek_v4')
    const original = {
      ...untouchedV11Dsv4CacheConfig(),
      cacheStackStartupDefaultsVersion: CURRENT_CACHE_DEFAULTS_VERSION,
      usePagedCache: true,
    }
    state.sessions = [localSession('dsv4-already-current', modelPath, original)]

    new SessionManager()

    const preserved = JSON.parse(state.sessions[0].config)
    expect(preserved.usePagedCache).toBe(true)
    expect(preserved.enableBlockDiskCache).toBe(original.enableBlockDiskCache)
    expect(preserved.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
  })

  it('keeps the exact v12 DSV4 SSD-only default on SSD, with the native block index', () => {
    const modelPath = modelBundle('deepseek_v4')
    state.sessions = [localSession('dsv4-v12', modelPath, untouchedV12Dsv4CacheConfig())]

    new SessionManager()

    const migrated = JSON.parse(state.sessions[0].config)
    expect(migrated.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
    expect(migrated.dsv4PrefixCache).toBe(true)
    expect(migrated.enablePrefixCache).toBe(true)
    // v17: SSD-first. The native composite index still owns block geometry.
    expect(migrated.usePagedCache).toBe(false)
    expect(migrated.enableDiskCache).toBe(false)
    expect(migrated.enableBlockDiskCache).toBe(true)
    expect(migrated.pagedCacheBlockSize).toBe(256)
    expect(migrated.maxCacheBlocks).toBe(4097)
    expect(migrated.kvCacheQuantization).toBe('auto')
  })

  it.each([
    ['RAM percentage', { cacheMemoryPercent: 37 }],
    ['RAM MiB ceiling', { cacheMemoryMb: 4096 }],
    ['RAM TTL', { cacheTtlMinutes: 30 }],
    ['memory-awareness override', { noMemoryAwareCache: true }],
    ['prefix entry capacity', { prefixCacheSize: 250 }],
    ['prefix byte capacity', { prefixCacheMaxBytes: 8 * 1024 ** 3 }],
    ['disk cache capacity', { diskCacheMaxGb: 25 }],
    ['legacy disk directory', { diskCacheDir: '/tmp/custom-prompt-cache' }],
    ['block-L2 capacity', { blockDiskCacheMaxGb: 25 }],
    ['block-L2 directory', { blockDiskCacheDir: '/tmp/custom-block-cache' }],
    ['cache block capacity', { maxCacheBlocks: 2048 }],
    ['cache codec group size', { kvCacheGroupSize: 128 }],
    ['prefill batch size', { prefillBatchSize: 256 }],
  ])('preserves a customized v12 DSV4 SSD-only session: %s', (_label, patch) => {
    const modelPath = modelBundle('deepseek_v4')
    state.sessions = [localSession('dsv4-v12-custom', modelPath, {
      ...untouchedV12Dsv4CacheConfig(),
      ...patch,
    })]

    new SessionManager()

    const preserved = JSON.parse(state.sessions[0].config)
    expect(preserved.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
    expect(preserved.usePagedCache).toBe(false)
    for (const [key, value] of Object.entries(patch)) {
      expect(preserved[key]).toBe(value)
    }
  })

  it('does not reopen legacy generic migrations when stamping a v12 session to v13', () => {
    const modelPath = modelBundle('qwen3')
    state.sessions = [localSession('generic-v12-custom', modelPath, {
      cacheStackStartupDefaultsVersion: 12,
      continuousBatching: true,
      enablePrefixCache: true,
      maxNumSeqs: 1,
      prefillBatchSize: 512,
      prefillStepSize: 2048,
      completionBatchSize: 512,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: true,
      kvCacheQuantization: 'auto',
      cacheMemoryPercent: 37,
    })]

    new SessionManager()

    const preserved = JSON.parse(state.sessions[0].config)
    expect(preserved.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
    // The customised value survives: no older generic predicate re-fired and
    // rebuilt this session from a stale default tuple.
    expect(preserved.cacheMemoryPercent).toBe(37)
    // v17 still applies -- an updated user gets the SSD-first default too.
    expect(preserved.usePagedCache).toBe(false)
    expect(preserved.enableBlockDiskCache).toBe(true)
  })

  it('retries the v12 DSV4 migration after an unavailable bundle returns', () => {
    const modelPath = modelBundle('deepseek_v4')
    rmSync(modelPath, { recursive: true, force: true })
    state.sessions = [localSession(
      'dsv4-v12-unmounted',
      modelPath,
      untouchedV12Dsv4CacheConfig(),
    )]

    new SessionManager()

    const unavailable = JSON.parse(state.sessions[0].config)
    expect(unavailable.cacheStackStartupDefaultsVersion).toBe(12)
    expect(unavailable.usePagedCache).toBe(false)

    mkdirSync(modelPath, { recursive: true })
    writeFileSync(join(modelPath, 'config.json'), JSON.stringify({ model_type: 'deepseek_v4' }))
    new SessionManager()

    const retried = JSON.parse(state.sessions[0].config)
    expect(retried.cacheStackStartupDefaultsVersion).toBe(CURRENT_CACHE_DEFAULTS_VERSION)
    // The point of the retry is that the migration RUNS once the bundle is back
    // (version advances past 12); under v17 it lands on the SSD-first default.
    expect(retried.usePagedCache).toBe(false)
    expect(retried.enableBlockDiskCache).toBe(true)
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

  it('deep-sleeps at the hard deadline when it is earlier than light sleep', async () => {
    const modelPath = modelBundle('muse_glimmer')
    const session = localSession('muse-sleep-order', modelPath, {
      cacheStackStartupDefaultsVersion: CURRENT_CACHE_DEFAULTS_VERSION,
      autoSleepEnabled: true,
      idleTimeoutSoftMin: 10,
      idleTimeoutHardMin: 1,
    })
    session.status = 'running'
    session.lastRequestAt = Date.now() - 61_000
    session.lastStartedAt = session.lastRequestAt
    state.sessions = [session]

    const manager = new SessionManager()
    const deepSleep = vi.spyOn(manager, 'deepSleep').mockResolvedValue({ success: true })
    const softSleep = vi.spyOn(manager, 'softSleep').mockResolvedValue({ success: true })

    await (manager as any).checkIdleSessions()

    expect(deepSleep).toHaveBeenCalledOnce()
    expect(deepSleep).toHaveBeenCalledWith(session.id)
    expect(softSleep).not.toHaveBeenCalled()
  })
})

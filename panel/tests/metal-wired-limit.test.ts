import { describe, expect, it } from 'vitest'
import { appendMetalWiredLimitGuidance, classifyLargeModelMemoryPreflight, isMeasuredMetalPressure, measuredWiredLimitCommand, readMeasuredMetalMemory, metalWiredLimitHelpText, type MeasuredMetalMemory } from '../src/shared/metalWiredLimit'
import { MemoryWarningStore } from '../src/main/memory-warning-store'
const GiB = 1024 ** 3
const now = 100_000
const reading = (extra: Partial<MeasuredMetalMemory> = {}): MeasuredMetalMemory => ({
  version: 1, available: true, pid: 42, measured_at_ms: now,
  source: 'mlx_active_working_set', reason: 'measurement',
  active_bytes: 96 * GiB, device_limit_bytes: 96 * GiB, limit_bytes: 96 * GiB,
  physical_bytes: 128 * GiB, ...extra,
})
describe('measured Metal warning', () => {
  it('stays quiet below the limit, even at 99.9%; no file-size estimate', () => {
    expect(isMeasuredMetalPressure(reading({ active_bytes: 95.999 * GiB }))).toBe(false)
    expect(isMeasuredMetalPressure(reading())).toBe(true)
    expect(isMeasuredMetalPressure(reading({ active_bytes: 99 * GiB }))).toBe(true)
  })
  it('reports an actual guard rejection but not a predicted envelope', () => {
    expect(isMeasuredMetalPressure(reading({ reason: 'guard_rejection', threshold_pct: 99, active_bytes: 95.1 * GiB }))).toBe(true)
    expect(isMeasuredMetalPressure(reading({ reason: 'guard_rejection', threshold_pct: 99, active_bytes: 90 * GiB }))).toBe(false)
  })
  it.each([
    { available: false }, { active_bytes: NaN }, { active_bytes: -1 }, { limit_bytes: 0 },
    { limit_bytes: Infinity }, { pid: 43 }, { measured_at_ms: now - 30_001 },
    { measured_at_ms: now + 1001 }, { source: 'file_size' },
    { reason: 'guard_rejection', threshold_pct: NaN }, { reason: 'guard_rejection', threshold_pct: 101 },
  ])('ignores invalid/stale/other-process measurement %j', (extra) => {
    expect(readMeasuredMetalMemory({ ...reading(), ...extra }, 42, now)).toBeNull()
  })
  it('accepts real zero active memory but never warns', () => {
    expect(isMeasuredMetalPressure(readMeasuredMetalMemory(reading({ active_bytes: 0 }), 42, now)!)).toBe(false)
  })
  it.each([16, 32, 64, 128, 256])('bounds optional advice for %i GiB using MiB and rounding DOWN', (physical) => {
    const v = reading({ active_bytes: physical * GiB * 0.7, limit_bytes: physical * GiB * 0.7, device_limit_bytes: physical * GiB * 0.7, physical_bytes: physical * GiB })
    const cmd = measuredWiredLimitCommand(v)
    if (cmd) {
      expect(cmd).toMatch(/^sudo sysctl iogpu\.wired_limit_mb=\d+$/)
      const bytes = Number(cmd.split('=')[1]) * 1024 ** 2
      expect(bytes).toBeGreaterThan(v.limit_bytes)
      expect(bytes).toBeLessThanOrEqual(v.physical_bytes! - Math.max(8 * GiB, v.physical_bytes! * 0.1))
    } else expect(physical).toBe(16)
  })
  it('offers no command with unknown hardware, no safe room, or an engine-only override', () => {
    expect(measuredWiredLimitCommand(reading({ physical_bytes: null }))).toBeNull()
    expect(measuredWiredLimitCommand(reading({ physical_bytes: 100 * GiB }))).toBeNull()
    expect(measuredWiredLimitCommand(reading({ device_limit_bytes: 110 * GiB }))).toBeNull()
  })
})
describe('notice persistence and deduplication', () => {
  const context = { sessionId: 's1', modelPath: '/models/one', modelName: 'one', pid: 42 }
  function state() {
    const data = new Map<string, string>()
    const settings = { getSetting: (k: string) => data.get(k), setSetting: (k: string, v: string) => { data.set(k, v) } }
    return { settings, data, store: new MemoryWarningStore(settings) }
  }
  it('does not repeat across samples, sessions, or PIDs for the same model', () => {
    const { store } = state()
    expect(store.observe(context, reading(), now)).toBe(true)
    expect(store.observe(context, reading(), now)).toBe(false)
    expect(store.observe({ ...context, sessionId: 's2', pid: 44 }, reading({ pid: 44 }), now)).toBe(false)
    expect(store.list()).toHaveLength(1)
    store.dismiss(store.list()[0].id, false)
    expect(store.list()).toHaveLength(0)
    expect(store.observe(context, reading(), now)).toBe(false)
  })
  it('persists per-model opt-out across app restarts without hiding another model', () => {
    const { store, settings, data } = state()
    store.observe(context, reading(), now)
    store.dismiss(store.list()[0].id, true)
    expect(data.size).toBe(1)
    const restarted = new MemoryWarningStore(settings)
    expect(restarted.observe(context, reading(), now)).toBe(false)
    expect(restarted.observe({ ...context, modelPath: '/models/two' }, reading(), now)).toBe(true)
  })
  it('normalizes equivalent paths and ignores unknown dismissal IDs', () => {
    const { store, data } = state()
    store.observe(context, reading(), now)
    expect(store.observe({ ...context, modelPath: '/models/a/../one' }, reading(), now)).toBe(false)
    store.dismiss('not-a-notice', true)
    expect(data.size).toBe(0)
    expect(store.list()).toHaveLength(1)
  })
  it('failed persistence leaves the notice actionable', () => {
    const store = new MemoryWarningStore({ getSetting: () => null, setSetting: () => { throw new Error('disk') } })
    store.observe(context, reading(), now)
    expect(() => store.dismiss(store.list()[0].id, true)).toThrow('disk')
    expect(store.list()).toHaveLength(1)
  })
})
describe('advisory memory preflight, never an admission veto', () => {
  it('does not give a universal sysctl value or assert OOM causality', () => {
    const text = appendMetalWiredLimitGuidance('Metal OOM: Insufficient Memory')
    expect(text).toContain(metalWiredLimitHelpText)
    expect(text).not.toContain('120000')
    expect(text).toContain('does not establish')
    expect(appendMetalWiredLimitGuidance(text)).toBe(text)
  })
  it('handles SIGKILL without claiming it proves a wired limit', () => {
    expect(appendMetalWiredLimitGuidance('Process was killed (SIGKILL)')).toContain(metalWiredLimitHelpText)
    expect(appendMetalWiredLimitGuidance('ImportError: mlx missing')).toBe('ImportError: mlx missing')
  })
  it('NEVER blocks even with essentially no free RAM', () => {
    const r = classifyLargeModelMemoryPreflight({ modelSizeBytes: 132.3e9, availableBytes: 0.1e9, totalBytes: 137e9 })
    expect(r.action).toBe('warn')
    expect(r.message).not.toContain('Refusing')
    expect(r.message).toContain('0.1 GB free')
  })
})

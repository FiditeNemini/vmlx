import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

// Execute the production loader with deferred IPC, without launching a model.
const source = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
const start = source.indexOf('const load = async () => {')
const end = source.indexOf('\n    load()', start)
if (start < 0 || end < 0) throw new Error('SessionSettings loader not found')
function loadFixture(config: string, detectConfig: () => Promise<unknown>) {
  const state: any = { metadata: 'loading', session: null, config: {} }
  const setConfig = (value: any) => { state.config = typeof value === 'function' ? value(state.config) : value }
  const context = {
    active: true, sessionId: 'test-session',
    window: { api: { sessions: { get: async () => ({ id: 'test-session', modelPath: '/fixture', config }) }, models: {
      getGenerationDefaults: async () => null, detectConfig,
    } } },
    DEFAULT_CONFIG: { port: 8000 }, setConfig,
    setSession: (value: any) => { state.session = value },
    setDirty: (value: boolean) => { state.dirty = value },
    setMessage: (value: any) => { state.message = value },
    setDetectedConfig: (value: any) => { state.detected = value },
    setPreviewMetadataState: (value: string) => { state.metadata = value },
    applyBundleGenerationDefaultsToSessionConfig: (value: any) => value,
    applyBundleDsv4PoolQuantToSessionConfig: (value: any) => value,
    t: (key: string) => key,
  }
  const execute = new Function(...Object.keys(context), `${source.slice(start, end)}; return load()`)
  return { state, pending: execute(...Object.values(context)) as Promise<void> }
}

describe('CLI preview metadata readiness', () => {
  it('waits for detection before marking the preview ready', async () => {
    let finish!: (value: unknown) => void
    const detection = new Promise(resolve => { finish = resolve })
    const { state, pending } = loadFixture('{}', () => detection)
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve()
    expect(state.metadata).toBe('loading')
    finish({ family: 'minimax_m3', forceTextOnly: true })
    await pending
    expect(state.metadata).toBe('ready')
    expect(state.detected.forceTextOnly).toBe(true)
  })
  it.each(['unknown', 'rejected'])('marks %s detection as incomplete rather than ready', async mode => {
    const { state, pending } = loadFixture('{}', async () => {
      if (mode === 'rejected') throw new Error('metadata unavailable')
      return { family: 'unknown' }
    })
    await pending
    expect(state.metadata).toBe('unavailable')
    expect(state.detected).toBeNull()
  })
  it('renders a recoverable session when saved config JSON is corrupt', async () => {
    const { state, pending } = loadFixture('{broken', async () => ({ family: 'qwen3' }))
    await pending
    expect(state.session?.id).toBe('test-session')
    expect(state.message.type).toBe('error')
    expect(state.config.port).toBe(8000)
    expect(state.dirty).toBe(true)
    expect(state.metadata).toBe('ready')
  })
})

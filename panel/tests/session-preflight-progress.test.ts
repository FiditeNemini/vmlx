import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'
import { createBundleRepairProgressReporter } from '../src/main/bundle-repair-progress'

const source = readFileSync('src/main/sessions.ts', 'utf8')
const ast = ts.createSourceFile('sessions.ts', source, ts.ScriptTarget.Latest, true)
let method = ''
function visit(node: ts.Node) {
  if (ts.isMethodDeclaration(node) && node.name.getText(ast) === 'preflightSessionStart') method = node.getText(ast)
  ts.forEachChild(node, visit)
}
visit(ast)
if (!method) throw new Error('Production preflight method missing')
const code = ts.transpileModule(`return ({${method}}).preflightSessionStart`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const ok = { cache_hit: true, shards: 2, tensors: 7, misaligned_tensors: 0, repairs: [] }
function deferred() {
  let resolve!: (value: typeof ok) => void
  let reject!: (error: Error) => void
  const promise = new Promise<typeof ok>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
function fixture() {
  const checks: Array<{ pending: ReturnType<typeof deferred>; progress: (line: string) => void }> = []
  const events: any[] = []
  const state: any = {
    bundlePreflightCounts: new Map(), bundlePreflightProgress: new Map(), lastLoadProgressEvents: new Map(),
    pendingBundlePreflightLogs: new Map(), logBuffers: new Map(),
    findEnginePath: () => ({ type: 'development' }), validateLocalSessionTarget: vi.fn(),
    pushLog: (id: string, line: string) => state.logBuffers.set(id, [...(state.logBuffers.get(id) || []), line]),
    emit: (name: string, value: any) => events.push({ name, ...value }),
    emitLoadProgress: (value: any) => { state.lastLoadProgressEvents.set(value.sessionId, value); events.push({ name: 'session:loadProgress', ...value }) },
  }
  const session = { id: 'target', type: 'local', modelPath: '/models/target', host: '127.0.0.1', port: 8001, config: '{}', status: 'stopped' }
  const db = { getSession: () => session, updateSession: vi.fn() }
  const runCheck = vi.fn((_engine, _path, progress) => { const pending = deferred(); checks.push({ pending, progress }); return pending.promise })
  const method = new Function('db', 'existsSync', 'createBundleRepairProgressReporter', 'runModelBundleIntegrityPreflight', 'console', 'SessionManager', code)(
    db, () => true, createBundleRepairProgressReporter, runCheck, { log: vi.fn() }, { LOG_BUFFER_MAX_LINES: 2000 },
  )
  return { state, events, checks, session, db, run: () => method.call(state, 'target') }
}
describe('actual preflight progress before engine start', () => {
  it('publishes a truthful pre-process phase without marking the session running/loading or evicting another model', async () => {
    const f = fixture(), pending = f.run()
    expect(f.events[0]).toMatchObject({ name: 'session:loadProgress', sessionId: 'target', preflightActive: true, phase: 'bundle_preflight', indeterminate: true })
    expect(f.db.updateSession).not.toHaveBeenCalled()
    expect(f.session.status).toBe('stopped')
    f.checks[0].progress('[BUNDLE-ALIGNMENT] {"stage":"COPYING","shard":"/models/target/a.safetensors","copied_bytes":1048576,"payload_bytes":4194304}')
    expect(f.state.lastLoadProgressEvents.get('target')).toMatchObject({ preflightActive: true, phase: 'bundle_repair', indeterminate: true })
    expect(f.state.lastLoadProgressEvents.get('target').label).toContain('1.0 / 4.0 MiB')
    f.checks[0].pending.resolve(ok); await pending
    expect(f.events.at(-1)).toEqual({ name: 'session:loadProgress', sessionId: 'target', cleared: true })
    expect(f.state.lastLoadProgressEvents.has('target')).toBe(false)
    expect(f.events.some(e => e.progress === 100 || e.name === 'session:ready')).toBe(false)
  })
  it('clears the pending bar on validation failure and preserves the rejection', async () => {
    const f = fixture(), pending = f.run()
    f.checks[0].pending.reject(new Error('Invalid shard'))
    await expect(pending).rejects.toThrow('Invalid shard')
    expect(f.events.at(-1)).toMatchObject({ cleared: true })
    expect(f.state.lastLoadProgressEvents.size).toBe(0)
    expect(f.state.bundlePreflightCounts.size).toBe(0)
  })
  it('does not clear a still-active overlapping check for the same session', async () => {
    const f = fixture(), first = f.run(), second = f.run()
    f.checks[0].pending.resolve(ok); await first
    expect(f.events.some(e => e.cleared)).toBe(false)
    expect(f.state.lastLoadProgressEvents.get('target').preflightActive).toBe(true)
    f.checks[1].pending.resolve(ok); await second
    expect(f.events.filter(e => e.cleared)).toHaveLength(1)
  })
  it('does not erase newer engine lifecycle progress when preflight finishes', async () => {
    const f = fixture(), pending = f.run()
    const newer = { sessionId: 'target', phase: 'loading_weights', progress: 25, progressGeneration: 1 }
    f.state.emitLoadProgress(newer)
    f.checks[0].pending.resolve(ok); await pending
    expect(f.state.lastLoadProgressEvents.get('target')).toBe(newer)
    expect(f.events.some(e => e.cleared)).toBe(false)
  })
  it('keeps actual copy progress advancing when another check waits on the same bundle', async () => {
    const f = fixture(), first = f.run()
    const copying = (copied_bytes: number) => '[BUNDLE-ALIGNMENT] ' + JSON.stringify({
      stage: 'COPYING', shard: '/models/target/a.safetensors', copied_bytes, payload_bytes: 4194304,
    })
    f.checks[0].progress(copying(1048576))
    const activeCopy = f.state.lastLoadProgressEvents.get('target')
    const second = f.run()
    expect(f.state.lastLoadProgressEvents.get('target')).toBe(activeCopy)
    f.checks[0].progress(copying(2097152))
    expect(f.state.lastLoadProgressEvents.get('target').label).toContain('2.0 / 4.0 MiB')
    f.checks[0].pending.resolve(ok); await first
    expect(f.events.some(e => e.cleared)).toBe(false)
    f.checks[1].pending.resolve(ok); await second
    expect(f.events.filter(e => e.cleared)).toHaveLength(1)
    expect(f.state.bundlePreflightProgress.size).toBe(0)
  })
  it('keeps the surviving check visible and advancing when its overlapping peer fails', async () => {
    const f = fixture(), first = f.run(), second = f.run()
    f.checks[1].pending.reject(new Error('Peer check failed'))
    await expect(second).rejects.toThrow('Peer check failed')
    expect(f.events.some(e => e.cleared)).toBe(false)
    f.checks[0].progress('[BUNDLE-ALIGNMENT] {"stage":"COPYING","shard":"/models/target/a.safetensors","copied_bytes":3145728,"payload_bytes":4194304}')
    expect(f.state.lastLoadProgressEvents.get('target')).toMatchObject({ phase: 'bundle_repair', preflightActive: true })
    expect(f.state.lastLoadProgressEvents.get('target').label).toContain('3.0 / 4.0 MiB')
    f.checks[0].pending.resolve(ok); await first
    expect(f.events.filter(e => e.cleared)).toHaveLength(1)
    expect(f.state.bundlePreflightCounts.size).toBe(0)
    expect(f.state.bundlePreflightProgress.size).toBe(0)
  })
  it('a cached no-op check does not invent a repair notice', async () => {
    const f = fixture(), pending = f.run()
    f.checks[0].pending.resolve(ok); await pending
    expect(f.events.some(e => e.bundleRepairNotice || e.phase === 'bundle_repair')).toBe(false)
  })
  it('a late overlapping check cannot replace or clear a newer engine progress event', async () => {
    const f = fixture(), first = f.run(), second = f.run()
    f.checks[0].pending.resolve(ok); await first
    const newer = { sessionId: 'target', phase: 'loading_weights', progress: 25, progressGeneration: 1 }
    f.state.emitLoadProgress(newer)
    f.checks[1].progress('[BUNDLE-ALIGNMENT] {"stage":"COPYING","shard":"/models/target/a.safetensors"}')
    f.checks[1].pending.resolve(ok); await second
    expect(f.state.lastLoadProgressEvents.get('target')).toBe(newer)
    expect(f.events.some(e => e.cleared)).toBe(false)
  })
})

const context = readFileSync('src/renderer/src/contexts/SessionsContext.tsx', 'utf8')
const contextAst = ts.createSourceFile('SessionsContext.tsx', context, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let handler = ''
let hydrate = ''
function findHandler(node: ts.Node) {
  if (ts.isCallExpression(node) && node.expression.getText(contextAst) === 'window.api.sessions.onLoadProgress') handler = node.arguments[0].getText(contextAst)
  if (ts.isArrowFunction(node) && node.parameters[0]?.name.getText(contextAst) === 'snapshot' && node.getText(contextAst).includes('Object.entries(snapshot)')) hydrate = node.getText(contextAst)
  ts.forEachChild(node, findHandler)
}
findHandler(contextAst)
if (!handler) throw new Error('Production progress subscription missing')
const handlerCode = ts.transpileModule(`return ${handler}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
if (!hydrate) throw new Error('Production snapshot hydration missing')
const hydrateCode = ts.transpileModule(`return ${hydrate}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
describe('renderer receives preflight events and clears', () => {
  function fixture() {
    let entries = new Map<string, any>()
    const progressEventsSeen = new Set<string>()
    const set = (update: any) => { entries = update(entries) }
    const callback = new Function('setLoadProgress', 'progressEventsSeen', handlerCode)(set, progressEventsSeen)
    const hydrate = new Function('setLoadProgress', 'progressEventsSeen', hydrateCode)(set, progressEventsSeen)
    return { callback, hydrate, entries: () => entries }
  }
  it('preserves the live discriminator and measured-stage label before loading status', () => {
    const f = fixture()
    f.callback({ sessionId: 'target', phase: 'bundle_repair', preflightActive: true, label: 'Copying', progress: 0, indeterminate: true })
    expect(f.entries().get('target')).toMatchObject({ phase: 'bundle_repair', preflightActive: true, label: 'Copying', indeterminate: true })
    f.callback({ sessionId: 'target', phase: 'loading_weights', label: 'Loading', progress: 25, indeterminate: false, progressGeneration: 1 })
    expect(f.entries().get('target')).toMatchObject({ preflightActive: false, phase: 'loading_weights', progress: 25 })
  })
  it('clears only the owned session, with no stale repair banner after rejection', () => {
    const f = fixture()
    for (const sessionId of ['target', 'other']) f.callback({ sessionId, preflightActive: true, progress: 0 })
    f.callback({ sessionId: 'target', cleared: true })
    expect(f.entries().has('target')).toBe(false)
    expect(f.entries().get('other').preflightActive).toBe(true)
  })
  it('does not resurrect a completed check from a delayed navigation snapshot', () => {
    const f = fixture()
    const stale = { target: { preflightActive: true, phase: 'bundle_repair', progress: 0 } }
    f.callback({ sessionId: 'target', cleared: true })
    f.hydrate(stale)
    expect(f.entries().has('target')).toBe(false)
    f.hydrate({ other: stale.target })
    expect(f.entries().get('other').preflightActive).toBe(true)
  })
})

import { describe, expect, it } from 'vitest'
import { adoptNativeMtpConfig } from '../src/shared/nativeMtpAdoption'
import { buildNativeMtpLaunchArgs } from '../src/shared/nativeMtpLaunchArgs'
import { planSessionConfigSave } from '../src/shared/sessionConfigLifecycle'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const sessions = ts.createSourceFile('sessions.ts', readFileSync('src/main/sessions.ts', 'utf8'), ts.ScriptTarget.Latest, true)
const preview = ts.createSourceFile('SessionSettings.tsx', readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)

function findNode(tree: ts.Node, matches: (node: ts.Node) => boolean): ts.Node {
  if (matches(tree)) return tree
  let found: ts.Node | undefined
  tree.forEachChild(node => { if (!found) { try { found = findNode(node, matches) } catch { /* keep searching */ } } })
  if (!found) throw new Error('Owning source expression not found')
  return found
}

function evaluateOwnedExpression(node: ts.Node, tree: ts.SourceFile, env: Record<string, unknown>): any {
  const js = ts.transpileModule(`const owned = ${node.getText(tree)};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS },
  }).outputText
  return new Function(...Object.keys(env), `${js}; return owned;`)(...Object.values(env))
}

// Execute the actual adoption-to-session JSON projection rather than a mirror.
function persistedAdoption(proc: Parameters<typeof adoptNativeMtpConfig>[0], defaultMode?: 'auto' | 'off') {
  const projection = findNode(sessions, node => ts.isArrowFunction(node)
    && node.getText(sessions).includes('const adopted = adoptNativeMtpConfig('))
  return JSON.parse(JSON.stringify(evaluateOwnedExpression(projection, sessions, {
    adoptNativeMtpConfig, proc, detectedFamily: 'qwen4-exp', detected: { nativeMtp: { depth: 1, defaultMode } },
  })()))
}

// Execute each real producer's policy input into the shared launch resolver.
function ownedLaunchArgs(tree: ts.SourceFile, config: Record<string, unknown>, defaultMode?: 'auto' | 'off') {
  const call = findNode(tree, node => ts.isCallExpression(node)
    && node.expression.getText(tree) === 'buildNativeMtpLaunchArgs') as ts.CallExpression
  const input = evaluateOwnedExpression(call.arguments[0], tree, {
    config, nativeMtp: { supported: true, depth: 1, defaultMode }, mode: config.nativeMtpMode, compatibleExternalSpeculative: false,
  })
  return buildNativeMtpLaunchArgs(input)
}

const ADAPTIVE = ['--native-mtp-depth-policy', 'adaptive', '--native-mtp-sampling-policy', 'compatible-only']

describe('native MTP adoption round-trip (two product modes)', () => {
  it('adopts a disabled engine as off and relaunches disabled', () => {
    for (const proc of [{ nativeMtpDisabled: true }, { nativeMtpSamplingPolicy: 'disabled' as const }]) {
      const adopted = adoptNativeMtpConfig(proc, 'qwen4-exp')
      expect(adopted).toEqual({ nativeMtpMode: 'off', nativeMtpAdoptionSource: 'process' })
      expect(buildNativeMtpLaunchArgs({ supported: true, mode: adopted.nativeMtpMode })).toEqual(['--disable-native-mtp'])
    }
  })

  it('adopts any live MTP policy (including a legacy fixed/greedy process) as adaptive', () => {
    for (const proc of [
      { nativeMtpSamplingPolicy: 'compatible-only' as const, nativeMtpDepthPolicy: 'adaptive' as const },
      { nativeMtpSamplingPolicy: 'deterministic-defaults' as const, nativeMtpDepthPolicy: 'fixed' as const, nativeMtpDepth: 3 },
      { nativeMtpSamplingPolicy: 'greedy-only' as const, nativeMtpDepthPolicy: 'fixed' as const, nativeMtpDepth: 2 },
      { nativeMtpDepth: 1 },
      { nativeMtpDepthPolicy: 'fixed' as const },
    ]) {
      const adopted = adoptNativeMtpConfig(proc, 'qwen3.5')
      expect(adopted).toEqual({ nativeMtpMode: 'adaptive', nativeMtpAdoptionSource: 'process' })
      expect(buildNativeMtpLaunchArgs({ supported: true, mode: adopted.nativeMtpMode })).toEqual(ADAPTIVE)
    }
  })

  it('uses the model-derived default only when the process exposes nothing', () => {
    expect(adoptNativeMtpConfig({}, 'qwen4-exp')).toEqual({ nativeMtpMode: 'adaptive', nativeMtpAdoptionSource: 'model-default' })
    expect(adoptNativeMtpConfig({}, 'glm5-next', 'off')).toEqual({ nativeMtpMode: 'off', nativeMtpAdoptionSource: 'model-default' })
  })

  it('persists only nativeMtpMode through the actual adoption projection', () => {
    expect(persistedAdoption({ nativeMtpSamplingPolicy: 'compatible-only', nativeMtpDepthPolicy: 'adaptive' })).toEqual({ nativeMtpMode: 'adaptive' })
    expect(persistedAdoption({ nativeMtpDisabled: true })).toEqual({ nativeMtpMode: 'off' })
    expect(persistedAdoption({}, 'off')).toEqual({ nativeMtpMode: 'off' })
  })

  it.each([
    [{ nativeMtpMode: 'adaptive' }, ADAPTIVE],
    [{ nativeMtpMode: 'auto' }, ADAPTIVE],
    [{ nativeMtpMode: 'deterministic', nativeMtpDepth: 2, nativeMtpDepthOverride: true }, ADAPTIVE],
    [{ nativeMtpMode: 'auto', nativeMtpAutoSamplingPolicy: 'deterministic-defaults' }, ADAPTIVE],
    [{ nativeMtpMode: 'off' }, ['--disable-native-mtp']],
    [{}, ADAPTIVE],
  ])('both launch producers agree for saved config %j', (saved, expected) => {
    expect(ownedLaunchArgs(sessions, saved)).toEqual(expected)
    expect(ownedLaunchArgs(preview, saved)).toEqual(expected)
  })

  it('legacy auto on a measured-off bundle stays AR; explicit adaptive opts in; both producers agree', () => {
    for (const saved of [{}, { nativeMtpMode: 'auto' }]) {
      expect(ownedLaunchArgs(sessions, saved, 'off')).toEqual(['--disable-native-mtp'])
      expect(ownedLaunchArgs(preview, saved, 'off')).toEqual(['--disable-native-mtp'])
    }
    for (const saved of [{ nativeMtpMode: 'adaptive' }, { nativeMtpMode: 'deterministic' }]) {
      expect(ownedLaunchArgs(sessions, saved, 'off')).toEqual(ADAPTIVE)
      expect(ownedLaunchArgs(preview, saved, 'off')).toEqual(ADAPTIVE)
    }
  })

  it('a mode change is staged until restart and only nativeMtpMode is restart-relevant among the MTP keys', () => {
    const owner = findNode(sessions, node => ts.isPropertyDeclaration(node) && node.name.getText(sessions) === 'RESTART_REQUIRED_KEYS') as ts.PropertyDeclaration
    const restartKeys = evaluateOwnedExpression(owner.initializer!, sessions, {}) as Set<string>
    expect(restartKeys.has('nativeMtpMode')).toBe(true)
    for (const retired of ['nativeMtpDepth', 'nativeMtpDepthOverride', 'nativeMtpAutoSamplingPolicy']) {
      expect(restartKeys.has(retired)).toBe(false)
    }
    const plan = planSessionConfigSave({ type: 'local', status: 'running' }, { nativeMtpMode: 'adaptive' }, { nativeMtpMode: 'off' }, restartKeys)
    expect(plan.changedKeys).toEqual(['nativeMtpMode'])
    expect(plan.restartRequired).toBe(true)
    expect(ownedLaunchArgs(sessions, plan.config)).toEqual(ADAPTIVE)
    expect(ownedLaunchArgs(sessions, JSON.parse(JSON.stringify(plan.pendingConfig)))).toEqual(['--disable-native-mtp'])
  })

  it('adoption of an existing row preserves saved and pending configs verbatim', () => {
    const owner = findNode(sessions, node => ts.isVariableDeclaration(node) && node.name.getText(sessions) === 'adoptedConfig') as ts.VariableDeclaration
    const session = { config: '{"nativeMtpMode":"off","maxTokens":8192}', pendingConfig: '{"nativeMtpMode":"adaptive"}' }
    expect(evaluateOwnedExpression(owner.initializer!, sessions, { session })).toBe(session.config)
    const pending = findNode(sessions, node => ts.isVariableDeclaration(node) && node.name.getText(sessions) === 'adoptedPendingConfig') as ts.VariableDeclaration
    expect(evaluateOwnedExpression(pending.initializer!, sessions, { session })).toBe(session.pendingConfig)
  })
})

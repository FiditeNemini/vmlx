import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'

const source = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
const tree = ts.createSourceFile('SessionConfigForm.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let effect: ts.ArrowFunction | undefined
let bundlePathExpression: string | undefined
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === 'bundlePath') {
    bundlePathExpression = node.initializer?.getText(tree)
  }
  if (ts.isCallExpression(node) && node.expression.getText(tree) === 'useEffect' &&
      node.arguments[0]?.getText(tree).includes('api.dflash2Drafters')) effect = node.arguments[0] as ts.ArrowFunction
  node.forEachChild(visit)
}
visit(tree)
if (!effect || !bundlePathExpression) throw new Error('Production drafter discovery effect not found')
const code = ts.transpileModule(`const bundlePath = ${bundlePathExpression}; return (${effect.getText(tree)})();`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText
function start(lookup: any, publish: any, modelPath = '/models/target') {
  return new Function('window', 'setDflash2Drafters', 'isImage', 'modelPath', 'qwen38Dense27b', 'modelIdentity', code)(
    { api: { models: { dflash2Drafters: lookup } } }, publish, false, modelPath, true, `Display name ${modelPath}`,
  ) as (() => void)
}
const bundled = [{ path: '/models/target/dflash2', name: 'bundled', bundled: true }]
const tick = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve() }

describe('production bundled-drafter discovery effect', () => {
  it('looks up the complete filesystem path when folder names contain spaces', async () => {
    const path = '/Volumes/Model Drive/Qwen 27B'
    const lookup = vi.fn(() => Promise.resolve([]))
    start(lookup, vi.fn(), path)
    await tick()
    expect(lookup.mock.calls).toEqual([[path, false], [path, true]])
  })
  it('publishes bundled capability while the library scan is pending', async () => {
    let finish!: (x: unknown) => void
    const lookup = vi.fn((_path, scan) => scan ? new Promise(r => { finish = r }) : Promise.resolve(bundled))
    const publish = vi.fn()
    start(lookup, publish)
    await tick()
    expect(lookup.mock.calls.map(c => c[1])).toEqual([false, true])
    expect(publish).toHaveBeenLastCalledWith(bundled)
    const all = [...bundled, { path: '/models/external', name: 'external' }]
    finish(all); await tick()
    expect(publish).toHaveBeenLastCalledWith(all)
  })
  it('retains bundled capability when optional library scanning fails', async () => {
    const publish = vi.fn()
    start(vi.fn((_path, scan) => scan ? Promise.reject(new Error('offline volume')) : Promise.resolve(bundled)), publish)
    await tick()
    expect(publish).toHaveBeenLastCalledWith(bundled)
  })
  it('does not publish stale results or start a scan after unmount', async () => {
    let finish!: (x: unknown) => void
    const lookup = vi.fn(() => new Promise(r => { finish = r }))
    const publish = vi.fn()
    const cancel = start(lookup, publish)
    cancel(); finish(bundled); await tick()
    expect(publish.mock.calls).toEqual([[[]]])
    expect(lookup).toHaveBeenCalledTimes(1)
  })
})

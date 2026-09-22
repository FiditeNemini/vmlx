import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

// Exercise the actual input callbacks. Native Electron proof covers rendering
// and the subsequent save/launch path separately.
function callback(name: string, scope: Record<string, unknown>) {
  const file = ts.createSourceFile('form.tsx', readFileSync(
    'src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8'),
    ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  let expression: ts.Expression | undefined
  function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(file) === name) expression = node.initializer
    ts.forEachChild(node, visit)
  }
  visit(file)
  if (!expression) throw new Error(`Missing ${name}`)
  const js = ts.transpileModule(`const handler = ${expression.getText(file)};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText
  return new Function(...Object.keys(scope), `${js}\nreturn handler;`)(...Object.values(scope))
}

describe('typed cache and output controls preserve user intent', () => {
  for (const [raw, expected, fractional] of [['0', 0, false], ['129', 129, false], ['0.003', .003, true]] as const) {
    it(`publishes and commits ${raw} without changing its meaning`, () => {
      const changed: number[] = []
      const scope = { min: fractional ? 0 : 1, maxInput: undefined, allowUnlimited: true,
        unlimitedValue: 0, allowFractional: fractional, defaultValue: 0,
        isUnlimited: false, localInput: raw, setLocalInput: () => {},
        onChange: (value: number) => changed.push(value) }
      callback('handleInputChange', scope)({ target: { value: raw } })
      callback('handleInputBlur', scope)()
      expect(changed).toEqual([expected, expected])
    })
  }
  it('clearing an output override restores the configured default', () => {
    const changed: number[] = []
    callback('handleInputBlur', { min: 1, allowUnlimited: true, unlimitedValue: 0,
      allowFractional: false, maxInput: undefined, defaultValue: 0,
      isUnlimited: false, localInput: '', setLocalInput: () => {},
      onChange: (value: number) => changed.push(value) })()
    expect(changed).toEqual([0])
  })
})

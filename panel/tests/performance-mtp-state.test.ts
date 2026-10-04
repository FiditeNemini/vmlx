import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

// Execute the actual JSX value expression, not a copied status mapper.
const filename = join(__dirname, '../src/renderer/src/components/sessions/PerformancePanel.tsx')
const source = ts.createSourceFile(filename, readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let expression: string | undefined
function visit(node: ts.Node): void {
  if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(source) === 'InfoCard') {
    const attrs = node.attributes.properties.filter(ts.isJsxAttribute)
    const label = attrs.find(a => a.name.getText(source) === 'label')
    if (label?.initializer?.getText(source) === "{t('sessions.performance.mtp')}") {
      const value = attrs.find(a => a.name.getText(source) === 'value')?.initializer
      if (value && ts.isJsxExpression(value)) expression = value.expression?.getText(source)
    }
  }
  ts.forEachChild(node, visit)
}
visit(source)
if (!expression) throw new Error('MTP InfoCard expression not found')
const render = new Function('health', 't', `return (${expression})`) as (
  health: { mtp: Record<string, unknown> }, t: (key: string) => string
) => string
const label = (mtp: Record<string, unknown>) => render({ mtp }, key => key)

describe('Performance MTP status from engine health', () => {
  it.each([false, true])('shows explicit Off as disabled, not unwired (runtime_available=%s)', available => {
    expect(label({ status: 'runtime_disabled', artifact_available: true, runtime_available: available, runtime_active: false }))
      .toBe('sessions.cache.statusDisabled')
  })

  it('does not infer disabled from weights alone', () => {
    expect(label({ status: 'artifact_only', artifact_available: true, runtime_available: false, runtime_active: false }))
      .toBe('sessions.performance.weightsPresentRuntimeUnwired')
  })

  it('retains a ready but inactive supported runtime', () => {
    expect(label({ status: 'runtime_ready', artifact_available: true, runtime_available: true, runtime_active: false }))
      .toBe('sessions.performance.weightsPresentRuntimeReady')
  })

  it('labels an attached runtime ready, not observed execution', () => {
    expect(label({ status: 'native_runtime_active', runtime_active: true, runtime_scope: 'text+vl' }))
      .toBe('sessions.performance.weightsPresentRuntimeReady (text+vl)')
  })

  it('retains an explicit engine reason without an artifact', () => {
    expect(label({ status: 'missing_weights', artifact_available: false, runtime_available: false, runtime_active: false }))
      .toBe('missing weights')
  })
})

// Execute the actual historical execution card guard: a loaded head or a
// different last request must never become current-request execution proof.
let observedGuard: string | undefined
function findObservedGuard(node: ts.Node): void {
  if (ts.isJsxExpression(node) && node.expression
      && node.getText(source).includes("label={t('sessions.performance.mtpLast')}")) {
    const expression = node.expression
    if (ts.isBinaryExpression(expression)) observedGuard = expression.left.getText(source)
  }
  ts.forEachChild(node, findObservedGuard)
}
findObservedGuard(source)
if (!observedGuard) throw new Error('Historical MTP guard not found')
const observed = new Function('lastNativeMtp', `return !!(${observedGuard})`) as (row: unknown) => boolean
it('shows historical execution only with a real request ID and completed cycles', () => {
  expect(observed(undefined)).toBe(false)
  expect(observed({ request_id: 'request-a', cycles: 0 })).toBe(false)
  expect(observed({ cycles: 2 })).toBe(false)
  expect(observed({ request_id: 'request-a', cycles: 2, accepted_tokens: 0 })).toBe(true)
})

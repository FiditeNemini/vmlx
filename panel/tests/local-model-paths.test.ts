import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'fs'
import { tmpdir } from 'os'
import { dirname, join } from 'path'
import { loadLocalModelPaths } from './helpers/local-model-paths'

const dirs: string[] = []
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })
function manifest(payload: unknown): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-fixture-manifest-'))
  dirs.push(dir)
  const path = join(dir, 'paths.json')
  writeFileSync(path, JSON.stringify(payload))
  return path
}
describe('local model fixture manifests', () => {
  it('leaves unconfigured rows unavailable', () => {
    expect(loadLocalModelPaths(undefined, ['hy3'])).toEqual({})
  })
  it.each([null, [], 'value', { unknown: '/tmp' }, { hy3: 7 }, { hy3: 'relative' }])('rejects invalid manifest %j', payload => {
    expect(() => loadLocalModelPaths(manifest(payload), ['hy3'])).toThrow()
  })
  it('rejects an explicit absent fixture instead of skipping it', () => {
    const p = manifest({})
    writeFileSync(p, JSON.stringify({ hy3: join(p, 'absent') }))
    expect(() => loadLocalModelPaths(p, ['hy3'])).toThrow()
  })
  it('requires a regular config file and preserves exact mapped paths', () => {
    const p = manifest({})
    const model = join(dirname(p), 'model')
    mkdirSync(model)
    writeFileSync(p, JSON.stringify({ hy3: model }))
    expect(() => loadLocalModelPaths(p, ['hy3'])).toThrow()
    mkdirSync(join(model, 'config.json'))
    expect(() => loadLocalModelPaths(p, ['hy3'])).toThrow()
    rmSync(join(model, 'config.json'), { recursive: true })
    writeFileSync(join(model, 'config.json'), '{}')
    expect(loadLocalModelPaths(p, ['hy3', 'dsv4_k'])).toEqual({ hy3: model })
  })
})

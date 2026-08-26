import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('engine child path isolation', () => {
  const source = readFileSync(
    new URL('../src/main/sessions.ts', import.meta.url),
    'utf8',
  )

  it('keeps a packaged Python child from importing a sibling mlx repo from the launcher cwd', () => {
    expect(source).toContain("PYTHONSAFEPATH: '1'")
    expect(source).toContain('cwd: developmentSourceRoot || dirname(engineResult.pythonPath)')
    expect(source).toContain('PYTHONPATH: developmentSourceRoot')
  })

  it('also gives a system engine binary a stable executable-owned cwd', () => {
    expect(source).toContain('cwd: dirname(engineResult.binaryPath)')
  })

  it('pins a reused system venv to the current development checkout source', () => {
    expect(source).toContain('getDevelopmentSourceRoot,')
    expect(source).toContain('const developmentSourceRoot = getDevelopmentSourceRoot() || undefined')
    expect(source).toContain('sourceRoot: developmentSourceRoot')
    expect(source).toContain('systemEnv.PYTHONPATH = engineResult.sourceRoot')
    expect(source).toContain('[SESSIONS] Development engine source:')
  })

  it('pins a symlinked project venv to the current development checkout source', () => {
    expect(source).toContain("type: 'development'")
    expect(source).toContain("engineResult.type === 'bundled' || engineResult.type === 'development'")
    expect(source).toContain("engineResult.type === 'development'")
    expect(source).toContain('PYTHONPATH: developmentSourceRoot')
    expect(source).toContain('cwd: developmentSourceRoot || dirname(engineResult.pythonPath)')
  })
})

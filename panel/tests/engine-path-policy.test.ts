import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('engine path policy', () => {
  it('prefers the repo project venv before stale user/system vmlx-engine binaries in dev mode', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const findEnginePath = source.slice(
      source.indexOf('findEnginePath(): EnginePath | null'),
      source.indexOf('  private async findAvailablePort'),
    )

    const projectVenvIndex = findEnginePath.indexOf('Development builds must exercise the source tree')
    const systemSearchIndex = findEnginePath.indexOf('// System binary search')
    const staleSystemIndex = findEnginePath.indexOf('ENGINE_SEARCH_DIRS.flatMap')

    expect(projectVenvIndex).toBeGreaterThanOrEqual(0)
    expect(systemSearchIndex).toBeGreaterThanOrEqual(0)
    expect(staleSystemIndex).toBeGreaterThanOrEqual(0)
    expect(projectVenvIndex).toBeLessThan(systemSearchIndex)
    expect(projectVenvIndex).toBeLessThan(staleSystemIndex)
  })

  it('uses one project-venv probe for setup detection and development session startup', () => {
    const engineManager = readFileSync('src/main/engine-manager.ts', 'utf8')
    const sessions = readFileSync('src/main/sessions.ts', 'utf8')
    const checkInstallation = engineManager.slice(
      engineManager.indexOf('export async function checkEngineInstallation'),
      engineManager.indexOf('async function getVersionFromBinary'),
    )
    const findEnginePath = sessions.slice(
      sessions.indexOf('findEnginePath(): EnginePath | null'),
      sessions.indexOf('  private async findAvailablePort'),
    )

    expect(engineManager).toContain('export function getDevelopmentProjectVenv(')
    expect(engineManager).toContain("join(sourceRoot, '.venv', 'bin', 'python3')")
    expect(engineManager).toContain("PYTHONPATH: ''")
    expect(checkInstallation).toContain(
      'const projectVenv = getDevelopmentProjectVenv(developmentSourceRoot)',
    )
    expect(checkInstallation.indexOf('getDevelopmentProjectVenv')).toBeLessThan(
      checkInstallation.indexOf('// 1. Check common paths'),
    )
    expect(sessions).toContain('getDevelopmentProjectVenv,')
    expect(sessions).toContain('getDevelopmentSourceRoot,')
    expect(findEnginePath).toContain(
      'const projectVenv = getDevelopmentProjectVenv(developmentSourceRoot || null)',
    )
    expect(findEnginePath).not.toContain("join(sourceDir, '.venv', 'bin', 'python3')")
  })

  it('uses the published vmlx package name for PyPI installs while preserving vmlx-engine entrypoint detection', () => {
    const engineManager = readFileSync('src/main/engine-manager.ts', 'utf8')
    const sessions = readFileSync('src/main/sessions.ts', 'utf8')
    const createSession = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')

    expect(engineManager).toContain("const PYPI_PACKAGE_NAME = 'vmlx'")
    expect(engineManager).toContain("export const ENGINE_ENTRY_POINT_NAMES = ['vmlx-engine', 'vmlx-serve', 'vmlx']")
    expect(sessions).toContain('ENGINE_ENTRY_POINT_NAMES,')
    expect(sessions).toContain('ENGINE_SEARCH_DIRS,')
    expect(sessions).toContain('ENGINE_ENTRY_POINT_NAMES.map')
    expect(engineManager).toContain("const pkg = bundledSource || PYPI_PACKAGE_NAME")
    expect(engineManager).toContain("['tool', 'upgrade', PYPI_PACKAGE_NAME]")
    expect(engineManager).not.toContain("const pkg = bundledSource || 'vmlx-engine'")
    expect(engineManager).not.toContain("['tool', 'upgrade', 'vmlx-engine']")

    expect(createSession).toContain('uv tool install vmlx')
    expect(createSession).toContain('pip3 install vmlx')
    expect(createSession).not.toContain('uv tool install vmlx-engine')
    expect(createSession).not.toContain('pip3 install vmlx-engine')
  })

  it('prefers the imported engine version over stale editable-install metadata', () => {
    const source = readFileSync('src/main/engine-manager.ts', 'utf8')
    const getVersion = source.slice(
      source.indexOf('async function getVersionFromBinary'),
      source.indexOf('function detectInstallMethod'),
    )

    const moduleImport = getVersion.indexOf('import vmlx_engine')
    const moduleVersion = getVersion.indexOf("getattr(vmlx_engine, '__version__', '')")
    const metadataFallback = getVersion.indexOf("for name in ('vmlx', 'vmlx-engine')")

    expect(moduleImport).toBeGreaterThanOrEqual(0)
    expect(moduleVersion).toBeGreaterThan(moduleImport)
    expect(metadataFallback).toBeGreaterThan(moduleVersion)
  })

  it('probes the same source root that development session launches pin on PYTHONPATH', () => {
    const source = readFileSync('src/main/engine-manager.ts', 'utf8')
    const checkInstallation = source.slice(
      source.indexOf('export async function checkEngineInstallation'),
      source.indexOf('function detectInstallMethod'),
    )
    const bundledSource = source.slice(
      source.indexOf('function getBundledSourcePath'),
      source.indexOf('function buildInstallCommand'),
    )

    expect(source).toContain('function getDevelopmentSourceRoot(): string | null')
    expect(source).toContain('if (app.isPackaged) return null')
    expect(checkInstallation).toContain(
      'const developmentSourceRoot = getDevelopmentSourceRoot()',
    )
    expect(checkInstallation).toContain(
      'getVersionFromBinary(path, developmentSourceRoot)',
    )
    expect(checkInstallation).toContain(
      "PYTHONPATH: developmentSourceRoot || ''",
    )
    expect(bundledSource).toContain('return getDevelopmentSourceRoot()')
  })
})

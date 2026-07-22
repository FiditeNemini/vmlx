import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

const repo = join(__dirname, '..')
const requireCjs = createRequire(import.meta.url)

function read(rel: string): string {
  return readFileSync(join(repo, rel), 'utf8')
}

describe('release packaging', () => {
  it('removes bundled pip distlib Windows launcher stubs before app signing', async () => {
    const afterPack = requireCjs(join(repo, 'scripts/electron-builder-after-pack.cjs'))
    const temp = mkdtempSync(join(tmpdir(), 'vmlx-after-pack-'))
    try {
      const bundledPython = join(
        temp,
        'vMLX.app',
        'Contents',
        'Resources',
        'bundled-python',
        'python',
        'lib',
        'python3.12',
        'site-packages',
        'pip',
        '_vendor',
        'distlib',
      )
      mkdirSync(bundledPython, { recursive: true })
      const launcher = join(bundledPython, 't32.exe')
      writeFileSync(launcher, 'windows launcher stub')

      await afterPack({
        appOutDir: temp,
        packager: { appInfo: { productFilename: 'vMLX' } },
      })

      expect(afterPack.isBundledPipDistlibWindowsLauncher(launcher)).toBe(true)
      expect(() => readFileSync(launcher)).toThrow()
    } finally {
      rmSync(temp, { recursive: true, force: true })
    }
  })

  it('keeps the afterPack hook scoped to pip distlib exe launchers', () => {
    const source = read('scripts/electron-builder-after-pack.cjs')

    expect(source).toContain('site-packages/pip/_vendor/distlib')
    expect(source).toContain('removeBundledWindowsLaunchers')
    expect(source).toContain('removedWindowsLaunchers')
  })
})

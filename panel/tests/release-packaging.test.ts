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

  it('refuses symlinked or cross-checkout release dependencies', () => {
    const source = read('scripts/build-release-dmgs.sh')

    expect(source).toContain('release node_modules must not be a symlink')
    expect(source).toContain('NODE_MODULES_REAL=')
    expect(source).toContain('release node_modules resolves outside this checkout')
  })

  it('uses recursive Developer-ID staging before final audit and reseal', () => {
    const source = read('scripts/build-release-dmgs.sh')
    const stage = source.indexOf(
      'npx electron-builder --mac --dir',
    )
    const finalSign = source.indexOf(
      'finalize_release_app_signature "$app_path" "$RELEASE_CODESIGN_IDENTITY"',
    )

    expect(stage).toBeGreaterThan(0)
    expect(finalSign).toBeGreaterThan(stage)
    expect(source).not.toContain(
      'CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder --mac --dir',
    )
    expect(source).toContain('inside-out Developer-ID signing')
  })

  it('Developer-ID signs and audits Mach-O leaves outside bundled Python', () => {
    const source = read('scripts/build-release-dmgs.sh')

    expect(source).toContain('sign_remaining_app_macho_leaves()')
    expect(source).toContain('Signature=adhoc|flags=.*adhoc|TeamIdentifier=not set')
    expect(source).toContain('verify_release_macho_leaves()')
    expect(source).toContain('^Authority=Developer ID Application:')
    expect(source).toContain('^Timestamp=')
    expect(source).toContain('flags=.*runtime')
    expect(source).toContain('sign_remaining_app_macho_leaves "$app_path" "$identity"')
    expect(source).toContain('verify_release_macho_leaves "$app_path"')
  })
})

import { existsSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import {
  ensureUserDataDirExists,
  resolveUserDataDirOverride,
} from '../src/main/user-data-dir'

describe('app user-data isolation bootstrap', () => {
  const source = () =>
    readFileSync(resolve(process.cwd(), 'src/main/index.ts'), 'utf8')

  it('applies --vmlx-user-data-dir before taking the single-instance lock', () => {
    const main = source()
    const bootstrapIndex = main.indexOf("import './user-data-dir'")
    const databaseIndex = main.indexOf("from './database'")
    const lockIndex = main.indexOf('app.requestSingleInstanceLock()')

    expect(bootstrapIndex).toBeGreaterThanOrEqual(0)
    expect(databaseIndex).toBeGreaterThanOrEqual(0)
    expect(lockIndex).toBeGreaterThanOrEqual(0)
    expect(bootstrapIndex).toBeLessThan(databaseIndex)
    expect(bootstrapIndex).toBeLessThan(lockIndex)
  })

  it('honors the isolated-secondary proof gate before taking the app-wide lock', () => {
    const main = source()

    expect(main).toContain("import { shouldAllowSecondaryInstance } from '../shared/userDataOverride'")
    expect(main).toContain(
      'const allowSecondaryInstance = shouldAllowSecondaryInstance(process.argv, process.env)',
    )
    expect(main).toContain(
      'const gotTheLock = allowSecondaryInstance || app.requestSingleInstanceLock()',
    )
  })

  it('supports environment override for non-UI packaged smoke tests', () => {
    expect(resolveUserDataDirOverride(['vMLX'], { VMLX_USER_DATA_DIR: 'build/user-data' })).toMatch(
      /build\/user-data$/,
    )
    expect(resolveUserDataDirOverride(['vMLX'], { VMLINUX_USER_DATA_DIR: 'build/legacy-user-data' })).toMatch(
      /build\/legacy-user-data$/,
    )
  })

  it('supports --vmlx-user-data-dir forms for repo-local dev app launches', () => {
    expect(resolveUserDataDirOverride(['vMLX', '--vmlx-user-data-dir=/tmp/vmlx-a'], {})).toBe(
      '/tmp/vmlx-a',
    )
    expect(resolveUserDataDirOverride(['vMLX', '--vmlx-user-data-dir', '/tmp/vmlx-b'], {})).toBe(
      '/tmp/vmlx-b',
    )
  })

  it('creates a fresh override directory before database startup', () => {
    const profile = resolve(
      tmpdir(),
      `vmlx-user-data-bootstrap-${process.pid}-${Date.now()}`,
      'nested',
      'profile',
    )
    rmSync(resolve(profile, '..', '..'), { recursive: true, force: true })

    try {
      ensureUserDataDirExists(profile)
      expect(existsSync(profile)).toBe(true)
    } finally {
      rmSync(resolve(profile, '..', '..'), { recursive: true, force: true })
    }
  })
})

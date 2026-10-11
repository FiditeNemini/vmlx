import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { rows, db } = vi.hoisted(() => {
  const rows: any[] = []
  const db = {
    getSessions: () => rows,
    getSession: (id: string) => rows.find(r => r.id === id),
    getSessionByModelPath: (path: string) => rows.find(r => r.modelPath === path),
    getSetting: () => undefined,
    createSession: (row: any) => rows.push(row),
    updateSession: (id: string, patch: any) => Object.assign(rows.find(r => r.id === id), patch),
  }
  return { rows, db }
})
vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({ app: { getAppPath: () => process.cwd(), getPath: () => '/tmp', isPackaged: false },
  powerSaveBlocker: { isStarted: () => false, start: () => 1, stop: () => undefined } }))
import { SessionManager } from '../src/main/sessions'
const dirs: string[] = []
function bundle(modelType: string, maxNew = 4096): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-cap-intent-'))
  dirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: modelType }))
  writeFileSync(join(dir, 'generation_config.json'), JSON.stringify({ max_new_tokens: maxNew }))
  return dir
}
function args(manager: SessionManager, row: any): string[] {
  return (manager as any).buildArgs({ ...JSON.parse(row.config), modelPath: row.modelPath })
}
beforeEach(() => { rows.length = 0 })
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })

describe('DFlash2 session SSD launch wiring', () => {
  it.each(['qwen3_5', 'glm5_next'])('preserves the native SSD settings for %s with a bundled drafter', async family => {
    const manager = new SessionManager()
    const modelPath = bundle(family)
    mkdirSync(join(modelPath, 'dflash2'))
    writeFileSync(join(modelPath, 'dflash2', 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }))
    const row = await manager.createSession(modelPath, { port: 8123,
      enableBlockDiskCache: true, blockDiskCacheDir: '/private/test-native-ssd', blockDiskCacheMaxPercent: 7 })
    const command = args(manager, row)
    expect(command).toContain('--no-continuous-batching')
    expect(command).toContain('--enable-block-disk-cache')
    expect(command[command.indexOf('--block-disk-cache-dir') + 1]).toBe('/private/test-native-ssd')
    expect(command[command.indexOf('--block-disk-cache-max-percent') + 1]).toBe('7')
    expect(command).not.toContain('--use-paged-cache')
    await manager.updateSessionConfig(row.id, { enableBlockDiskCache: false })
    expect(args(manager, row)).toContain('--disable-block-disk-cache')
  })
})

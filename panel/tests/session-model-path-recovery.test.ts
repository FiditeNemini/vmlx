import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  classifySessionModelPaths,
  validateModelBundleDirectory,
} from '../src/main/session-model-path'

const createdDirs: string[] = []

function temporaryRoot(): string {
  const root = mkdtempSync(join(tmpdir(), 'vmlx-session-path-'))
  createdDirs.push(root)
  return root
}

afterEach(() => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir) rmSync(dir, { recursive: true, force: true })
  }
})

describe('session model-path list classification', () => {
  it('flags missing local paths and identifies an existing identity twin', () => {
    const root = temporaryRoot()
    const twinPath = join(root, 'mounted', 'Model-A')
    mkdirSync(twinPath, { recursive: true })

    const sessions = [
      { id: 'stale', modelPath: join(root, 'unmounted', 'Model-A'), type: 'local' as const },
      { id: 'twin', modelPath: twinPath, type: 'local' as const },
      { id: 'gone', modelPath: join(root, 'deleted', 'Model-B'), type: 'local' as const },
      { id: 'remote', modelPath: 'remote://Model-C@example.test', type: 'remote' as const },
    ]

    expect(classifySessionModelPaths(sessions)).toEqual([
      { ...sessions[0], modelPathMissing: true, usableTwinId: 'twin' },
      { ...sessions[1], modelPathMissing: false },
      { ...sessions[2], modelPathMissing: true },
      { ...sessions[3], modelPathMissing: false },
    ])
  })

  it('is list metadata only: it neither removes nor mutates persisted sessions', () => {
    const sessions = Object.freeze([
      Object.freeze({ id: 'stale', modelPath: '/missing/Model-A', type: 'local' as const }),
      Object.freeze({ id: 'remote', modelPath: 'remote://Model-B@example.test', type: 'remote' as const }),
    ])

    const classified = classifySessionModelPaths(sessions, () => false)

    expect(classified.map((session) => session.id)).toEqual(['stale', 'remote'])
    expect(sessions[0]).not.toHaveProperty('modelPathMissing')
  })
})

describe('repoint model-bundle validation', () => {
  it('rejects a selected directory without config.json', () => {
    const directory = temporaryRoot()

    expect(validateModelBundleDirectory(directory)).toMatchObject({
      valid: false,
      path: directory,
      error: expect.stringContaining('config.json'),
    })
  })

  it('accepts a directory that contains config.json', () => {
    const directory = temporaryRoot()
    writeFileSync(join(directory, 'config.json'), '{}')

    expect(validateModelBundleDirectory(`${directory}/`)).toEqual({
      valid: true,
      path: directory,
    })
  })
})

describe('session repoint persistence contract', () => {
  const databaseSource = readFileSync(
    new URL('../src/main/database.ts', import.meta.url),
    'utf8',
  )
  const sessionsSource = readFileSync(
    new URL('../src/main/sessions.ts', import.meta.url),
    'utf8',
  )
  const ipcSource = readFileSync(
    new URL('../src/main/ipc/sessions.ts', import.meta.url),
    'utf8',
  )
  const dashboardSource = readFileSync(
    new URL('../src/renderer/src/components/sessions/SessionDashboard.tsx', import.meta.url),
    'utf8',
  )
  const cardSource = readFileSync(
    new URL('../src/renderer/src/components/sessions/SessionCard.tsx', import.meta.url),
    'utf8',
  )

  it('updates the session and rebinds matching chats in one transaction', () => {
    const block = databaseSource.slice(
      databaseSource.indexOf('repointSessionModelPath('),
      databaseSource.indexOf('deleteSession(', databaseSource.indexOf('repointSessionModelPath(')),
    )

    expect(block).toContain('this.db.transaction')
    expect(block).toContain('this.updateSession(id, updates)')
    expect(block).toContain('UPDATE chats')
    expect(block).toContain('WHERE model_path = ?')
    expect(block).toContain('oldModelPath')
    expect(block).not.toContain('DELETE FROM chats')
    expect(block).not.toContain('DELETE FROM messages')
  })

  it('validates before writing and requires native confirmation for a different identity', () => {
    const block = ipcSource.slice(
      ipcSource.indexOf("ipcMain.handle('sessions:repointModelPath'"),
      ipcSource.indexOf("ipcMain.handle('sessions:createRemote'"),
    )

    expect(block).toContain("properties: ['openDirectory']")
    expect(block.indexOf('validateModelBundleDirectory')).toBeLessThan(
      block.indexOf('sessionManager.repointSessionModelPath'),
    )
    expect(block).toContain('dialog.showMessageBox')
    expect(block).toContain('confirmation.response !== 1')
    expect(block).not.toContain('deleteSession')
  })

  it('exposes only explicit missing-session recovery actions in the renderer', () => {
    expect(cardSource).toContain("t('sessions.card.modelPathMissing')")
    expect(cardSource).toContain("t('sessions.card.repointModelPath')")
    expect(cardSource).toContain("t('sessions.card.removeSession')")

    const repointHandler = dashboardSource.slice(
      dashboardSource.indexOf('const handleRepoint'),
      dashboardSource.indexOf('const handleDetect'),
    )
    expect(repointHandler).toContain('window.api.sessions.repointModelPath(sessionId)')
    expect(repointHandler).not.toContain('.delete(')
  })

  it('does not auto-repoint missing paths during list or start', () => {
    const listBlock = sessionsSource.slice(
      sessionsSource.indexOf('getSessions(): Array<Session & SessionModelPathClassification>'),
      sessionsSource.indexOf('getSession(id:', sessionsSource.indexOf('getSessions(): Array<Session & SessionModelPathClassification>')),
    )
    expect(listBlock).toContain('classifySessionModelPaths(db.getSessions())')
    expect(listBlock).not.toContain('updateSession')
    expect(listBlock).not.toContain('deleteSession')

    const startMissingBlock = sessionsSource.slice(
      sessionsSource.indexOf('if (!existsSync(config.modelPath))'),
      sessionsSource.indexOf('// Block starting a session with an actively downloading model'),
    )
    // LE10 contract: Start may re-resolve by model IDENTITY when EXACTLY ONE
    // distinct valid twin path exists (live-proven ccb31dfea) — never a guess.
    expect(startMissingBlock).toContain('sessionMatchesModelPath')
    expect(startMissingBlock).toContain('resolved.length === 1')
    // Persistence stays guarded against UNIQUE collisions, and the no-twin
    // path fails with the explicit repoint hint instead of silently mutating.
    expect(startMissingBlock).toContain('pathOwnedElsewhere')
    expect(startMissingBlock).toContain('Repoint the session')
  })
})

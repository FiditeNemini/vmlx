/**
 * Coding-tool auto-configure must never destroy a user's existing config.
 *
 * Three real defects found 2026-08-17 while auditing detect/toggle/remove:
 *
 *  1. safeReadJSON collapsed "absent" and "unparseable" into null, and every
 *     addEntry does `read() || {}` before writing the whole file back — so a
 *     config we merely failed to PARSE was replaced by our single entry. This
 *     needs no corruption to hit: openclaw.json is documented JSON5, where a
 *     comment or trailing comma is legal and JSON.parse rejects it.
 *  2. Claude's addEntry overwrote ANTHROPIC_API_KEY with 'mlxstudio' and
 *     removeEntry then DELETED it — a user with a real Anthropic key lost it
 *     by toggling vMLX on and off.
 *  3. OpenClaw replaced its whole provider per add, so a second model dropped
 *     the first while the allowlist kept both, leaving an allowlist entry
 *     pointing at a model the provider no longer declared.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const electronMock = vi.hoisted(() => ({
  handlers: new Map<string, (...args: any[]) => Promise<any>>(),
}))

vi.mock('electron', () => ({
  ipcMain: {
    handle: vi.fn((channel: string, handler: (...args: any[]) => Promise<any>) => {
      electronMock.handlers.set(channel, handler)
    }),
  },
}))

function mkTempHome(): string {
  const dir = join(tmpdir(), `vmlx-ct-safety-${Date.now()}-${Math.random().toString(16).slice(2)}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

describe('coding tool config safety', () => {
  let home: string
  const oldHome = process.env.HOME
  const oldPath = process.env.PATH
  const oldFetch = global.fetch
  const BASE = 'http://127.0.0.1:8080'

  beforeEach(async () => {
    home = mkTempHome()
    process.env.HOME = home
    process.env.PATH = `${join(home, '.local', 'bin')}:${oldPath || ''}`
    global.fetch = vi.fn(async () => ({ ok: false, json: async () => ({}) })) as any
    mkdirSync(join(home, '.local', 'bin'), { recursive: true })
    for (const cmd of ['claude', 'codex', 'opencode', 'openclaw']) {
      writeFileSync(join(home, '.local', 'bin', cmd), '#!/bin/sh\nexit 0\n', { mode: 0o755 })
    }
    electronMock.handlers.clear()
    vi.resetModules()
    const mod = await import('../src/main/ipc/coding-tools')
    mod.registerCodingToolHandlers()
  })

  afterEach(() => {
    process.env.HOME = oldHome
    process.env.PATH = oldPath
    global.fetch = oldFetch
    rmSync(home, { recursive: true, force: true })
  })

  const add = () => electronMock.handlers.get('tools:addCodingToolConfig')!
  const remove = () => electronMock.handlers.get('tools:removeCodingToolConfig')!
  const status = () => electronMock.handlers.get('tools:getCodingToolStatus')!

  it('REFUSES to overwrite an unparseable config instead of clobbering it', async () => {
    // Legal JSON5, invalid JSON — exactly what an openclaw user may have.
    const path = join(home, '.openclaw', 'openclaw.json')
    mkdirSync(join(home, '.openclaw'), { recursive: true })
    const original = '{\n  // my настройки\n  "models": { "mode": "merge" },\n}\n'
    writeFileSync(path, original)

    const result = await add()({}, 'openclaw', BASE, 'my-model', 8080)

    expect(result?.success).toBe(false)
    expect(String(result?.error)).toMatch(/could not be parsed|Refusing to overwrite/i)
    // The user's bytes must still be there, untouched.
    expect(readFileSync(path, 'utf8')).toBe(original)
  })

  it('still creates a config when the file is absent or empty', async () => {
    const path = join(home, '.openclaw', 'openclaw.json')
    expect(existsSync(path)).toBe(false)
    expect((await add()({}, 'openclaw', BASE, 'model-a', 8080))?.success).toBe(true)
    expect(JSON.parse(readFileSync(path, 'utf8')).models.providers.mlxstudio).toBeTruthy()

    // An empty file is "nothing to preserve", not "unreadable".
    const claudePath = join(home, '.claude', 'settings.json')
    mkdirSync(join(home, '.claude'), { recursive: true })
    writeFileSync(claudePath, '   \n')
    expect((await add()({}, 'claude-code', BASE, 'model-a', 8080))?.success).toBe(true)
    expect(JSON.parse(readFileSync(claudePath, 'utf8')).env.ANTHROPIC_MODEL).toBe('model-a')
  })

  it("gives back the user's real Anthropic key after add -> remove", async () => {
    const path = join(home, '.claude', 'settings.json')
    mkdirSync(join(home, '.claude'), { recursive: true })
    writeFileSync(
      path,
      JSON.stringify({
        env: {
          ANTHROPIC_API_KEY: 'sk-ant-USERS-REAL-KEY',
          ANTHROPIC_MODEL: 'claude-opus-4',
          KEEP_ME: 'untouched',
        },
        otherSetting: true,
      }),
    )

    expect((await add()({}, 'claude-code', BASE, 'local-model', 8080))?.success).toBe(true)
    let cfg = JSON.parse(readFileSync(path, 'utf8'))
    expect(cfg.env.ANTHROPIC_BASE_URL).toBe(BASE)
    expect(cfg.env.ANTHROPIC_API_KEY).toBe('mlxstudio')

    // A second add (switching model) must NOT snapshot our own values.
    expect((await add()({}, 'claude-code', BASE, 'other-model', 8080))?.success).toBe(true)

    expect((await remove()({}, 'claude-code', 'local-model'))?.success).toBe(true)
    cfg = JSON.parse(readFileSync(path, 'utf8'))
    expect(cfg.env.ANTHROPIC_API_KEY).toBe('sk-ant-USERS-REAL-KEY')
    expect(cfg.env.ANTHROPIC_MODEL).toBe('claude-opus-4')
    expect(cfg.env.ANTHROPIC_BASE_URL).toBeUndefined() // user had none
    expect(cfg.env.KEEP_ME).toBe('untouched')
    expect(cfg.env._mlxstudio).toBeUndefined()
    expect(cfg.env._mlxstudio_prev).toBeUndefined()
    expect(cfg.otherSetting).toBe(true)
  })

  it('removes cleanly when the user had no Anthropic settings at all', async () => {
    const path = join(home, '.claude', 'settings.json')
    expect((await add()({}, 'claude-code', BASE, 'local-model', 8080))?.success).toBe(true)
    expect((await remove()({}, 'claude-code', 'local-model'))?.success).toBe(true)
    const cfg = JSON.parse(readFileSync(path, 'utf8'))
    // env should be gone entirely rather than left holding empty keys.
    expect(cfg.env).toBeUndefined()
  })

  it('keeps BOTH models when two are configured for OpenClaw', async () => {
    const path = join(home, '.openclaw', 'openclaw.json')
    expect((await add()({}, 'openclaw', BASE, 'model-a', 8080))?.success).toBe(true)
    expect((await add()({}, 'openclaw', BASE, 'model-b', 8080))?.success).toBe(true)

    const cfg = JSON.parse(readFileSync(path, 'utf8'))
    const ids = cfg.models.providers.mlxstudio.models.map((m: any) => m.id).sort()
    expect(ids).toEqual(['model-a', 'model-b'])

    // Every allowlist entry must name a model the provider actually declares.
    const allow = Object.keys(cfg.agents.defaults.models)
    for (const key of allow) {
      expect(ids).toContain(key.replace(/^mlxstudio\//, ''))
    }
    expect(allow.sort()).toEqual(['mlxstudio/model-a', 'mlxstudio/model-b'])
  })

  it('re-adding the same OpenClaw model updates it rather than duplicating', async () => {
    const path = join(home, '.openclaw', 'openclaw.json')
    await add()({}, 'openclaw', BASE, 'model-a', 8080)
    await add()({}, 'openclaw', BASE, 'model-a', 8080)
    const cfg = JSON.parse(readFileSync(path, 'utf8'))
    expect(cfg.models.providers.mlxstudio.models).toHaveLength(1)
  })

  it('status reporting survives one tool having a malformed config', async () => {
    // A broken openclaw file must not blank out the whole panel.
    mkdirSync(join(home, '.openclaw'), { recursive: true })
    writeFileSync(join(home, '.openclaw', 'openclaw.json'), '{ not json')
    await add()({}, 'claude-code', BASE, 'local-model', 8080)

    const result = await status()({})
    expect(result).toBeTruthy()
    const text = JSON.stringify(result)
    expect(text).toContain('claude-code')
  })
})

describe('hermes agent config', () => {
  let home: string
  const oldHome = process.env.HOME
  const oldPath = process.env.PATH
  const oldFetch = global.fetch
  const BASE = 'http://127.0.0.1:8080'

  beforeEach(async () => {
    home = mkTempHome()
    process.env.HOME = home
    process.env.PATH = `${join(home, '.local', 'bin')}:${oldPath || ''}`
    global.fetch = vi.fn(async () => ({ ok: false, json: async () => ({}) })) as any
    mkdirSync(join(home, '.local', 'bin'), { recursive: true })
    writeFileSync(join(home, '.local', 'bin', 'hermes'), '#!/bin/sh\nexit 0\n', { mode: 0o755 })
    electronMock.handlers.clear()
    vi.resetModules()
    const mod = await import('../src/main/ipc/coding-tools')
    mod.registerCodingToolHandlers()
  })

  afterEach(() => {
    process.env.HOME = oldHome
    process.env.PATH = oldPath
    global.fetch = oldFetch
    rmSync(home, { recursive: true, force: true })
  })

  const add = () => electronMock.handlers.get('tools:addCodingToolConfig')!
  const remove = () => electronMock.handlers.get('tools:removeCodingToolConfig')!
  const yamlPath = () => join(home, '.hermes', 'config.yaml')
  const readYaml = async () => {
    const { load } = await import('js-yaml')
    return load(readFileSync(yamlPath(), 'utf8')) as any
  }

  it('writes a providers MAP with base_url ending at /v1', async () => {
    expect((await add()({}, 'hermes', BASE, 'my-model', 8080))?.success).toBe(true)
    const doc = await readYaml()
    const p = doc.providers.mlxstudio
    // Hermes appends /chat/completions itself — a URL past /v1 would 404.
    expect(p.base_url).toBe(`${BASE}/v1`)
    expect(p.base_url.endsWith('/v1')).toBe(true)
    expect(Object.keys(p.models)).toEqual(['my-model'])
    expect(p._mlxstudio).toBe(true)
  })

  it("preserves the user's unrelated hermes settings", async () => {
    mkdirSync(join(home, '.hermes'), { recursive: true })
    writeFileSync(
      yamlPath(),
      'auxiliary:\n  compression:\n    model: glm-4.7\nproviders:\n  theirs:\n    base_url: https://api.z.ai/v4\n',
    )
    expect((await add()({}, 'hermes', BASE, 'my-model', 8080))?.success).toBe(true)
    const doc = await readYaml()
    expect(doc.auxiliary.compression.model).toBe('glm-4.7')
    expect(doc.providers.theirs.base_url).toBe('https://api.z.ai/v4')
    expect(doc.providers.mlxstudio).toBeTruthy()
  })

  it('keeps both models, and removing one leaves the other', async () => {
    await add()({}, 'hermes', BASE, 'model-a', 8080)
    await add()({}, 'hermes', BASE, 'model-b', 8080)
    expect(Object.keys((await readYaml()).providers.mlxstudio.models).sort()).toEqual([
      'model-a', 'model-b',
    ])

    expect((await remove()({}, 'hermes', 'model-a'))?.success).toBe(true)
    expect(Object.keys((await readYaml()).providers.mlxstudio.models)).toEqual(['model-b'])

    // Removing the last model drops the provider rather than leaving a shell.
    expect((await remove()({}, 'hermes', 'model-b'))?.success).toBe(true)
    const doc = await readYaml()
    expect(doc?.providers?.mlxstudio).toBeUndefined()
  })

  it('REFUSES to overwrite malformed hermes yaml', async () => {
    mkdirSync(join(home, '.hermes'), { recursive: true })
    const bad = 'providers:\n  - this is a list not a map\n\tbad_tab: 1\n'
    writeFileSync(yamlPath(), bad)
    const result = await add()({}, 'hermes', BASE, 'my-model', 8080)
    expect(result?.success).toBe(false)
    expect(readFileSync(yamlPath(), 'utf8')).toBe(bad)
  })
})

// MLX Studio — Coding Tool Integration IPC
// Non-destructive config management for Claude Code, Codex CLI, OpenCode,
// OpenClaw, and Hermes Agent
import { ipcMain } from 'electron'
import { execFileSync } from 'child_process'
import { homedir } from 'os'
import { join } from 'path'
import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { dump as dumpYAML, load as loadYAML } from 'js-yaml'

const MLXSTUDIO_TAG = '_mlxstudio'  // Tag to identify our entries

interface ToolConfig {
  detect: () => boolean
  installCmd: string
  installArgs: string[]
  configPath: string
  getEntries: () => Array<{ label: string; baseUrl: string }>
  addEntry: (baseUrl: string, modelName: string, port: number | null, limits: CodingToolModelLimits) => void
  removeEntry: (label: string) => void
}

interface CodingToolModelLimits {
  context: number
  output: number
}

const FALLBACK_CODING_TOOL_LIMITS: CodingToolModelLimits = {
  context: 32768,
  output: 4096,
}

function normalizePositiveInt(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return undefined
  return Math.round(value)
}

async function getCodingToolModelLimits(baseUrl: string, modelName: string): Promise<CodingToolModelLimits> {
  const limits: CodingToolModelLimits = { ...FALLBACK_CODING_TOOL_LIMITS }
  try {
    const health = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(1500) })
    if (health.ok) {
      const body = await health.json()
      limits.context = normalizePositiveInt(body?.max_prompt_tokens) ?? limits.context
    }
  } catch {}

  try {
    const encoded = encodeURIComponent(modelName)
    const caps = await fetch(`${baseUrl}/v1/models/${encoded}/capabilities`, {
      signal: AbortSignal.timeout(1500),
    })
    if (caps.ok) {
      const body = await caps.json()
      limits.output =
        normalizePositiveInt(body?.sampling_defaults?.max_new_tokens) ??
        normalizePositiveInt(body?.sampling_defaults?.max_tokens) ??
        limits.output
    }
  } catch {}

  return limits
}

/**
 * Thrown when a user's config file exists but cannot be parsed.
 *
 * This distinction is load-bearing. `safeReadJSON` used to collapse "absent"
 * and "unparseable" into the same `null`, and every addEntry does
 * `safeReadJSON(path) || {}` before `safeWriteJSON` — so an existing config we
 * merely failed to PARSE was overwritten by our single entry. That destroys
 * real user data, and it is reachable without any corruption: openclaw.json is
 * documented as JSON5, so comments or a trailing comma are perfectly legal
 * there and JSON.parse rejects them.
 *
 * Absent is safe to create. Unparseable must REFUSE and say so.
 */
export class UnreadableConfigError extends Error {
  constructor(
    readonly path: string,
    readonly cause: unknown,
  ) {
    super(
      `Config file exists but could not be parsed: ${path} — ` +
        `${cause instanceof Error ? cause.message : String(cause)}. ` +
        `Refusing to overwrite it. Fix or move the file, then retry.`,
    )
    this.name = 'UnreadableConfigError'
  }
}

/** Read a JSON config. Returns null ONLY when the file does not exist. */
function safeReadJSON(path: string): any {
  if (!existsSync(path)) return null
  let raw: string
  try {
    raw = readFileSync(path, 'utf-8')
  } catch (err) {
    throw new UnreadableConfigError(path, err)
  }
  // An empty/whitespace-only file is treated as absent: nothing to preserve,
  // and JSON.parse would reject it.
  if (!raw.trim()) return null
  try {
    return JSON.parse(raw)
  } catch (err) {
    throw new UnreadableConfigError(path, err)
  }
}

/**
 * Read a JSON config for DISPLAY only — never for a read-modify-write.
 *
 * getEntries must not explode the settings panel just because one unrelated
 * tool has a malformed file, so it degrades to "no entries". Writers must use
 * safeReadJSON so they refuse instead.
 */
function readJSONForDisplay(path: string): any {
  try {
    return safeReadJSON(path)
  } catch {
    return null
  }
}

function safeReadTOML(path: string): string | null {
  try {
    if (!existsSync(path)) return null
    return readFileSync(path, 'utf-8')
  } catch { return null }
}

function safeWriteTOML(path: string, content: string): void {
  if (existsSync(path)) {
    try { copyFileSync(path, path + '.bak') } catch {}
  }
  const dir = path.substring(0, path.lastIndexOf('/'))
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(path, content, { encoding: 'utf-8', mode: 0o600 })
  try { chmodSync(path, 0o600) } catch {}
}

function safeWriteJSON(path: string, data: any): void {
  // Backup before writing
  if (existsSync(path)) {
    try { copyFileSync(path, path + '.bak') } catch {}
  }
  const dir = path.substring(0, path.lastIndexOf('/'))
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(path, JSON.stringify(data, null, 2) + '\n', { encoding: 'utf-8', mode: 0o600 })
  try { chmodSync(path, 0o600) } catch {}
}

function commandExists(cmd: string): boolean {
  // Check common install locations (Electron strips user PATH)
  const paths = [
    join(homedir(), '.local', 'bin', cmd),
    join(homedir(), '.npm-global', 'bin', cmd),
    '/usr/local/bin/' + cmd,
    '/usr/bin/' + cmd,
    '/opt/homebrew/bin/' + cmd,
    join(homedir(), '.cargo', 'bin', cmd),
    join(homedir(), '.bun', 'bin', cmd),
    join(homedir(), '.volta', 'bin', cmd),
  ]
  if (paths.some(p => existsSync(p))) return true
  // Fallback: try which with augmented PATH (includes nvm, pnpm, yarn)
  try {
    const extraPaths = [
      `${homedir()}/.local/bin`,
      '/opt/homebrew/bin',
      '/usr/local/bin',
      `${homedir()}/.cargo/bin`,
      `${homedir()}/.bun/bin`,
      `${homedir()}/.volta/bin`,
      `${homedir()}/.yarn/bin`,
    ].join(':')
    const env = { ...process.env, PATH: `${process.env.PATH}:${extraPaths}` }
    execFileSync('which', [cmd], { stdio: 'pipe', env })
    return true
  } catch { return false }
}

// ═══ Claude Code ═══
// Config: ~/.claude/settings.json — env vars (ANTHROPIC_BASE_URL, ANTHROPIC_MODEL)
// Claude Code requires Anthropic Messages API format (/v1/messages) — vMLX supports this
const CLAUDE_SETTINGS = join(homedir(), '.claude', 'settings.json')
/** env keys we overwrite, and therefore must be able to hand back. */
const CLAUDE_MANAGED_ENV_KEYS = [
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_MODEL',
  'ANTHROPIC_API_KEY',
] as const
/** Where the user's pre-vMLX values are parked so remove can restore them. */
const CLAUDE_PREV_KEY = `${MLXSTUDIO_TAG}_prev`
const claudeCode: ToolConfig = {
  detect: () => commandExists('claude'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@anthropic-ai/claude-code'],
  configPath: CLAUDE_SETTINGS,
  getEntries: () => {
    const cfg = readJSONForDisplay(CLAUDE_SETTINGS)
    if (!cfg?.env?.ANTHROPIC_BASE_URL || !cfg?.env?.[MLXSTUDIO_TAG]) return []
    return [{ label: cfg.env.ANTHROPIC_MODEL || 'default', baseUrl: cfg.env.ANTHROPIC_BASE_URL }]
  },
  addEntry: (baseUrl, modelName) => {
    const cfg = safeReadJSON(CLAUDE_SETTINGS) || {}
    if (!cfg.env) cfg.env = {}

    // Snapshot whatever the user already had, ONCE, before we clobber it.
    // Without this, a user with a real ANTHROPIC_API_KEY (or their own
    // BASE_URL/MODEL) lost it permanently: addEntry overwrote the key with
    // 'mlxstudio' and removeEntry then DELETED it. Toggling vMLX on and off
    // must leave the user exactly where they started.
    //
    // Guarded by absence so a second add (e.g. switching model) does not
    // snapshot our OWN values over the user's originals.
    if (cfg.env[CLAUDE_PREV_KEY] === undefined) {
      const prev: Record<string, string | null> = {}
      for (const key of CLAUDE_MANAGED_ENV_KEYS) {
        prev[key] = Object.prototype.hasOwnProperty.call(cfg.env, key)
          ? cfg.env[key]
          : null // null = "the user did not have this key at all"
      }
      cfg.env[CLAUDE_PREV_KEY] = prev
    }

    // Claude Code appends /v1/messages itself — base URL should NOT include /v1
    cfg.env.ANTHROPIC_BASE_URL = baseUrl
    cfg.env.ANTHROPIC_MODEL = modelName
    cfg.env.ANTHROPIC_API_KEY = 'mlxstudio'  // Required non-empty value
    cfg.env[MLXSTUDIO_TAG] = 'true'
    safeWriteJSON(CLAUDE_SETTINGS, cfg)
  },
  removeEntry: () => {
    const cfg = safeReadJSON(CLAUDE_SETTINGS)
    if (!cfg?.env) return

    const prev = cfg.env[CLAUDE_PREV_KEY]
    if (prev && typeof prev === 'object') {
      // Restore the user's originals rather than deleting blindly.
      for (const key of CLAUDE_MANAGED_ENV_KEYS) {
        const value = prev[key]
        if (value === null || value === undefined) delete cfg.env[key]
        else cfg.env[key] = value
      }
      delete cfg.env[CLAUDE_PREV_KEY]
    } else {
      // No snapshot (config predates this fix): fall back to the old
      // delete-only behaviour, which is still correct when the user had
      // nothing of their own here.
      for (const key of CLAUDE_MANAGED_ENV_KEYS) delete cfg.env[key]
    }

    delete cfg.env[MLXSTUDIO_TAG]
    // Clean up empty env object
    if (Object.keys(cfg.env).length === 0) delete cfg.env
    safeWriteJSON(CLAUDE_SETTINGS, cfg)
  },
}

// ═══ Codex CLI ═══
// Config: ~/.codex/config.toml — TOML format with [model_providers.NAME] sections
const CODEX_TOML = join(homedir(), '.codex', 'config.toml')
const codexCli: ToolConfig = {
  detect: () => commandExists('codex'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@openai/codex'],
  configPath: CODEX_TOML,
  getEntries: () => {
    const toml = safeReadTOML(CODEX_TOML)
    if (!toml) return []
    // Parse TOML: find [model_providers.MLXSTUDIO_*] sections with _mlxstudio marker
    const entries: Array<{ label: string; baseUrl: string }> = []
    const sectionRegex = /\[model_providers\.([^\]]+)\]/g
    let match
    while ((match = sectionRegex.exec(toml)) !== null) {
      const name = match[1]
      if (!name.startsWith('MLXSTUDIO_')) continue
      // Extract base_url from this section
      const sectionStart = match.index + match[0].length
      const nextSection = toml.indexOf('\n[', sectionStart)
      const sectionBody = nextSection >= 0 ? toml.slice(sectionStart, nextSection) : toml.slice(sectionStart)
      const urlMatch = sectionBody.match(/base_url\s*=\s*"([^"]+)"/)
      entries.push({ label: name, baseUrl: urlMatch ? urlMatch[1] : '' })
    }
    return entries
  },
  addEntry: (baseUrl, modelName, _port, limits) => {
    let toml = safeReadTOML(CODEX_TOML) || ''
    const providerKey = `MLXSTUDIO_${modelName.replace(/[^a-zA-Z0-9_]/g, '_').toUpperCase()}`
    // Remove existing section if present
    const sectionPattern = new RegExp(`\\[model_providers\\.${providerKey}\\][\\s\\S]*?(?=\\n\\[|$)`, 'g')
    toml = toml.replace(sectionPattern, '').replace(/\n{3,}/g, '\n\n').trim()
    // Append new section
    const section = `\n\n[model_providers.${providerKey}]\nname = "MLX Studio (${modelName})"\nbase_url = "${baseUrl}/v1"\nwire_api = "responses"\nmax_context = ${limits.context}\n`
    toml += section
    safeWriteTOML(CODEX_TOML, toml)
  },
  removeEntry: (label) => {
    let toml = safeReadTOML(CODEX_TOML)
    if (!toml) return
    // Remove the [model_providers.LABEL] section
    const sectionPattern = new RegExp(`\\[model_providers\\.${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\][\\s\\S]*?(?=\\n\\[|$)`, 'g')
    toml = toml.replace(sectionPattern, '').replace(/\n{3,}/g, '\n\n').trim()
    safeWriteTOML(CODEX_TOML, toml + '\n')
  },
}

// ═══ OpenCode ═══
// Config: ~/.config/opencode/opencode.json — we add provider entries tagged with MLXSTUDIO_TAG
const openCode: ToolConfig = {
  detect: () => commandExists('opencode'),
  installCmd: 'npm',
  installArgs: ['install', '-g', 'opencode'],
  configPath: join(homedir(), '.config', 'opencode', 'opencode.json'),
  getEntries: () => {
    const cfg = readJSONForDisplay(join(homedir(), '.config', 'opencode', 'opencode.json'))
    if (!cfg?.provider) return []
    return Object.entries(cfg.provider)
      .filter(([_, v]: any) => v?.[MLXSTUDIO_TAG])
      .map(([k, v]: any) => ({ label: k, baseUrl: (v as any)?.options?.baseURL || '' }))
  },
  addEntry: (baseUrl, modelName, _port, limits) => {
    const path = join(homedir(), '.config', 'opencode', 'opencode.json')
    const cfg = safeReadJSON(path) || { '$schema': 'https://opencode.ai/config.json' }
    if (!cfg.provider) cfg.provider = {}
    const key = `mlxstudio-${modelName.replace(/[^a-zA-Z0-9-]/g, '-')}`
    cfg.provider[key] = {
      npm: '@ai-sdk/openai-compatible',
      name: `MLX Studio (${modelName})`,
      options: { baseURL: `${baseUrl}/v1` },
      models: {
        [modelName]: {
          name: modelName,
          limit: { context: limits.context, output: limits.output },
          modalities: { input: ['text'], output: ['text'] },
        },
      },
      [MLXSTUDIO_TAG]: true,
    }
    safeWriteJSON(path, cfg)
  },
  removeEntry: (label) => {
    const path = join(homedir(), '.config', 'opencode', 'opencode.json')
    const cfg = safeReadJSON(path)
    if (!cfg?.provider?.[label]) return
    delete cfg.provider[label]
    safeWriteJSON(path, cfg)
  },
}

// ═══ OpenClaw ═══
// Config: ~/.openclaw/openclaw.json (JSON5) — models.providers + agents.defaults.models allowlist
// OpenClaw uses OpenAI-compatible API via "openai-completions" wire format
const OPENCLAW_JSON = join(homedir(), '.openclaw', 'openclaw.json')
const openClaw: ToolConfig = {
  detect: () => commandExists('openclaw'),
  installCmd: 'npm',
  installArgs: ['install', '-g', 'openclaw@latest'],
  configPath: OPENCLAW_JSON,
  getEntries: () => {
    const cfg = readJSONForDisplay(OPENCLAW_JSON)
    if (!cfg?.models?.providers) return []
    const entries: Array<{ label: string; baseUrl: string }> = []
    for (const [name, provider] of Object.entries(cfg.models.providers)) {
      const p = provider as any
      if (!p?.[MLXSTUDIO_TAG]) continue
      entries.push({ label: name, baseUrl: p.baseUrl || '' })
    }
    return entries
  },
  addEntry: (baseUrl, modelName, _port, limits) => {
    const cfg = safeReadJSON(OPENCLAW_JSON) || {}
    if (!cfg.models) cfg.models = {}
    if (!cfg.models.mode) cfg.models.mode = 'merge'
    if (!cfg.models.providers) cfg.models.providers = {}
    if (!cfg.agents) cfg.agents = {}
    if (!cfg.agents.defaults) cfg.agents.defaults = {}
    if (!cfg.agents.defaults.models) cfg.agents.defaults.models = {}
    const providerKey = `mlxstudio`
    const fqModel = `${providerKey}/${modelName}`
    const model = {
      id: modelName,
      name: modelName,
      reasoning: false,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: limits.context,
      maxTokens: limits.output,
    }

    // OpenClaw keeps ONE provider holding a LIST of models, so configuring a
    // second model must APPEND to that list. Replacing the whole provider
    // (the previous behaviour) silently dropped model A when model B was
    // added, while the allowlist below kept BOTH — leaving an allowlist entry
    // pointing at a model the provider no longer declared.
    const existing = cfg.models.providers[providerKey]
    const models: any[] = Array.isArray(existing?.models) ? [...existing.models] : []
    const at = models.findIndex(m => m?.id === modelName)
    if (at >= 0) models[at] = model
    else models.push(model)

    cfg.models.providers[providerKey] = {
      baseUrl: `${baseUrl}/v1`,
      apiKey: 'mlxstudio',
      api: 'openai-completions',
      models,
      [MLXSTUDIO_TAG]: true,
    }
    cfg.agents.defaults.models[fqModel] = { alias: modelName }
    safeWriteJSON(OPENCLAW_JSON, cfg)
  },
  removeEntry: () => {
    const cfg = safeReadJSON(OPENCLAW_JSON)
    if (!cfg) return
    if (cfg.models?.providers?.mlxstudio) delete cfg.models.providers.mlxstudio
    // Remove all mlxstudio/ entries from allowlist
    if (cfg.agents?.defaults?.models) {
      for (const key of Object.keys(cfg.agents.defaults.models)) {
        if (key.startsWith('mlxstudio/')) delete cfg.agents.defaults.models[key]
      }
    }
    safeWriteJSON(OPENCLAW_JSON, cfg)
  },
}

// ═══ Hermes Agent (NousResearch) ═══
// Config: ~/.hermes/config.yaml — a `providers` MAP (not a list). Verified
// against the official docs, not guessed:
//   providers:
//     <id>:
//       base_url: "https://host/v1"
//       api_key: "..."
//       models: { <model>: { timeout_seconds: N } }
// Hermes appends /chat/completions itself, so base_url must END at /v1 — a URL
// that already includes the path (or a trailing slash) 404s. Setting base_url
// makes Hermes call that endpoint directly instead of a built-in provider.
//
// Parsed with a real YAML library rather than regex: this file belongs to the
// user and may hold arbitrary unrelated settings, and the regex-edited TOML
// path in this same module is exactly the fragility worth not repeating.
const HERMES_YAML = join(homedir(), '.hermes', 'config.yaml')
const HERMES_PROVIDER_KEY = 'mlxstudio'

function readHermesConfig(path: string, forDisplay: boolean): any {
  if (!existsSync(path)) return null
  let raw: string
  try {
    raw = readFileSync(path, 'utf-8')
  } catch (err) {
    if (forDisplay) return null
    throw new UnreadableConfigError(path, err)
  }
  if (!raw.trim()) return null
  try {
    const doc = loadYAML(raw)
    // A YAML scalar/list at the root is not a config we can safely merge into.
    if (doc === null || doc === undefined) return null
    if (typeof doc !== 'object' || Array.isArray(doc)) {
      throw new Error('expected a YAML mapping at the document root')
    }
    return doc
  } catch (err) {
    if (forDisplay) return null
    throw new UnreadableConfigError(path, err)
  }
}

function writeHermesConfig(path: string, doc: any): void {
  if (existsSync(path)) {
    try { copyFileSync(path, path + '.bak') } catch {}
  }
  const dir = path.substring(0, path.lastIndexOf('/'))
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(path, dumpYAML(doc, { lineWidth: 120, noRefs: true }), {
    encoding: 'utf-8',
    mode: 0o600,
  })
  try { chmodSync(path, 0o600) } catch {}
}

const hermesAgent: ToolConfig = {
  detect: () => commandExists('hermes'),
  installCmd: 'npm',
  installArgs: ['install', '-g', '@nousresearch/hermes-agent'],
  configPath: HERMES_YAML,
  getEntries: () => {
    const doc = readHermesConfig(HERMES_YAML, true)
    const provider = doc?.providers?.[HERMES_PROVIDER_KEY]
    if (!provider?.[MLXSTUDIO_TAG]) return []
    const models = provider.models && typeof provider.models === 'object'
      ? Object.keys(provider.models)
      : []
    const baseUrl = typeof provider.base_url === 'string' ? provider.base_url : ''
    return models.length
      ? models.map(m => ({ label: m, baseUrl }))
      : [{ label: HERMES_PROVIDER_KEY, baseUrl }]
  },
  addEntry: (baseUrl, modelName) => {
    const doc = readHermesConfig(HERMES_YAML, false) || {}
    if (!doc.providers || typeof doc.providers !== 'object' || Array.isArray(doc.providers)) {
      doc.providers = {}
    }
    const existing = doc.providers[HERMES_PROVIDER_KEY]
    // Keep sibling models: like OpenClaw, one provider holds many models, so
    // adding a second must not silently drop the first.
    const models =
      existing?.models && typeof existing.models === 'object' && !Array.isArray(existing.models)
        ? { ...existing.models }
        : {}
    models[modelName] = { ...(models[modelName] || {}) }
    doc.providers[HERMES_PROVIDER_KEY] = {
      ...(existing && typeof existing === 'object' ? existing : {}),
      // Hermes appends /chat/completions — end at /v1.
      base_url: `${baseUrl}/v1`,
      api_key: 'mlxstudio',
      models,
      [MLXSTUDIO_TAG]: true,
    }
    writeHermesConfig(HERMES_YAML, doc)
  },
  removeEntry: (label) => {
    const doc = readHermesConfig(HERMES_YAML, false)
    const provider = doc?.providers?.[HERMES_PROVIDER_KEY]
    if (!provider) return
    // Drop just this model when others remain; drop the provider when it was
    // the last one, so no empty shell is left referencing our endpoint.
    if (provider.models && typeof provider.models === 'object' && label in provider.models) {
      delete provider.models[label]
    }
    if (!provider.models || Object.keys(provider.models).length === 0) {
      delete doc.providers[HERMES_PROVIDER_KEY]
      if (Object.keys(doc.providers).length === 0) delete doc.providers
    }
    writeHermesConfig(HERMES_YAML, doc)
  },
}

const TOOLS: Record<string, ToolConfig> = {
  'claude-code': claudeCode,
  'codex': codexCli,
  'hermes': hermesAgent,
  'opencode': openCode,
  'openclaw': openClaw,
}

let registered = false

export function registerCodingToolHandlers(): void {
  if (registered) return
  registered = true

  ipcMain.handle('tools:getCodingToolStatus', async () => {
    const result: Record<string, any> = {}
    for (const [id, tool] of Object.entries(TOOLS)) {
      const installed = tool.detect()
      const entries = installed ? tool.getEntries() : []
      result[id] = {
        installed,
        configured: entries.length > 0,
        configPath: tool.configPath,
        entries,
      }
    }
    return result
  })

  ipcMain.handle('tools:installCodingTool', async (_, toolId: string) => {
    const tool = TOOLS[toolId]
    if (!tool) return { success: false, error: 'Unknown tool' }
    try {
      execFileSync(tool.installCmd, tool.installArgs, { stdio: 'pipe', timeout: 120000 })
      return { success: true }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  })

  ipcMain.handle('tools:addCodingToolConfig', async (_, toolId: string, baseUrl: string, modelName: string, port: number | null) => {
    const tool = TOOLS[toolId]
    if (!tool) return { success: false, error: 'Unknown tool' }
    if (!tool.detect()) return { success: false, error: 'Tool not installed' }
    try {
      const limits = await getCodingToolModelLimits(baseUrl, modelName)
      tool.addEntry(baseUrl, modelName, port, limits)
      return { success: true }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  })

  ipcMain.handle('tools:removeCodingToolConfig', async (_, toolId: string, label: string) => {
    const tool = TOOLS[toolId]
    if (!tool) return { success: false, error: 'Unknown tool' }
    try {
      tool.removeEntry(label)
      return { success: true }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  })

  // Returns tailored config snippets for manual setup instructions
  ipcMain.handle('tools:getConfigSnippets', async (_, baseUrl: string, modelName: string) => {
    const home = homedir()
    const limits = await getCodingToolModelLimits(baseUrl, modelName)
    return {
      'claude-code': {
        filePath: `${home}/.claude/settings.json`,
        language: 'json',
        snippet: JSON.stringify({
          env: {
            ANTHROPIC_BASE_URL: baseUrl,
            ANTHROPIC_MODEL: modelName,
            ANTHROPIC_API_KEY: 'mlxstudio',
          }
        }, null, 2),
        notes: 'Claude Code appends /v1/messages automatically. Merge the "env" key into your existing settings.json — do not replace the whole file. Verify: claude --version && claude /status',
      },
      'codex': {
        filePath: `${home}/.codex/config.toml`,
        language: 'toml',
        snippet: `[model_providers.MLXSTUDIO_${modelName.replace(/[^a-zA-Z0-9_]/g, '_').toUpperCase()}]\nname = "MLX Studio (${modelName})"\nbase_url = "${baseUrl}/v1"\nwire_api = "responses"\nmax_context = ${limits.context}`,
        notes: 'Append this section to the end of your config.toml. If the file doesn\'t exist, create ~/.codex/config.toml. Then run: codex --provider MLXSTUDIO_... Verify: codex --version',
      },
      'opencode': {
        filePath: `${home}/.config/opencode/opencode.json`,
        language: 'json',
        snippet: JSON.stringify({
          provider: {
            [`mlxstudio-${modelName.replace(/[^a-zA-Z0-9-]/g, '-')}`]: {
              npm: '@ai-sdk/openai-compatible',
              name: `MLX Studio (${modelName})`,
              options: { baseURL: `${baseUrl}/v1` },
              models: {
                [modelName]: {
                  name: modelName,
                  limit: { context: limits.context, output: limits.output },
                  modalities: { input: ['text'], output: ['text'] },
                },
              },
            },
          },
        }, null, 2),
        notes: 'Merge the "provider" key into your existing opencode.json. If the file doesn\'t exist, create it with { "$schema": "https://opencode.ai/config.json", "provider": { ... } }. Verify: opencode --version',
      },
      'openclaw': {
        filePath: `${home}/.openclaw/openclaw.json`,
        language: 'json',
        snippet: JSON.stringify({
          models: {
            mode: 'merge',
            providers: {
              mlxstudio: {
                baseUrl: `${baseUrl}/v1`,
                apiKey: 'mlxstudio',
                api: 'openai-completions',
                models: [{
                  id: modelName,
                  name: modelName,
                  reasoning: false,
                  input: ['text'],
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                  contextWindow: limits.context,
                  maxTokens: limits.output,
                }],
              },
            },
          },
          agents: {
            defaults: {
              models: {
                [`mlxstudio/${modelName}`]: { alias: modelName },
              },
            },
          },
        }, null, 2),
        notes: 'Merge into your existing openclaw.json (JSON5 format). "mode": "merge" ensures your existing providers are kept. Both the provider AND the agents.defaults.models allowlist entry are required. Verify: openclaw --version && openclaw doctor',
      },
    }
  })
}

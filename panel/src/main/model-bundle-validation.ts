import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

export type JangBundleMetadataValidation = {
  ok: boolean
  schema: 'legacy' | 'capabilities' | 'chat' | 'invalid'
  family?: string
  error?: string
}

function readJsonObject(path: string, label: string): Record<string, unknown> {
  let parsed: unknown
  try {
    parsed = JSON.parse(readFileSync(path, 'utf8'))
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${(error as Error).message}`)
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must contain a JSON object`)
  }
  return parsed as Record<string, unknown>
}

/**
 * Mirror the Python runtime's legacy-vs-authoritative JANG family contract.
 *
 * A pre-capabilities bundle may still resolve its family from config.json.
 * Once capabilities or the chat schema is present, its family is authoritative
 * and must be complete before Electron is allowed to spawn the engine.
 */
export function validateJangBundleMetadataForLaunch(
  modelPath: string,
): JangBundleMetadataValidation {
  try {
    const configPath = join(modelPath, 'config.json')
    if (existsSync(configPath)) {
      readJsonObject(configPath, configPath)
    }

    const jangPath = join(modelPath, 'jang_config.json')
    if (!existsSync(jangPath)) {
      return { ok: true, schema: 'legacy' }
    }
    const jang = readJsonObject(jangPath, jangPath)
    const capabilities = jang.capabilities
    if (capabilities && typeof capabilities === 'object' && !Array.isArray(capabilities)) {
      const family = (capabilities as Record<string, unknown>).family
      if (typeof family !== 'string' || !family.trim()) {
        return {
          ok: false,
          schema: 'invalid',
          error: `${jangPath} capabilities.family must be a non-empty string`,
        }
      }
      return { ok: true, schema: 'capabilities', family: family.trim() }
    }

    const chat = jang.chat
    if (chat && typeof chat === 'object' && !Array.isArray(chat)) {
      const family = jang.model_family
      if (typeof family !== 'string' || !family.trim()) {
        return {
          ok: false,
          schema: 'invalid',
          error: `${jangPath} chat-authoritative metadata requires a non-empty model_family`,
        }
      }
      return { ok: true, schema: 'chat', family: family.trim() }
    }

    return { ok: true, schema: 'legacy' }
  } catch (error) {
    return {
      ok: false,
      schema: 'invalid',
      error: (error as Error).message,
    }
  }
}

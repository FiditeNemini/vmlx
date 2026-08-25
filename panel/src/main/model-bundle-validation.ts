import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

export type JangBundleMetadataValidation = {
  ok: boolean
  schema: 'legacy' | 'config-fallback' | 'capabilities' | 'chat' | 'invalid'
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
 * Complete capability/chat identities are authoritative. Older sidecars may
 * carry those blocks without the later family field; in that case the Python
 * runtime intentionally ignores the incomplete block and resolves the exact
 * config.json model_type/text_model_type. Malformed JSON and non-string family
 * values remain invalid before Electron is allowed to spawn the engine.
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
      const familyValue = (capabilities as Record<string, unknown>).family
      if (familyValue != null && typeof familyValue !== 'string') {
        throw new Error(`${jangPath} capabilities.family must be a string when present`)
      }
      const topFamilyValue = jang.model_family
      if (topFamilyValue != null && typeof topFamilyValue !== 'string') {
        throw new Error(`${jangPath} model_family must be a string when present`)
      }
      const family = typeof familyValue === 'string' && familyValue.trim()
        ? familyValue.trim()
        : typeof topFamilyValue === 'string' && topFamilyValue.trim()
          ? topFamilyValue.trim()
          : undefined
      return family
        ? { ok: true, schema: 'capabilities', family }
        : { ok: true, schema: 'config-fallback' }
    }

    const chat = jang.chat
    if (chat && typeof chat === 'object' && !Array.isArray(chat)) {
      const familyValue = jang.model_family
      if (familyValue != null && typeof familyValue !== 'string') {
        throw new Error(`${jangPath} model_family must be a string when present`)
      }
      const family = typeof familyValue === 'string' ? familyValue.trim() : ''
      return family
        ? { ok: true, schema: 'chat', family }
        : { ok: true, schema: 'config-fallback' }
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

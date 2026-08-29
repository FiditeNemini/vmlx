import { canonicalizeReasoningParserForCli } from './reasoningParserAliases'

export const GENERATION_STARTUP_DEFAULTS_VERSION = 4
export const MODEL_PARSER_DEFAULTS_VERSION = 3
export const LEGACY_GENERIC_MAX_OUTPUT_TOKENS = new Set([4096, 12000, 12068, 32768])

/**
 * Lift the historical per-token streaming default exactly once during the
 * database migration that owns old persisted rows. Current UI saves must not
 * call this helper: after migration, streamInterval=1 is an explicit supported
 * user choice and the launcher must preserve it verbatim.
 */
export function migrateLegacyStreamIntervalDefault(config: Record<string, any>): boolean {
  if (Number(config.streamInterval) !== 1) return false
  config.streamInterval = 8
  return true
}

export function migrateModelParserDefaults(
  config: Record<string, any>,
  detectedFamily?: string,
  detectedReasoningParser?: string,
): boolean {
  if (Number(config.modelParserDefaultsVersion || 0) >= MODEL_PARSER_DEFAULTS_VERSION) {
    return false
  }

  // The original Electron Laguna row persisted qwen even though the Python
  // registry and the bundle's Poolside template require the GLM-style
  // <arg_key>/<arg_value> parser. Migrate that one known auto-derived value
  // once; after the version marker is written, explicit user choices survive.
  if (detectedFamily === 'laguna' && config.toolCallParser === 'qwen') {
    config.toolCallParser = 'glm47'
  }
  // The original static Laguna fallback used qwen3. Current S2.1 bundles
  // stamp Poolside/deepseek_r1, and qwen3 cannot parse that rail correctly.
  // Migrate the old auto-derived value once when current bundle detection
  // proves the replacement. After the v2 marker is written, a later explicit
  // user choice remains untouched.
  if (
    detectedFamily === 'laguna' &&
    config.reasoningParser === 'qwen3' &&
    canonicalizeReasoningParserForCli(detectedReasoningParser) === 'deepseek_r1'
  ) {
    config.reasoningParser = 'deepseek_r1'
  }
  // v3: Qwen4Exp bundles were converter-stamped tool_parser="hermes", and
  // sessions created in that era persisted "hermes" as if it were an
  // explicit user override, so the registry/panel stale-stamp
  // neutralization never reached them. Hermes parses only JSON bodies while
  // the bundle template emits Qwen <function=/<parameter= XML — live-proven
  // sampled required-mode tool_calls_required 400s on Flash-Next JANG_1L.
  // Migrate exactly this one auto-derived value, once; every other family
  // and every non-hermes choice (qwen/auto/''/None/other) is untouched.
  if (detectedFamily === 'qwen4-exp' && config.toolCallParser === 'hermes') {
    config.toolCallParser = 'qwen'
  }
  config.modelParserDefaultsVersion = MODEL_PARSER_DEFAULTS_VERSION
  return true
}

function isMiniMaxSessionModel(modelPath?: string): boolean {
  const lower = String(modelPath || '').toLowerCase()
  return lower.includes('minimax-m2') || lower.includes('minimax_m2') || lower.includes('/minimax')
}

export function migrateLegacySessionStartupConfig(config: Record<string, any>, modelPath?: string): boolean {
  let changed = false
  if (LEGACY_GENERIC_MAX_OUTPUT_TOKENS.has(Number(config.maxTokens))) {
    config.maxTokens = 0
    config.generationStartupDefaultsVersion = GENERATION_STARTUP_DEFAULTS_VERSION
    changed = true
  }
  if (
    config.reasoningParser === 'minimax' ||
    config.reasoningParser === 'minimax_m2' ||
    config.reasoningParser === 'minimax_m2_5'
  ) {
    config.reasoningParser = 'minimax_m2'
    changed = true
  }
  if (isMiniMaxSessionModel(modelPath) && config.reasoningParser === 'qwen3') {
    config.reasoningParser = 'minimax_m2'
    changed = true
  }
  return changed
}

export const GENERATION_STARTUP_DEFAULTS_VERSION = 4
export const MODEL_PARSER_DEFAULTS_VERSION = 1
export const LEGACY_GENERIC_MAX_OUTPUT_TOKENS = new Set([4096, 12000, 12068, 32768])

export function migrateModelParserDefaults(
  config: Record<string, any>,
  detectedFamily?: string,
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

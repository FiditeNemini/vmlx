export const REASONING_PARSERS_FOR_CLI = new Set([
  'qwen3',
  'deepseek_r1',
  'minimax_m2',
  'minimax_m3',
  'openai_gptoss',
  'mistral',
  'gemma4',
  'think_xml',
])

export function canonicalizeReasoningParserForCli(parser?: string): string | undefined {
  if (!parser || parser === 'auto' || parser === '') return undefined
  if (parser === 'none') return 'none'
  if (parser === 'minimax' || parser === 'minimax_m2_5') return 'minimax_m2'
  // Poolside/Laguna publishes `poolside_v1` in generation_config.json, while
  // the engine registers it as an exact alias of deepseek_r1. Canonicalize at
  // the panel/argv boundary so the vendor spelling cannot be silently dropped
  // to Auto or replaced by the incompatible think_xml parser.
  if (parser === 'poolside_v1') return 'deepseek_r1'
  return REASONING_PARSERS_FOR_CLI.has(parser) ? parser : undefined
}

export interface ReasoningParserResolution {
  configuredParser?: string | null
  detectedParser?: string | null
  supportsThinking?: boolean
}

/**
 * Resolve the one parser identity used by launch argv, command preview,
 * gateway capabilities, chat request policy, and renderer settings.
 *
 * `""` and `"none"` are explicit opt-outs. `auto` and an omitted setting use
 * current bundle detection. A detector that explicitly says the model does
 * not support thinking emits the engine's literal opt-out rather than leaving
 * an absent CLI flag that the backend could reinterpret as Auto.
 */
export function resolveEffectiveReasoningParser(
  input: ReasoningParserResolution,
): string | undefined {
  const configured = typeof input.configuredParser === 'string'
    ? input.configuredParser.trim()
    : undefined
  const detected = typeof input.detectedParser === 'string'
    ? input.detectedParser.trim()
    : undefined

  if (configured === '' || configured === 'none') return 'none'
  if (input.supportsThinking === false) return 'none'

  return canonicalizeReasoningParserForCli(
    configured && configured !== 'auto' ? configured : detected,
  )
}

export function reasoningParserIsEnabled(parser?: string): boolean {
  const canonical = canonicalizeReasoningParserForCli(parser)
  return canonical !== undefined && canonical !== 'none'
}

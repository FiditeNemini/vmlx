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

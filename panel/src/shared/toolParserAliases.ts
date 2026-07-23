const TOOL_PARSER_CANONICAL_ALIASES: Record<string, string> = {
  deepseek_v4: 'dsml',
  hy_v3: 'hunyuan',
}

export const TOOL_PARSERS_FOR_CLI = new Set([
  'mistral',
  'qwen',
  'llama',
  'hermes',
  'deepseek',
  'kimi',
  'lfm2',
  'granite',
  'nemotron',
  'minimax',
  'xlam',
  'functionary',
  'glm47',
  'step3p5',
  'gemma3',
  'gemma3n',
  'xml_function',
  'dsml',
  'zaya_xml',
  'hunyuan',
  'openpangu',
  'generic',
  'qwen3',
  'llama3',
  'llama4',
  'nous',
  'deepseek_v3',
  'deepseek_r1',
  'kimi_k2',
  'moonshot',
  'liquid',
  'granite3',
  'nemotron3',
  'minimax_m2',
  'minimax_m3',
  'meetkai',
  'stepfun',
  'glm4',
  'gemma4',
  'tencent',
  'openpangu_v2',
])

export function canonicalizeToolParserId(
  value: string | null | undefined,
): string | undefined {
  if (value == null) return undefined
  const parser = value.trim()
  if (!parser || parser === 'auto' || parser === 'none') return parser
  const canonical = TOOL_PARSER_CANONICAL_ALIASES[parser] || parser
  return TOOL_PARSERS_FOR_CLI.has(canonical) ? canonical : undefined
}

export interface ToolParserResolution {
  configuredParser?: string | null
  detectedParser?: string | null
}

/**
 * Resolve the one parser identity used by launch argv, command preview, and
 * protocol capability reporting.
 *
 * Empty string and "none" are explicit opt-outs. Auto/missing settings use
 * current bundle detection. A stale unsupported saved value falls back to the
 * current detector instead of reaching argparse as an invalid choice.
 */
export function resolveEffectiveToolParser(
  input: ToolParserResolution,
): string | undefined {
  const configured = typeof input.configuredParser === 'string'
    ? input.configuredParser.trim()
    : undefined
  const detected = typeof input.detectedParser === 'string'
    ? input.detectedParser.trim()
    : undefined

  if (configured === '' || configured === 'none') return 'none'

  const explicit = configured && configured !== 'auto'
    ? canonicalizeToolParserId(configured)
    : undefined
  if (explicit) return explicit
  return canonicalizeToolParserId(detected)
}

export function toolParserIsEnabled(parser?: string): boolean {
  const canonical = canonicalizeToolParserId(parser)
  return canonical !== undefined && canonical !== '' && canonical !== 'auto' && canonical !== 'none'
}

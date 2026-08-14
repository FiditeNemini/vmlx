const TOOL_PARSER_CANONICAL_ALIASES: Record<string, string> = {
  deepseek_v4: 'dsml',
  hy_v3: 'hunyuan',
  // Muse Glimmer's parser registers under all three names in the engine
  // (atem_tool_parser.py: register_module(["atem", "muse_glimmer", "muse"])).
  muse: 'atem',
  muse_glimmer: 'atem',
  // Zaya likewise registers "zaya"/"zyphra" alongside "zaya_xml".
  zaya: 'zaya_xml',
  zyphra: 'zaya_xml',
  // Qwen3.6-27B D-series (and the Qwen 3.8 line) stamp `qwen3_coder`, whose
  // template emits the XML-function shape (<function=NAME><parameter=KEY>),
  // NOT the plain qwen `<tool_call>{json}`. The engine registers the literal
  // name too, but the alias maps to a name every RELEASED engine also knows,
  // so app sessions work against older bundled runtimes as well.
  qwen3_coder: 'xml_function',
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
  // A name missing here is not a no-op: canonicalizeToolParserId returns
  // undefined and toolLaunchArgs then emits NEITHER --tool-call-parser NOR
  // --enable-auto-tool-choice, so the family loses tool calling entirely when
  // launched from the app while `vmlx serve --tool-call-parser <name>` works.
  // That is how Muse Glimmer shipped with dead tool calling in the app: the
  // panel's own registry stamps toolParser:'atem' and the dropdown offers it,
  // but 'atem' was absent here so the choice could not even stick.
  'atem',
  'poolside_v1',
  'mimo_xml_function',
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

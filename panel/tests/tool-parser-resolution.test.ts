import { describe, expect, it } from 'vitest'
import {
  describeDetectedToolParser,
  resolveEffectiveToolParser,
  toolParserIsEnabled,
} from '../src/shared/toolParserAliases'

describe('effective tool parser resolution', () => {
  it.each(['', 'none'])('keeps explicit None (%j) disabled', configuredParser => {
    const parser = resolveEffectiveToolParser({
      configuredParser,
      detectedParser: 'qwen',
    })

    expect(parser).toBe('none')
    expect(toolParserIsEnabled(parser)).toBe(false)
  })

  it('keeps Auto disabled when detection finds no parser', () => {
    const parser = resolveEffectiveToolParser({
      configuredParser: 'auto',
      detectedParser: undefined,
    })

    expect(parser).toBeUndefined()
    expect(toolParserIsEnabled(parser)).toBe(false)
  })

  it('uses a valid detected parser for Auto', () => {
    const parser = resolveEffectiveToolParser({
      configuredParser: 'auto',
      detectedParser: 'qwen',
    })

    expect(parser).toBe('qwen')
    expect(toolParserIsEnabled(parser)).toBe(true)
  })

  it('falls back from a stale saved parser to current detection', () => {
    expect(resolveEffectiveToolParser({
      configuredParser: 'removed_parser_v0',
      detectedParser: 'deepseek_v4',
    })).toBe('dsml')
  })

  it('drops a stale saved parser when detection also has no valid parser', () => {
    expect(resolveEffectiveToolParser({
      configuredParser: 'removed_parser_v0',
      detectedParser: 'also_unknown',
    })).toBeUndefined()
  })
})

// qwen3_coder (Qwen3.6-27B D-series / Qwen 3.8 stamps) must resolve to the
// XML-function parser — the template emits <function=NAME><parameter=KEY>,
// not qwen JSON. A name missing from the alias/CLI tables is not a no-op:
// the session launches with NO tool parser at all (the Muse lesson, hit again
// live on the first D-series serve: "Ignoring unsupported tool parser").
import { canonicalizeToolParserId } from '../src/shared/toolParserAliases'

describe('qwen3_coder alias', () => {
  it('resolves to xml_function for CLI launch', () => {
    expect(canonicalizeToolParserId('qwen3_coder')).toBe('xml_function')
  })
})

describe('describeDetectedToolParser (Auto label)', () => {
  it('shows the canonical resolution for a bundle-stamped alias', () => {
    expect(describeDetectedToolParser('qwen3_coder')).toBe('qwen3_coder → xml_function')
    expect(describeDetectedToolParser('muse_glimmer')).toBe('muse_glimmer → atem')
  })

  it('passes canonical names through untouched', () => {
    expect(describeDetectedToolParser('xml_function')).toBe('xml_function')
    expect(describeDetectedToolParser('qwen')).toBe('qwen')
  })

  it('returns undefined for missing/blank detection', () => {
    expect(describeDetectedToolParser(undefined)).toBeUndefined()
    expect(describeDetectedToolParser(null)).toBeUndefined()
    expect(describeDetectedToolParser('  ')).toBeUndefined()
  })
})

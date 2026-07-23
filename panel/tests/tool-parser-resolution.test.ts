import { describe, expect, it } from 'vitest'
import {
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

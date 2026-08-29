import { describe, expect, it } from 'vitest'
import {
  migrateLegacySessionStartupConfig,
  migrateLegacyStreamIntervalDefault,
  migrateModelParserDefaults,
  MODEL_PARSER_DEFAULTS_VERSION,
} from '../src/shared/sessionConfigMigrations'

describe('database startup migrations', () => {
  it('lifts only the historical stream interval default', () => {
    const legacy = { streamInterval: 1 }
    expect(migrateLegacyStreamIntervalDefault(legacy)).toBe(true)
    expect(legacy.streamInterval).toBe(8)

    for (const streamInterval of [2, 8, 100, undefined]) {
      const current = { streamInterval }
      expect(migrateLegacyStreamIntervalDefault(current)).toBe(false)
      expect(current.streamInterval).toBe(streamInterval)
    }
  })

  it('migrates the stale auto-derived Laguna qwen tool parser exactly once', () => {
    const config: Record<string, any> = { toolCallParser: 'qwen' }

    expect(migrateModelParserDefaults(config, 'laguna')).toBe(true)
    expect(config.toolCallParser).toBe('glm47')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)

    config.toolCallParser = 'qwen'
    expect(migrateModelParserDefaults(config, 'laguna')).toBe(false)
    expect(config.toolCallParser).toBe('qwen')
  })

  it('migrates stale Laguna qwen3 reasoning only when current bundle detection proves deepseek_r1', () => {
    const config: Record<string, any> = {
      modelParserDefaultsVersion: 1,
      reasoningParser: 'qwen3',
    }

    expect(
      migrateModelParserDefaults(config, 'laguna', 'poolside_v1'),
    ).toBe(true)
    expect(config.reasoningParser).toBe('deepseek_r1')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)

    config.reasoningParser = 'qwen3'
    expect(
      migrateModelParserDefaults(config, 'laguna', 'poolside_v1'),
    ).toBe(false)
    expect(config.reasoningParser).toBe('qwen3')
  })

  it('does not rewrite Laguna qwen3 when current bundle detection does not prove a replacement', () => {
    const config: Record<string, any> = {
      modelParserDefaultsVersion: 1,
      reasoningParser: 'qwen3',
    }

    expect(migrateModelParserDefaults(config, 'laguna', 'qwen3')).toBe(true)
    expect(config.reasoningParser).toBe('qwen3')
  })

  it('versions parser defaults without changing unrelated family choices', () => {
    const config: Record<string, any> = { toolCallParser: 'qwen' }

    expect(migrateModelParserDefaults(config, 'qwen3')).toBe(true)
    expect(config.toolCallParser).toBe('qwen')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)
  })

  it('migrates stale qwen4-exp hermes tool parser to qwen exactly once', () => {
    const config: Record<string, any> = {
      modelParserDefaultsVersion: 2,
      toolCallParser: 'hermes',
    }

    expect(migrateModelParserDefaults(config, 'qwen4-exp')).toBe(true)
    expect(config.toolCallParser).toBe('qwen')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)

    // After the version marker is written, a later explicit hermes choice
    // survives — the migration fires exactly once.
    config.toolCallParser = 'hermes'
    expect(migrateModelParserDefaults(config, 'qwen4-exp')).toBe(false)
    expect(config.toolCallParser).toBe('hermes')
  })

  it('leaves non-hermes qwen4-exp parser choices untouched', () => {
    for (const parser of ['qwen', 'auto', '', undefined, 'xml_function']) {
      const config: Record<string, any> = {
        modelParserDefaultsVersion: 2,
        toolCallParser: parser,
      }
      migrateModelParserDefaults(config, 'qwen4-exp')
      expect(config.toolCallParser).toBe(parser)
    }
  })

  it('leaves hermes on every non-qwen4-exp family untouched', () => {
    for (const family of ['qwen3.5', 'qwen3-next', 'qwen3', 'qwen2', 'phi4', 'hermes', 'laguna']) {
      const config: Record<string, any> = {
        modelParserDefaultsVersion: 2,
        toolCallParser: 'hermes',
      }
      migrateModelParserDefaults(config, family)
      expect(config.toolCallParser).toBe('hermes')
    }
  })

  it('preserves an explicit non-stale Laguna parser while versioning defaults', () => {
    const config: Record<string, any> = { toolCallParser: 'none' }

    expect(migrateModelParserDefaults(config, 'laguna')).toBe(true)
    expect(config.toolCallParser).toBe('none')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)
  })

  it.each([4096, 12000, 12068, 32768])(
    'clears legacy session maxTokens=%i before launch can reuse it',
    maxTokens => {
      const config = { maxTokens, reasoningParser: 'qwen3' }

      expect(migrateLegacySessionStartupConfig(config)).toBe(true)

      expect(config.maxTokens).toBe(0)
      expect(config.generationStartupDefaultsVersion).toBe(4)
    },
  )

  it('clears string legacy session maxTokens before launch can reuse it', () => {
    const config = { maxTokens: '32768', reasoningParser: 'qwen3' }

    expect(migrateLegacySessionStartupConfig(config)).toBe(true)

    expect(config.maxTokens).toBe(0)
    expect(config.generationStartupDefaultsVersion).toBe(4)
  })

  it('canonicalizes stale MiniMax reasoning aliases to the registered engine parser', () => {
    for (const reasoningParser of ['minimax', 'minimax_m2', 'minimax_m2_5']) {
      const config = { maxTokens: 0, reasoningParser }

      expect(migrateLegacySessionStartupConfig(config)).toBe(true)

      expect(config.reasoningParser).toBe('minimax_m2')
    }
  })

  it('canonicalizes old MiniMax sessions that persisted qwen3 before minimax_m2 existed', () => {
    const config = { maxTokens: 0, reasoningParser: 'qwen3', toolCallParser: 'minimax' }

    expect(
      migrateLegacySessionStartupConfig(config, '/Users/example/models/JANGQ/MiniMax-M2.7-JANGTQ_K'),
    ).toBe(true)

    expect(config.reasoningParser).toBe('minimax_m2')
  })

  it('keeps qwen3 for non-MiniMax families', () => {
    const config = { maxTokens: 0, reasoningParser: 'qwen3' }

    expect(
      migrateLegacySessionStartupConfig(config, '/Users/example/models/JANGQ/ZAYA1-8B-JANGTQ_K'),
    ).toBe(false)

    expect(config.reasoningParser).toBe('qwen3')
  })

  it('preserves non-generic explicit output caps and supported parsers', () => {
    const config = { maxTokens: 12345, reasoningParser: 'deepseek_r1' }

    expect(migrateLegacySessionStartupConfig(config)).toBe(false)

    expect(config.maxTokens).toBe(12345)
    expect(config.reasoningParser).toBe('deepseek_r1')
  })
})

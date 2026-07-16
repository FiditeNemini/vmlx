import { describe, expect, it } from 'vitest'
import {
  migrateLegacySessionStartupConfig,
  migrateModelParserDefaults,
  MODEL_PARSER_DEFAULTS_VERSION,
} from '../src/shared/sessionConfigMigrations'

describe('database startup migrations', () => {
  it('migrates the stale auto-derived Laguna qwen tool parser exactly once', () => {
    const config: Record<string, any> = { toolCallParser: 'qwen' }

    expect(migrateModelParserDefaults(config, 'laguna')).toBe(true)
    expect(config.toolCallParser).toBe('glm47')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)

    config.toolCallParser = 'qwen'
    expect(migrateModelParserDefaults(config, 'laguna')).toBe(false)
    expect(config.toolCallParser).toBe('qwen')
  })

  it('versions parser defaults without changing unrelated family choices', () => {
    const config: Record<string, any> = { toolCallParser: 'qwen' }

    expect(migrateModelParserDefaults(config, 'qwen3')).toBe(true)
    expect(config.toolCallParser).toBe('qwen')
    expect(config.modelParserDefaultsVersion).toBe(MODEL_PARSER_DEFAULTS_VERSION)
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

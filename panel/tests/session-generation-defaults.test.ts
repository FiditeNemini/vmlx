import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  applyBundleGenerationDefaultsToSessionConfig,
  resolveBundleGenerationDefaults,
} from '../src/shared/sessionGenerationDefaults'

describe('session generation-default hydration', () => {
  it('uses one resolver for standard and JANG sampling precedence', () => {
    expect(resolveBundleGenerationDefaults(
      {
        do_sample: false,
        temperature: 1,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.1,
        max_new_tokens: 2048,
      },
      {
        chat: {
          reasoning: { default_mode: 'chat' },
          sampling_defaults: {
            temperature: 0.6,
            top_p: 0.9,
            repetition_penalty_chat: 1.05,
          },
        },
      },
      { model_type: 'qwen3' },
    )).toEqual({
      temperature: 0.6,
      topP: 0.9,
      topK: 0,
      repeatPenalty: 1.05,
      maxNewTokens: 2048,
      source: 'jang_config',
    })
  })

  it('rejects invalid output-token defaults and normalizes disabled top-k', () => {
    expect(resolveBundleGenerationDefaults(
      { max_new_tokens: -1, top_k: -1 },
      null,
      null,
    )).toEqual({
      topK: 0,
      source: 'generation_config',
    })
  })

  it('keeps DSV4 direct-chat scalar repetition precedence', () => {
    expect(resolveBundleGenerationDefaults(
      null,
      {
        chat: {
          reasoning: { default_mode: 'chat' },
          sampling_defaults: {
            repetition_penalty: 1,
            repetition_penalty_chat: 1.05,
            repetition_penalty_thinking: 1.1,
          },
        },
      },
      { model_type: 'deepseek_v4' },
    )).toMatchObject({
      repeatPenalty: 1,
      source: 'jang_config',
    })
  })

  it('keeps Laguna top-k metadata-owned', () => {
    expect(resolveBundleGenerationDefaults(
      {
        do_sample: true,
        temperature: 1,
        top_p: 1,
      },
      {
        chat: {
          sampling_defaults: {
            temperature: 1,
            top_p: 1,
          },
        },
      },
      { model_type: 'laguna' },
    )).toEqual({
      temperature: 1,
      topP: 1,
      source: 'jang_config',
    })

    expect(resolveBundleGenerationDefaults(
      { do_sample: true, top_k: 32 },
      null,
      { model_type: 'laguna' },
    )).toMatchObject({ topK: 32 })
    expect(resolveBundleGenerationDefaults(
      { do_sample: false },
      null,
      { model_type: 'laguna' },
    )).toMatchObject({ doSample: false })
  })

  it('preserves an explicit do_sample=false declaration', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { doSample: false })).toMatchObject({
      defaultDoSample: false,
      defaultSamplingDefaultsDeclared: true,
    })
  })

  it('preserves an unusually large model-derived top-k', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { topK: 151552 }).defaultTopK)
      .toBe(151552)
  })

  it('hydrates the model-derived maximum output-token default', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { maxNewTokens: 32768 }).defaultMaxNewTokens)
      .toBe(32768)
  })

  it('resets absent bundle defaults to neutral inheritance sentinels', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({ unrelated: true }, null)).toEqual({
      unrelated: true,
      defaultTemperature: 0,
      defaultTopP: 0,
      defaultTopK: 0,
      defaultMinP: 0,
      defaultRepetitionPenalty: 0,
      defaultMaxNewTokens: 0,
      defaultDoSample: undefined,
      defaultSamplingDefaultsDeclared: false,
    })
  })

  it('wires both settings surfaces on initial load and Reset', () => {
    for (const sourcePath of [
      'src/renderer/src/components/sessions/SessionSettings.tsx',
      'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
    ]) {
      const source = readFileSync(sourcePath, 'utf8')
      expect(source.match(/getGenerationDefaults\(/g)).toHaveLength(2)
      expect(source.match(/applyBundleGenerationDefaultsToSessionConfig\(/g)).toHaveLength(2)
      expect(source).toContain('return () => { active = false }')
      expect(source).toContain('resetStillCurrent')
      expect(source).toContain('setConfig(current => applyBundleGenerationDefaultsToSessionConfig(current, generationDefaults))')
    }
  })

  it('uses one shared mapper for fresh and previously launched sessions', () => {
    const source = readFileSync(
      'src/renderer/src/components/sessions/CreateSession.tsx',
      'utf8',
    )
    expect(source.match(/applyBundleGenerationDefaultsToSessionConfig\(/g)).toHaveLength(3)
    expect(source).not.toContain('function applyGenerationDefaultsToConfig')
    expect(source).not.toContain('function applyGenerationDefaultsToStoredConfig')
    expect(source).toContain('Promise.all([')
  })
})

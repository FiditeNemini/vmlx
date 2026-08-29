import { describe, expect, it } from 'vitest'

import {
  buildBenchmarkMessages,
  detectBenchmarkFamily,
  getBenchmarkProfile,
} from '../src/shared/benchmarkProfiles'

describe('benchmark peak profiles', () => {
  it.each([
    ['jangq-ai/Qwen3.8-Flash-Next-JANG_4M', 'qwen38-flash-next'],
    ['jangq-ai/Qwen3.8-27B-JANG_4D', 'qwen38-27b'],
    ['DeepSeek-V4-Flash-0731-JANG-CRACK', 'dsv4-flash'],
    ['GLM-5.3-Flash-JANG-Affine', 'glm53-flash'],
    ['qwen4_exp_text', 'qwen38-flash-next'],
    ['glm5_next', 'glm53-flash'],
  ])('detects %s as %s', (identity, family) => {
    expect(detectBenchmarkFamily(identity)).toBe(family)
  })

  it('uses retained tuned prefill shapes for the two measured Qwen families', () => {
    const flash = getBenchmarkProfile('peak', 'Qwen3.8-Flash-Next-JANG_4M')
    const dense = getBenchmarkProfile('peak', 'Qwen3.8-27B-JANG_4D')

    expect(
      flash.scenarios.filter((row) => row.kind === 'prefill'),
    ).toMatchObject([{ targetPromptTokens: 12_000, repetitions: 3 }])
    expect(
      dense.scenarios.filter((row) => row.kind === 'prefill'),
    ).toMatchObject([{ targetPromptTokens: 6_000, repetitions: 3 }])
  })

  it('keeps the headline decode arm explicit, short, deterministic, and thinking-off', () => {
    const profile = getBenchmarkProfile('peak', 'DeepSeek-V4-Flash')
    const decode = profile.scenarios.find((row) => row.kind === 'decode')

    expect(decode).toMatchObject({
      id: 'peak-code-burst',
      maxTokens: 128,
      temperature: 0,
      repetitions: 3,
      disableThinking: true,
    })
    expect(profile.disclosure).toContain('not sustained agentic throughput')
  })

  it('puts a nonce before the prefill body so repeated runs cannot claim a warm prefix', () => {
    const profile = getBenchmarkProfile('peak', 'GLM-5.3-Flash')
    const scenario = profile.scenarios.find((row) => row.kind === 'prefill')!
    const messages = buildBenchmarkMessages(scenario, 'unique-proof')

    expect(messages[0].content.startsWith('[benchmark:unique-proof]')).toBe(
      true,
    )
    expect(messages[0].content.length).toBeGreaterThan(5_000)
  })

  it('keeps representative prompts separate from peak headline rows', () => {
    const profile = getBenchmarkProfile('representative', 'Qwen3.8-27B')

    expect(profile.scenarios).toHaveLength(4)
    expect(profile.scenarios.every((row) => row.repetitions === 1)).toBe(true)
    expect(profile.scenarios.some((row) => row.kind === 'decode')).toBe(false)
  })
})

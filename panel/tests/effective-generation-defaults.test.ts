import { describe, expect, it } from 'vitest'
import { applyEffectiveSessionGenerationDefaults } from '../src/shared/effectiveGenerationDefaults'

const bundleDefaults = {
  temperature: 1,
  topP: 0.95,
  topK: 20,
  minP: 0.05,
  repeatPenalty: 1.1,
  maxTokens: 4096,
}

describe('effective session generation defaults', () => {
  it('shows the greedy defaults actually installed by deterministic native MTP', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      JSON.stringify({ nativeMtpMode: 'deterministic' }),
      { supported: true },
    )).toEqual({
      temperature: 0,
      topP: 1,
      topK: 0,
      minP: 0,
      repeatPenalty: 1.1,
      maxTokens: 4096,
    })
  })

  it('uses deterministic mode when an older session omitted the mode field', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      '{}',
      { supported: true },
    ).temperature).toBe(0)
  })

  it.each(['auto', 'off'])('keeps bundle defaults for native MTP mode %s', mode => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: mode },
      { supported: true },
    )).toEqual(bundleDefaults)
  })

  it('does not change a non-MTP model or malformed stored config', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      '{bad json',
      { supported: false },
    )).toEqual(bundleDefaults)
  })
})

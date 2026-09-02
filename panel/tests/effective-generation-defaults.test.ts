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
  it('shows greedy values only for an explicit saved deterministic MTP override', () => {
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

  it('pins greedy startup defaults when the session has no explicit MTP mode (Auto)', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      '{}',
      { supported: true },
    )).toEqual({ ...bundleDefaults, temperature: 0, topP: 1, topK: 0, minP: 0 })
  })

  it('pins greedy startup defaults for Auto MTP (deterministic-defaults launch)', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: 'auto' },
      { supported: true },
    )).toEqual({ ...bundleDefaults, temperature: 0, topP: 1, topK: 0, minP: 0 })
  })

  it('preserves bundle sampling when MTP is Off', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: 'off' },
      { supported: true },
    )).toEqual(bundleDefaults)
  })

  it('preserves GLM bundle sampling in model-default-off Auto but pins when a depth override enables MTP', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: 'auto' },
      { supported: true, defaultMode: 'off' },
    )).toEqual(bundleDefaults)

    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: 'auto', nativeMtpDepthOverride: true },
      { supported: true, defaultMode: 'off' },
    )).toEqual({ ...bundleDefaults, temperature: 0, topP: 1, topK: 0, minP: 0 })
  })

  it('does not change a non-MTP model or malformed stored config', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      '{bad json',
      { supported: false },
    )).toEqual(bundleDefaults)
  })
})

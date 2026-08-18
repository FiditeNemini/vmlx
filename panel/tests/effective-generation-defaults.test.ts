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

  // 2026-08-17 BEHAVIOUR CHANGE — these two used to assert that `auto` and a
  // missing mode PRESERVED the bundle's sampling. That is exactly what made
  // native MTP never run: MTP decodes multi-token only on GREEDY requests, and
  // MTP bundles ship temperature 1.0, so the default session silently disabled
  // the headline feature. Measured live: dots3-note at 21.9 t/s with the mode
  // selector reading "Auto". A bundle with MTP heads now decodes greedily
  // unless the user explicitly turns MTP OFF.
  it('pins greedy for an MTP bundle when the session has no explicit MTP mode', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      '{}',
      { supported: true },
    )).toEqual({ ...bundleDefaults, temperature: 0, topP: 1, topK: 0, minP: 0 })
  })

  it.each(['auto', 'deterministic'])('pins greedy for an MTP bundle in mode %s', mode => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: mode },
      { supported: true },
    )).toEqual({ ...bundleDefaults, temperature: 0, topP: 1, topK: 0, minP: 0 })
  })

  it('OFF is the only mode that preserves the bundle sampling', () => {
    expect(applyEffectiveSessionGenerationDefaults(
      bundleDefaults,
      { nativeMtpMode: 'off' },
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

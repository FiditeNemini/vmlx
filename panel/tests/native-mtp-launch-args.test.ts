import { describe, expect, it } from 'vitest'
import { buildNativeMtpLaunchArgs } from '../src/shared/nativeMtpLaunchArgs'

describe('native MTP launch args', () => {
  it('uses the detected measured depth for auto mode', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 3,
      mode: 'auto',
    })).toEqual([
      '--native-mtp-depth',
      '3',
      '--native-mtp-depth-policy',
      'adaptive',
      '--native-mtp-sampling-policy',
      'greedy-only',
    ])
  })

  it('falls back conservatively to depth 1 when the bundle declares none', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      mode: 'auto',
    })).toContain('1')
  })

  it('lets a valid manual override win and clamps it to verifier support', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 1,
      configuredDepth: 99,
      depthOverride: true,
      mode: 'deterministic',
    })).toEqual([
      '--native-mtp-depth',
      '3',
      '--native-mtp-depth-policy',
      'fixed',
      '--native-mtp-sampling-policy',
      'greedy-only',
    ])
  })

  it('enforces greedy sampling in every app MTP-on mode', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 1,
      mode: 'deterministic',
    })).toContain('greedy-only')
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 1,
      mode: 'auto',
    })).toContain('greedy-only')
  })

  it('disables native MTP when an external drafter owns the decode step', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 3,
      mode: 'auto',
      externalSpeculativeActive: true,
    })).toEqual(['--disable-native-mtp'])
  })

  it('emits nothing for unsupported bundles', () => {
    expect(buildNativeMtpLaunchArgs({ supported: false, mode: 'auto' })).toEqual([])
  })
})

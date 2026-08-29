import { describe, expect, it } from 'vitest'
import { buildNativeMtpLaunchArgs } from '../src/shared/nativeMtpLaunchArgs'

describe('native MTP launch args', () => {
  it('emits NO explicit depth for auto mode — adaptive owns the depth', () => {
    // --native-mtp-depth becomes the engine's explicit
    // VMLINUX_NATIVE_MTP_DEPTH override, which pins the start depth and
    // bypasses tuning sidecars, bundle stamps, and the session-scoped
    // adaptive profile. detectedDepth here is mtp_num_hidden_layers — a
    // head LAYER COUNT (1), not a draft depth — so always sending it
    // silently pinned every "adaptive" app session to fixed D1
    // (live-proven on Flash-Next JANG_4M: effective_depth=1
    // source=VMLINUX_NATIVE_MTP_DEPTH).
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 3,
      mode: 'auto',
    })).toEqual([
      '--native-mtp-depth-policy',
      'adaptive',
      '--native-mtp-sampling-policy',
      'compatible-only',
    ])
  })

  it('emits no explicit depth in auto mode even when the bundle declares none', () => {
    const args = buildNativeMtpLaunchArgs({
      supported: true,
      mode: 'auto',
    })
    expect(args).not.toContain('--native-mtp-depth')
    expect(args).toContain('adaptive')
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

  it('preserves sampled requests in Auto and makes greedy an explicit mode', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 1,
      mode: 'deterministic',
    })).toContain('greedy-only')
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 1,
      mode: 'auto',
    })).toContain('compatible-only')
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

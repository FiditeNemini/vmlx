import { describe, expect, it } from 'vitest'
import { adoptNativeMtpConfig } from '../src/shared/nativeMtpAdoption'
import { buildNativeMtpLaunchArgs } from '../src/shared/nativeMtpLaunchArgs'

function relaunchArgs(adopted: ReturnType<typeof adoptNativeMtpConfig>) {
  return buildNativeMtpLaunchArgs({
    supported: true,
    detectedDepth: 1,
    configuredDepth: adopted.nativeMtpDepth,
    depthOverride: adopted.nativeMtpDepthOverride,
    mode: adopted.nativeMtpMode,
  })
}

describe('native MTP adoption round-trip', () => {
  it('keeps an Auto session Auto: deterministic-defaults adopts as auto and relaunches as deterministic-defaults', () => {
    const adopted = adoptNativeMtpConfig(
      { nativeMtpSamplingPolicy: 'deterministic-defaults', nativeMtpDepthPolicy: 'fixed', nativeMtpDepth: 3 },
      'qwen4-exp', 1,
    )
    expect(adopted).toMatchObject({ nativeMtpMode: 'auto', nativeMtpDepth: 3, nativeMtpDepthOverride: true, nativeMtpAdoptionSource: 'process' })
    expect(relaunchArgs(adopted)).toEqual(['--native-mtp-depth', '3', '--native-mtp-depth-policy', 'fixed', '--native-mtp-sampling-policy', 'deterministic-defaults'])
  })

  it('keeps a Deterministic override: greedy-only adopts as deterministic and relaunches as greedy-only', () => {
    const adopted = adoptNativeMtpConfig({ nativeMtpSamplingPolicy: 'greedy-only', nativeMtpDepthPolicy: 'fixed', nativeMtpDepth: 2 }, 'qwen3.5', 1)
    expect(adopted.nativeMtpMode).toBe('deterministic')
    expect(relaunchArgs(adopted)).toEqual(['--native-mtp-depth', '2', '--native-mtp-depth-policy', 'fixed', '--native-mtp-sampling-policy', 'greedy-only'])
  })

  it('recovers explicit D1 / D2 / D3 and Adaptive from the process instead of inventing D3', () => {
    for (const depth of [1, 2, 3]) {
      const adopted = adoptNativeMtpConfig({ nativeMtpSamplingPolicy: 'deterministic-defaults', nativeMtpDepthPolicy: 'fixed', nativeMtpDepth: depth }, 'qwen4-exp', 1)
      expect(adopted).toMatchObject({ nativeMtpDepth: depth, nativeMtpDepthOverride: true })
      expect(relaunchArgs(adopted)).toContain(String(depth))
    }
    const adaptive = adoptNativeMtpConfig({ nativeMtpSamplingPolicy: 'deterministic-defaults', nativeMtpDepthPolicy: 'adaptive' }, 'qwen4-exp', 1)
    expect(adaptive).toMatchObject({ nativeMtpMode: 'auto', nativeMtpDepthOverride: false })
    expect(relaunchArgs(adaptive)).toEqual(['--native-mtp-depth-policy', 'adaptive', '--native-mtp-sampling-policy', 'deterministic-defaults'])
  })

  it('adopts a disabled engine as Off and relaunches disabled', () => {
    for (const proc of [{ nativeMtpDisabled: true }, { nativeMtpSamplingPolicy: 'disabled' as const }]) {
      const adopted = adoptNativeMtpConfig(proc, 'qwen4-exp', 1)
      expect(adopted.nativeMtpMode).toBe('off')
      expect(relaunchArgs(adopted)).toEqual(['--disable-native-mtp'])
    }
  })

  it('falls back to the explicitly requested fixed D3 family default only when the process exposes nothing', () => {
    expect(adoptNativeMtpConfig({}, 'qwen4-exp', 1)).toMatchObject({ nativeMtpMode: 'auto', nativeMtpDepth: 3, nativeMtpDepthOverride: true, nativeMtpAdoptionSource: 'family-default' })
    expect(adoptNativeMtpConfig({}, 'glm5-next', 2)).toMatchObject({ nativeMtpMode: 'auto', nativeMtpDepth: 2, nativeMtpDepthOverride: false, nativeMtpAdoptionSource: 'detected-default' })
    // compatible-only has no UI mode of its own: adopt as auto, which relaunches with deterministic-defaults (documented change)
    expect(adoptNativeMtpConfig({ nativeMtpSamplingPolicy: 'compatible-only' }, 'qwen4-exp', 1).nativeMtpMode).toBe('auto')
  })
})

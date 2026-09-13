import { resolveNativeMtpStartupMode } from './nativeMtpLaunchArgs'

/**
 * Native MTP settings recovered from a LIVE engine process during adoption.
 *
 * The adopted session's stored settings feed the launcher on the next
 * restart, so every mapping here must round-trip through
 * buildNativeMtpLaunchArgs without changing the effective policy:
 *   launch policy 'deterministic-defaults' is what an Auto session emits, so
 *   it maps back to mode 'auto' (mapping it to 'deterministic' re-launched
 *   the session as 'greedy-only' and pinned explicit request temperatures to
 *   greedy — found 2026-09-06 by source audit);
 *   'greedy-only' is the Deterministic override; 'compatible-only' has no UI
 *   mode of its own (the launcher never emits it) and adopts as 'auto';
 *   a disabled engine adopts as mode 'off'.
 * Depth/policy come from the process (health first, argv second), never from
 * a family guess; only when the process exposes nothing does the family
 * default apply (Flash Next Off, 27B Auto; both keep a fixed D3 ceiling;
 * otherwise the detected depth without an override).
 */
export type AdoptedSamplingPolicy = 'compatible-only' | 'deterministic-defaults' | 'greedy-only' | 'disabled'

export interface AdoptableNativeMtpProcess {
  nativeMtpSamplingPolicy?: AdoptedSamplingPolicy
  nativeMtpDepth?: number
  nativeMtpDepthPolicy?: 'fixed' | 'adaptive'
  nativeMtpDisabled?: boolean
}

export interface AdoptedNativeMtpConfig {
  nativeMtpMode: 'auto' | 'deterministic' | 'off'
  nativeMtpDepth: number
  nativeMtpDepthOverride: boolean
  nativeMtpAdoptionSource: 'process' | 'family-default' | 'detected-default'
}

const FIXED_D3_FAMILIES = new Set(['qwen4-exp', 'qwen3.5'])

function clampDepth(raw: unknown): number | undefined {
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return undefined
  const depth = Math.round(raw)
  return depth >= 1 && depth <= 3 ? depth : undefined
}

export function adoptNativeMtpConfig(
  proc: AdoptableNativeMtpProcess,
  detectedFamily: string | undefined,
  detectedDepth: number | undefined,
): AdoptedNativeMtpConfig {
  const familyDefault = FIXED_D3_FAMILIES.has(detectedFamily ?? '')
  const fallbackDepth = familyDefault ? 3 : (clampDepth(detectedDepth) ?? 3)
  const fallback: AdoptedNativeMtpConfig = {
    nativeMtpMode: resolveNativeMtpStartupMode(detectedFamily),
    nativeMtpDepth: fallbackDepth,
    nativeMtpDepthOverride: familyDefault,
    nativeMtpAdoptionSource: familyDefault ? 'family-default' : 'detected-default',
  }
  const policy = proc.nativeMtpSamplingPolicy
  const disabled = proc.nativeMtpDisabled === true || policy === 'disabled'
  const liveDepth = clampDepth(proc.nativeMtpDepth)
  const hasLivePolicy = policy !== undefined || liveDepth !== undefined
    || proc.nativeMtpDepthPolicy !== undefined
  const mode: AdoptedNativeMtpConfig['nativeMtpMode'] = disabled
    ? 'off'
    : policy === 'greedy-only'
      ? 'deterministic'
      : hasLivePolicy ? 'auto' : fallback.nativeMtpMode
  if (proc.nativeMtpDepthPolicy === 'fixed' && liveDepth !== undefined) {
    return { nativeMtpMode: mode, nativeMtpDepth: liveDepth, nativeMtpDepthOverride: true, nativeMtpAdoptionSource: 'process' }
  }
  if (proc.nativeMtpDepthPolicy === 'adaptive') {
    return { nativeMtpMode: mode, nativeMtpDepth: liveDepth ?? fallbackDepth, nativeMtpDepthOverride: false, nativeMtpAdoptionSource: 'process' }
  }
  return { ...fallback, nativeMtpMode: mode, nativeMtpAdoptionSource: policy || disabled ? 'process' : fallback.nativeMtpAdoptionSource }
}

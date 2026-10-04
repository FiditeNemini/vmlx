import { resolveNativeMtpStartupMode, type NativeMtpMode } from './nativeMtpLaunchArgs'

/**
 * Native MTP mode recovered from a LIVE engine process during adoption.
 *
 * The adopted session's stored mode feeds the launcher on the next restart,
 * so the mapping must round-trip through buildNativeMtpLaunchArgs without
 * changing the effective policy:
 *   a disabled engine adopts as 'off';
 *   any engine that exposes a native-MTP policy, depth or depth policy
 *   (including a legacy fixed-depth or greedy-only process started by an
 *   older app) adopts as 'adaptive' — the only MTP-on mode the app launches;
 *   a process that exposes nothing takes the startup default.
 * A legacy process's fixed depth or greedy-only sampling is NOT carried
 * forward: on restart the session runs the adaptive policy with the bundle's
 * own sampler, which is the product contract.
 */
export type AdoptedSamplingPolicy = 'compatible-only' | 'deterministic-defaults' | 'greedy-only' | 'disabled'

export interface AdoptableNativeMtpProcess {
  nativeMtpSamplingPolicy?: AdoptedSamplingPolicy
  nativeMtpDepth?: number
  nativeMtpDepthPolicy?: 'fixed' | 'adaptive'
  nativeMtpDisabled?: boolean
}

export interface AdoptedNativeMtpConfig {
  nativeMtpMode: NativeMtpMode
  nativeMtpAdoptionSource: 'process' | 'model-default'
}

export function adoptNativeMtpConfig(
  proc: AdoptableNativeMtpProcess,
  detectedFamily: string | undefined,
  modelDefaultMode?: 'auto' | 'off',
): AdoptedNativeMtpConfig {
  const policy = proc.nativeMtpSamplingPolicy
  const disabled = proc.nativeMtpDisabled === true || policy === 'disabled'
  if (disabled) return { nativeMtpMode: 'off', nativeMtpAdoptionSource: 'process' }
  const hasLivePolicy = policy !== undefined
    || proc.nativeMtpDepthPolicy !== undefined
    || (typeof proc.nativeMtpDepth === 'number' && Number.isFinite(proc.nativeMtpDepth))
  if (hasLivePolicy) return { nativeMtpMode: 'adaptive', nativeMtpAdoptionSource: 'process' }
  return {
    nativeMtpMode: resolveNativeMtpStartupMode(detectedFamily, undefined, modelDefaultMode),
    nativeMtpAdoptionSource: 'model-default',
  }
}

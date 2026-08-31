import { finitePositiveInteger } from './launchArgValues'

export interface NativeMtpLaunchPolicyInput {
  supported: boolean
  detectedDepth?: number
  configuredDepth?: number
  depthOverride?: boolean
  mode?: 'auto' | 'deterministic' | 'off'
  modelDefaultMode?: 'auto' | 'off'
  externalSpeculativeActive?: boolean
}

export function resolveNativeMtpMode(
  input: Pick<NativeMtpLaunchPolicyInput, 'mode' | 'modelDefaultMode' | 'depthOverride'>,
): 'auto' | 'deterministic' | 'off' {
  const configured = input.mode || 'auto'
  // GLM-5.3 is measured slower under its current verifier, so its bundle-level
  // Auto policy is AR. A fixed D1-D3 selection is an explicit opt-in and must
  // remain available for measurement/tuning.
  if (
    configured === 'auto'
    && input.modelDefaultMode === 'off'
    && input.depthOverride !== true
  ) return 'off'
  return configured
}

/** One source of truth for Electron preview and the process launcher. */
export function buildNativeMtpLaunchArgs(
  input: NativeMtpLaunchPolicyInput,
): string[] {
  if (!input.supported) return []
  const mode = resolveNativeMtpMode(input)
  if (mode === 'off' || input.externalSpeculativeActive) {
    return ['--disable-native-mtp']
  }

  const samplingArgs = [
    '--native-mtp-sampling-policy',
    mode === 'deterministic' ? 'greedy-only' : 'compatible-only',
  ]
  if (input.depthOverride !== true) {
    // Adaptive policy: emit NO explicit depth. --native-mtp-depth becomes
    // the engine's explicit VMLINUX_NATIVE_MTP_DEPTH override, which pins
    // the start depth and bypasses tuning sidecars, bundle stamps, and the
    // session-scoped adaptive profile. This launcher used to always send
    // it with detectedDepth — and detectedDepth comes from
    // mtp_num_hidden_layers, a HEAD LAYER COUNT (1), not a draft depth —
    // so every "adaptive" app session was silently pinned to fixed D1
    // (live-proven on Flash-Next JANG_4M: effective_depth=1
    // source=VMLINUX_NATIVE_MTP_DEPTH).
    return ['--native-mtp-depth-policy', 'adaptive', ...samplingArgs]
  }
  const depth = Math.max(
    1,
    Math.min(
      3,
      finitePositiveInteger(input.configuredDepth)
        || finitePositiveInteger(input.detectedDepth)
        || 1,
    ),
  )
  return [
    '--native-mtp-depth',
    depth.toString(),
    '--native-mtp-depth-policy',
    'fixed',
    ...samplingArgs,
  ]
}

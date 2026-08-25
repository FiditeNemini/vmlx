import { finitePositiveInteger } from './launchArgValues'

export interface NativeMtpLaunchPolicyInput {
  supported: boolean
  detectedDepth?: number
  configuredDepth?: number
  depthOverride?: boolean
  mode?: 'auto' | 'deterministic' | 'off'
  externalSpeculativeActive?: boolean
}

/** One source of truth for Electron preview and the process launcher. */
export function buildNativeMtpLaunchArgs(
  input: NativeMtpLaunchPolicyInput,
): string[] {
  if (!input.supported) return []
  if (input.mode === 'off' || input.externalSpeculativeActive) {
    return ['--disable-native-mtp']
  }

  const selectedDepth = input.depthOverride === true
    ? input.configuredDepth
    : input.detectedDepth
  const depth = Math.max(
    1,
    Math.min(
      3,
      finitePositiveInteger(selectedDepth)
        || finitePositiveInteger(input.detectedDepth)
        || 1,
    ),
  )
  return [
    '--native-mtp-depth',
    depth.toString(),
    '--native-mtp-sampling-policy',
    'deterministic-defaults',
  ]
}

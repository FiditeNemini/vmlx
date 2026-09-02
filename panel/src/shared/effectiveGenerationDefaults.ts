import { resolveNativeMtpMode } from './nativeMtpLaunchArgs'

export interface GenerationDefaultsLike {
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxTokens?: number
  maxThinkingTokens?: number
}

interface NativeMtpDetection {
  supported?: boolean
  defaultMode?: 'auto' | 'off'
}

function sessionConfigObject(config: string | Record<string, unknown> | undefined): Record<string, unknown> {
  if (!config) return {}
  if (typeof config !== 'string') return config
  try {
    const parsed = JSON.parse(config)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

/**
 * Apply the effective sampling a session will really use, given native MTP.
 *
 * Any enabled native-MTP mode pins the engine's STARTUP DEFAULTS to greedy:
 * Auto launches with the deterministic-defaults policy (temperature 0,
 * top_p 1, top_k dropped as defaults; an explicit per-request temperature in
 * API kwargs or chat overrides still wins), and Deterministic launches with
 * hard greedy-only. Chat Settings must display the same pinned defaults the
 * engine resolves, or a fresh chat would show the bundle's sampled
 * temperature while actually running greedy. Off preserves the bundle
 * distribution and uses AR.
 */
export function applyEffectiveSessionGenerationDefaults<T extends GenerationDefaultsLike>(
  defaults: T,
  sessionConfig: string | Record<string, unknown> | undefined,
  nativeMtp: NativeMtpDetection | undefined,
): T {
  const config = sessionConfigObject(sessionConfig)
  const configuredMode = typeof config.nativeMtpMode === 'string'
    ? config.nativeMtpMode as 'auto' | 'deterministic' | 'off'
    : 'auto'
  const mode = resolveNativeMtpMode({
    mode: configuredMode,
    modelDefaultMode: nativeMtp?.defaultMode,
    depthOverride: config.nativeMtpDepthOverride === true,
  })
  // No MTP heads -> MTP has no say over sampling.
  if (nativeMtp?.supported !== true) return defaults
  // Off preserves the bundle/request sampling distribution and uses AR.
  if (mode === 'off') return defaults
  return {
    ...defaults,
    temperature: 0,
    topP: 1,
    topK: 0,
    minP: 0,
  }
}

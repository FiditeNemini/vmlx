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
 * Auto launches the engine's compatible sampling policy, which preserves the
 * bundle/request distribution and uses stochastic speculative verification
 * when temperature is nonzero. Deterministic is the explicit greedy-only
 * policy. Chat Settings must display the same distinction as the launcher;
 * coercing Auto to greedy here would silently override the bundle contract
 * even though the engine can verify sampled proposals exactly.
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
  // Auto and Off both preserve the bundle/request sampling distribution. Auto
  // still runs native MTP through rejection-sampling verification; Off uses AR.
  if (mode !== 'deterministic') return defaults
  return {
    ...defaults,
    temperature: 0,
    topP: 1,
    topK: 0,
    minP: 0,
  }
}

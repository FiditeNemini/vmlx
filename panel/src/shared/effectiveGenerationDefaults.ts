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
 * Apply server-startup policy to bundle-derived generation defaults.
 *
 * Native MTP's `deterministic-defaults` launch mode intentionally changes
 * omitted requests to greedy sampling so they exercise identity-verified MTP.
 * Chat sliders describe the effective inherited request, so they must show
 * that startup policy instead of the bundle's stochastic generation_config.
 * Explicit per-chat slider changes still travel in the request and therefore
 * retain request-over-server precedence.
 */
export function applyEffectiveSessionGenerationDefaults<T extends GenerationDefaultsLike>(
  defaults: T,
  sessionConfig: string | Record<string, unknown> | undefined,
  nativeMtp: NativeMtpDetection | undefined,
): T {
  const config = sessionConfigObject(sessionConfig)
  const mode = typeof config.nativeMtpMode === 'string'
    ? config.nativeMtpMode
    : 'deterministic'
  if (nativeMtp?.supported !== true || mode !== 'deterministic') return defaults
  return {
    ...defaults,
    temperature: 0,
    topP: 1,
    topK: 0,
    minP: 0,
  }
}

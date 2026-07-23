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
 * Apply an explicit saved server policy to bundle-derived chat defaults.
 *
 * Fresh sessions default to nativeMtpMode=auto and therefore preserve the
 * bundle's generation_config/jang_config sampling. Only a session explicitly
 * saved as deterministic replaces omitted request values with greedy sampling;
 * its Chat Settings controls must display that real effective inheritance.
 */
export function applyEffectiveSessionGenerationDefaults<T extends GenerationDefaultsLike>(
  defaults: T,
  sessionConfig: string | Record<string, unknown> | undefined,
  nativeMtp: NativeMtpDetection | undefined,
): T {
  const config = sessionConfigObject(sessionConfig)
  const mode = typeof config.nativeMtpMode === 'string'
    ? config.nativeMtpMode
    : 'auto'
  if (nativeMtp?.supported !== true || mode !== 'deterministic') return defaults
  return {
    ...defaults,
    temperature: 0,
    topP: 1,
    topK: 0,
    minP: 0,
  }
}

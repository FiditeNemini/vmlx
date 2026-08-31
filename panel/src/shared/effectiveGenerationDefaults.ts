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
 * 2026-08-17 — BEHAVIOUR CHANGE. This previously required an explicitly saved
 * `deterministic` session, so the DEFAULT (`auto`) preserved the bundle's own
 * sampling. Native MTP only decodes multiple tokens per step on GREEDY
 * requests, and MTP bundles ship `temperature: 1.0` (dots3-note, the Qwen MTP
 * builds) — so out of the box MTP never engaged, users got ordinary decode
 * speed, and nothing said why. Measured live: dots3-note at 21.9 t/s with the
 * mode selector reading "Auto".
 *
 * Eric: "if it has mtp it sets temp to 0 and lets user know in the chat
 * settings bar temp area and shows also in the mtp server settings".
 *
 * So a bundle that carries MTP heads normally decodes greedily unless the user
 * turns MTP OFF. A measured family may declare ``defaultMode: off``; for that
 * family Auto is ordinary AR and a fixed D1-D3 selection is the explicit MTP
 * opt-in. Chat Settings must show the same effective policy as launch.
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
  // The user explicitly disabled MTP: honour the bundle's sampling.
  if (mode === 'off') return defaults
  return {
    ...defaults,
    temperature: 0,
    topP: 1,
    topK: 0,
    minP: 0,
  }
}

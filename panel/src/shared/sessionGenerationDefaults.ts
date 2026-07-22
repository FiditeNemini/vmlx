export interface BundleGenerationDefaults {
  doSample?: boolean
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxNewTokens?: number
}

export interface SessionGenerationDefaultFields {
  defaultTemperature: number
  defaultTopP: number
  defaultTopK: number
  defaultMinP: number
  defaultRepetitionPenalty: number
  defaultMaxNewTokens: number
  defaultDoSample?: boolean
  defaultSamplingDefaultsDeclared: boolean
}

export function hasDeclaredBundleSamplingDefaults(
  defaults: BundleGenerationDefaults | null | undefined,
): boolean {
  return !!defaults && (
    defaults.doSample === false ||
    defaults.temperature != null ||
    defaults.topP != null ||
    defaults.topK != null ||
    defaults.minP != null ||
    defaults.repeatPenalty != null
  )
}

/**
 * Copy bundle-derived generation defaults into the fields displayed by the
 * session settings surfaces. These are inherited request defaults, not CLI
 * overrides, so absent values intentionally reset to the neutral sentinel.
 */
export function applyBundleGenerationDefaultsToSessionConfig<T extends object>(
  config: T,
  defaults: BundleGenerationDefaults | null | undefined,
): T & SessionGenerationDefaultFields {
  return {
    ...config,
    defaultTemperature: defaults?.temperature != null
      ? Math.round(defaults.temperature * 100)
      : 0,
    defaultTopP: defaults?.topP != null ? Math.round(defaults.topP * 100) : 0,
    defaultTopK: defaults?.topK != null ? Math.max(0, Math.round(defaults.topK)) : 0,
    defaultMinP: defaults?.minP != null ? Math.max(0, Math.round(defaults.minP * 100)) : 0,
    defaultRepetitionPenalty: defaults?.repeatPenalty != null
      ? Math.round(defaults.repeatPenalty * 100)
      : 0,
    defaultMaxNewTokens: defaults?.maxNewTokens != null
      ? Math.max(0, Math.round(defaults.maxNewTokens))
      : 0,
    defaultDoSample: typeof defaults?.doSample === 'boolean' ? defaults.doSample : undefined,
    defaultSamplingDefaultsDeclared: hasDeclaredBundleSamplingDefaults(defaults),
  }
}

/**
 * DSV4 Flash runtime env mapping.
 *
 * Product sessions use the normal prefix/paged/block-disk CLI controls for
 * native composite reuse. This helper owns only the fixed DSV4 runtime
 * envelope and the bundle-derived pool codec; saved panel cache fields must
 * never be translated into a second hidden cache policy.
 *
 * Knobs:
 *   - `dsv4FinalizerTokens` and `dsv4ForceDirect` are retained only for
 *     migration compatibility with older saved sessions. They intentionally
 *     do not emit env vars: the runtime must not inject thinking tags or
 *     silently flip requested reasoning rails.
 *   - `DSV4_POOL_QUANT` controls the model's internal CSA/HCA pool codec and
 *     is independent of reusable prefix/paged/L2 state. Electron emits it only
 *     when the live bundle detector found an explicit boolean cache stamp;
 *     otherwise the engine loader derives the value from `jang_config.json`.
 *   - `DSV4_ACTIVATION_QAT` is a user-owned, restart-required fidelity knob.
 *     It defaults Off. On enables the source-native E4M3 attention-KV/pool
 *     round-trips and Hadamard+FP4 indexer round-trips in the JANG graph. It
 *     does not control the separate FP32 compressor staging contract.
 *   - `--dsv4-enable-prefix-cache` is a deprecated compatibility alias. Product
 *     sessions intentionally use only the normal prefix/paged/block-disk flags.
 *
 * Natural model behavior wins: bundle chat/generation config plus explicit
 * per-request controls are the only model-behavior inputs.
 */

export interface Dsv4EnvConfig {
  /** Kept for config migration compatibility; raw max is no longer env-gated. */
  dsv4RawMax?: boolean
  dsv4FinalizerTokens?: number
  dsv4ForceDirect?: boolean
  dsv4ActivationQat?: boolean
}

export interface Dsv4EnvOptions {
  dsv4Active?: boolean
  dsv4PoolQuantDefault?: boolean
}

/**
 * Resolve the family the engine will actually use for this session.
 *
 * `--model-family` bypasses engine autodetection, so every family-gated panel
 * behavior must give a saved, non-Auto override the same precedence. Empty and
 * Auto values retain the detected family.
 */
export function resolveEffectiveModelFamily(
  modelFamilyOverride: string | null | undefined,
  detectedFamily: string | null | undefined,
): string | undefined {
  const override = String(modelFamilyOverride || '').trim()
  if (override && override !== 'auto') return override
  const detected = String(detectedFamily || '').trim()
  return detected || undefined
}

export function dsv4EnvFromConfig(
  config: Dsv4EnvConfig | null | undefined,
  options: Dsv4EnvOptions = {},
): Record<string, string> {
  if (!config) return {}
  const env: Record<string, string> = {}

  if (options.dsv4Active === true) {
    env.DSV4_LONG_CTX = '1'
    env.DSV4_ACTIVATION_QAT = config.dsv4ActivationQat === true ? '1' : '0'
    if (typeof options.dsv4PoolQuantDefault === 'boolean') {
      env.DSV4_POOL_QUANT = options.dsv4PoolQuantDefault ? '1' : '0'
    }
  }

  return env
}

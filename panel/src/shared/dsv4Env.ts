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
}

export interface Dsv4EnvOptions {
  dsv4Active?: boolean
  dsv4PoolQuantDefault?: boolean
}

export function dsv4EnvFromConfig(
  config: Dsv4EnvConfig | null | undefined,
  options: Dsv4EnvOptions = {},
): Record<string, string> {
  if (!config) return {}
  const env: Record<string, string> = {}

  if (options.dsv4Active === true) {
    env.DSV4_LONG_CTX = '1'
    if (typeof options.dsv4PoolQuantDefault === 'boolean') {
      env.DSV4_POOL_QUANT = options.dsv4PoolQuantDefault ? '1' : '0'
    }
  }

  return env
}

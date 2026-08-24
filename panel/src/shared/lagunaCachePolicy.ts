export interface LagunaCacheDetection {
  family?: string
  architectureHints?: Record<string, string | number | boolean>
}

export interface LagunaTurboQuantPolicyInput {
  detected?: LagunaCacheDetection | null
  kvCacheQuantization?: string
  /**
   * True only when the launcher will emit an explicit
   * `--kv-cache-quantization` value. Explicit q4/q8/none disables the
   * loader-owned live TurboQuant cache; merely retaining a saved value while
   * prefix caching is inactive does not.
   */
  explicitKvCacheQuantizationApplied: boolean
}

export interface LagunaJitLaunchPolicyInput
  extends LagunaTurboQuantPolicyInput {
  /** The saved Electron JIT toggle for this session. */
  enableJitRequested: boolean
}

export const DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV =
  'VMLX_DISABLE_JANG_AFFINE_JIT_DEFAULT'

export function shouldDisableLagunaJitDefault(
  input: LagunaJitLaunchPolicyInput,
): boolean {
  if (String(input.detected?.family || '').toLowerCase() !== 'laguna') {
    return false
  }
  // Honor the saved Electron Off choice for every Laguna bundle. The
  // environment variable is inert for non-affine weights, while it prevents
  // the CLI's affine-JANG auto-default from silently reversing the toggle.
  if (!input.enableJitRequested) return true
  return isLagunaMixedSwaTurboQuantEffective(input)
}

/**
 * Apply the exact child-process environment contract for this topology.
 *
 * This mutates only the fresh child environment. It deliberately leaves an
 * inherited value untouched for unrelated models; a parent-shell opt-out is
 * not prior-session leakage. Returns true when Electron explicitly installs
 * the Laguna disable for this launch.
 */
export function applyLagunaJitDefaultEnvironment(
  env: Record<string, string | undefined>,
  input: LagunaJitLaunchPolicyInput,
): boolean {
  if (!shouldDisableLagunaJitDefault(input)) return false
  env[DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV] = '1'
  return true
}

export function isLagunaMixedFullSlidingTopology(
  detected?: LagunaCacheDetection | null,
): boolean {
  if (String(detected?.family || '').toLowerCase() !== 'laguna') return false
  const hints = detected?.architectureHints
  return (
    hints?.attentionArch === 'full_and_sliding_kv' &&
    hints?.cacheSchema === 'mixed_swa_kv_v1' &&
    hints?.selectiveTurboQuantKv === true
  )
}

/**
 * Compatibility seam for the retired Laguna Auto-TQ launch policy.
 *
 * Production Auto now preserves Laguna's native interleaved KVCache and
 * RotatingKVCache slots. The Electron app has no control that requests the
 * environment-only generic-TQ diagnostic override, so this must remain false;
 * otherwise the preview suppresses JIT for a wrapper the spawned engine never
 * instantiates.
 */
export function isLagunaMixedSwaTurboQuantEffective(
  _input: LagunaTurboQuantPolicyInput,
): boolean {
  return false
}

/**
 * Canonical production policy for stored prefix-cache representation.
 *
 * Prefix reuse is answer-preserving only when the SSD record keeps the exact
 * architecture-native state. That may be full KV, rotating/SWA metadata,
 * recurrent SSM/GDN companions, sparse-index state, or a bundle-owned native
 * compressed representation. The Electron product must never add q4/q8 on top.
 *
 * Live-proven on Laguna-S-2.1-JANG_4M-CRACK 2026-08-12 at temperature 0, same
 * prompt sent cold and then on a confirmed cache hit (sha of generated text):
 *
 *     stored q8       cold bb040715 -> hit 633c133d   DIVERGED
 *     stored exact    cold bb040715 -> hit bb040715   EXACT
 *
 * Mixed-SWA supplied the first live proof that the old q4/q8 selector was not
 * answer preserving. The policy is intentionally global now: one native value
 * in persisted settings, one launch path, and no family-specific exception.
 *
 * Keep every consumer on these functions. The JIT rule in this codebase was
 * hand-copied into four places and two of them spelled a condition differently
 * -- see `jitPolicy.ts` for why that matters.
 */

/** Cache subtypes whose native cache interleaves full and sliding slots. */
const MIXED_SWA_CACHE_SUBTYPES = new Set([
  'mixed_swa_kv',
  'step3p7_full_sliding_kv',
])

export interface StoredKvQuantPolicyInput {
  cacheType?: string
  cacheSubtype?: string
  architectureHints?: Record<string, string | number | boolean> | null
}

/**
 * True when the bundle interleaves sliding-window and full attention.
 *
 * Accepts the three independent signals the panel already carries, because a
 * bundle may surface only one of them: the detected cache subtype, the
 * `rotating_kv` cache type, and the architecture hints the loader reports.
 */
export function isMixedSwaBundle(input: StoredKvQuantPolicyInput): boolean {
  if (MIXED_SWA_CACHE_SUBTYPES.has(String(input.cacheSubtype || ''))) {
    return true
  }
  if (String(input.cacheType || '') === 'rotating_kv') {
    return true
  }
  const hints = input.architectureHints
  if (hints) {
    if (String(hints.cacheSchema || '') === 'mixed_swa_kv_v1') return true
    if (String(hints.attentionArch || '') === 'full_and_sliding_kv') return true
  }
  return false
}

/**
 * True when a lossy stored prefix codec (q8/q4) must not be offered.
 */
export function storedKvQuantMustBeExact(
  _input: StoredKvQuantPolicyInput,
): boolean {
  return true
}

/** Stored-codec options the selector may offer for this bundle. */
export function allowedStoredKvQuantOptions(
  _input: StoredKvQuantPolicyInput,
): string[] {
  return ['auto']
}

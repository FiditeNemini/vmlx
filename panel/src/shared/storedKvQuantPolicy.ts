/**
 * Canonical policy for which bundles may use a LOSSY stored prefix cache.
 *
 * Mixed sliding/full attention bundles (Laguna, Gemma 4, Step-3.7) keep
 * RotatingKVCache window metadata, and a quantized STORED prefix changes their
 * answers on reuse: the cold prefill computes full-precision KV while the warm
 * turn reads back the quantized copy. Asking the same question twice then gives
 * two different answers, and the second one is the degraded one.
 *
 * Live-proven on Laguna-S-2.1-JANG_4M-CRACK 2026-08-12 at temperature 0, same
 * prompt sent cold and then on a confirmed cache hit (sha of generated text):
 *
 *     stored q8       cold bb040715 -> hit 633c133d   DIVERGED
 *     stored exact    cold bb040715 -> hit bb040715   EXACT
 *
 * The engine already refuses this by default (`b6522591f`, mixed-SWA bundles
 * auto-select exact stored KV). This module is the UI half: the selector must
 * not offer a setting that silently changes answers.
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
 *
 * Distinct from "the runtime owns the codec" (DSV4/M3/openPangu): those have a
 * native typed cache, whereas this is about answer stability on reuse.
 */
export function storedKvQuantMustBeExact(
  input: StoredKvQuantPolicyInput,
): boolean {
  return isMixedSwaBundle(input)
}

/** Stored-codec options the selector may offer for this bundle. */
export function allowedStoredKvQuantOptions(
  input: StoredKvQuantPolicyInput,
): string[] {
  return storedKvQuantMustBeExact(input)
    ? ['auto', 'none']
    : ['auto', 'none', 'q8', 'q4']
}

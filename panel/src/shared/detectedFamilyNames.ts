/**
 * Canonical mapping from an ENGINE family_name to the panel's REGISTRY name.
 *
 * The engine and the panel deliberately spell some families differently
 * (`deepseek_v4` vs `deepseek-v4`, `zaya1_vl` vs `zaya1-vl`, `bailing_hybrid`
 * vs `ling`). Getting that translation wrong is not a cosmetic bug: a lookup
 * keyed on the wrong spelling silently misses, the setting appears to apply,
 * and the fix reads as a no-op — this project has shipped that failure before.
 *
 * These two rules previously existed as three byte-identical private copies
 * each (src/main/sessions.ts, SessionConfigForm.tsx, SessionSettings.tsx),
 * spanning the main AND renderer processes. They live here so a new family
 * alias is added once, and so main and renderer can never disagree about what
 * a family is called.
 */

export function normalizeDetectedFamilyName(family?: string): string | undefined {
  if (!family) return undefined
  if (family === 'deepseek_v4') return 'deepseek-v4'
  if (family === 'zaya1_vl') return 'zaya1-vl'
  if (family === 'bailing_hybrid') return 'ling'
  return family
}

/**
 * ZAYA's CCA cache is path-dependent, so several settings surfaces gate on it.
 * Always ask through the normalizer — the engine spells the VL variant
 * `zaya1_vl`, the registry spells it `zaya1-vl`.
 */
export function isZayaCcaFamily(family?: string): boolean {
  const normalized = normalizeDetectedFamilyName(family)
  return normalized === 'zaya' || normalized === 'zaya1-vl'
}

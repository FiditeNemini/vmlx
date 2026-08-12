/**
 * ONE definition of which detected cache shapes require paged KV and which can
 * run the block disk cache as an SSD-only tier.
 *
 * These four predicates were byte-identical copies in src/main/sessions.ts (the
 * process that BUILDS the engine argv) and SessionSettings.tsx (the surface that
 * PREVIEWS that argv and enables/disables the matching toggles). A divergence
 * here is invisible until someone launches: the preview offers an SSD-only tier
 * the launcher never passes, or a toggle stays greyed out for a cache shape that
 * in fact supports it.
 *
 * Values are the engine's cache_type / cache_subtype strings as reported by
 * model detection — NOT panel registry family names.
 */

const PAGED_REQUIRED_CACHE_TYPES = new Set(['hybrid', 'mamba', 'rotating_kv'])
const PAGED_REQUIRED_CACHE_SUBTYPES = new Set(['step3p7_full_sliding_kv', 'mixed_swa_kv'])

export function cacheTypeRequiresPaged(cacheType?: string): boolean {
  return !!cacheType && PAGED_REQUIRED_CACHE_TYPES.has(cacheType)
}

export function cacheSubtypeRequiresPaged(cacheSubtype?: string): boolean {
  return !!cacheSubtype && PAGED_REQUIRED_CACHE_SUBTYPES.has(cacheSubtype)
}

/**
 * The same shapes that require paged KV are the ones whose block chain can be
 * served straight off SSD without a RAM tier. They are deliberately separate
 * predicates rather than aliases: the two properties are independent in
 * principle, and collapsing them would make a future divergence unexpressible.
 */
export function cacheTypeSupportsBlockDiskOnly(cacheType?: string): boolean {
  return cacheTypeRequiresPaged(cacheType)
}

export function cacheSubtypeSupportsBlockDiskOnly(cacheSubtype?: string): boolean {
  return cacheSubtypeRequiresPaged(cacheSubtype)
}

/** Native/generic execution strings and structured adaptive selections. */
export function formatCacheSelection(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() || null
  if (!value || typeof value !== 'object') return null
  const { selected, rejected } = value as { selected?: unknown; rejected?: unknown }
  if (typeof selected !== 'string' || !selected.trim()) return null
  return selected + (typeof rejected === 'string' && rejected ? ` ← ${rejected}` : '')
}

/**
 * Preserve every cache tier observed across one logical agent turn.
 * A later tool generation must not erase the earlier disk-restored evidence.
 */
export function mergeCacheDetails(current?: string, next?: string): string {
  const tiers: string[] = [];
  const seen = new Set<string>();
  for (const detail of [current, next]) {
    for (const tier of String(detail || "").split("+")) {
      const normalized = tier.trim();
      if (!normalized || seen.has(normalized)) continue;
      seen.add(normalized);
      tiers.push(normalized);
    }
  }
  return tiers.join("+");
}

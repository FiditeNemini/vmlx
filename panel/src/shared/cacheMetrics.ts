/**
 * Preserve every cache tier observed across one logical agent turn.
 *
 * Tool execution creates multiple HTTP generations. A disk-restored first
 * iteration can be followed by a resident paged hit after the tool result; the
 * later usage event must not erase the earlier `disk` evidence.
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

const DEFAULT_BADGE_CHARACTER_LIMIT = 24;

/**
 * Keep a detector label recognizable without inventing a broader codec name.
 * The full authoritative value remains available to the caller for a tooltip.
 */
export function compactQuantizationBadgeLabel(
  label: string,
  characterLimit = DEFAULT_BADGE_CHARACTER_LIMIT,
): string {
  const trimmed = label.trim();
  if (!trimmed || trimmed.length <= characterLimit) return trimmed;

  const tokens = trimmed.split("_");
  let compact = tokens[0] || trimmed.slice(0, characterLimit - 1);
  for (const token of tokens.slice(1)) {
    const candidate = `${compact}_${token}`;
    if (candidate.length > characterLimit - 1) break;
    compact = candidate;
  }

  if (compact.length >= characterLimit) {
    compact = compact.slice(0, Math.max(1, characterLimit - 1));
  }
  return `${compact}…`;
}

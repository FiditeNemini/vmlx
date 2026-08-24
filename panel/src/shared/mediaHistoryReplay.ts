/**
 * Rebuild one persisted user content array for the next model request.
 *
 * A multimodal conversation must replay historical media parts exactly. Turning
 * an earlier image/video/audio item into explanatory text rewrites the prompt,
 * prevents longest-prefix SSD reuse, and makes a cold/restarted engine unable
 * to reconstruct the same model-visible history. Text-only routes still strip
 * media so an imported or stale multimodal row cannot break a text model.
 */
export function replayPersistedUserContentParts(
  parts: any[],
  acceptsMedia: boolean,
): any[] | string {
  if (acceptsMedia) return parts

  const text = parts
    .filter((part) => part && typeof part === 'object')
    .filter((part) => part.type === 'text' || part.type === 'input_text')
    .map((part) => String(part.text || ''))
    .filter((value) => value.trim())
    .join('\n')
  return text || '[Image omitted]'
}

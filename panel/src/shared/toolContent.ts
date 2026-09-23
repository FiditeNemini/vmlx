/** Preserve bytes already streamed before a tool boundary or continuation.
 * Trimming a segment here rewrites the visible answer and its tool offsets.
 * Empty/whitespace-only passes do not create a new visible segment.
 */
export function appendVisibleToolContent(previous: string, current: string): string {
  if (!current.trim()) return previous;
  return previous ? previous + "\n\n" + current : current;
}

/** Hold separator-only prefixes until a pass produces visible content.
 * Tool-only passes discard those separators at the boundary, so publishing them
 * early would require retracting the renderer's accumulated answer afterward.
 * Once content exists, retain every byte, including leading code indentation.
 */
export function visibleToolStreamContent(previous: string, current: string): string | null {
  return current.trim() ? appendVisibleToolContent(previous, current) : null;
}

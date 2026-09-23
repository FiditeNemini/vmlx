/** Preserve bytes already streamed before a tool boundary or continuation.
 * Trimming a segment here rewrites the visible answer and its tool offsets.
 * Empty/whitespace-only passes do not create a new visible segment.
 */
export function appendVisibleToolContent(previous: string, current: string): string {
  if (!current.trim()) return previous;
  return previous ? previous + "\n\n" + current : current;
}

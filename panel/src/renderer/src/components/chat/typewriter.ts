/**
 * Whether `next` only appends to the text the typewriter has already shown.
 *
 * The typewriter reveals a prefix of the content. Anything else — a shorter
 * string, or a final content that differs from what is on screen (completion
 * trims leading whitespace: a stream showing "\n\n4" can complete as "482",
 * the SAME length) — must snap to `next`, or the bubble keeps the stale text
 * until the chat re-renders.
 */
export function isTypewriterAppend(shown: string, next: string): boolean {
  return next.length >= shown.length && next.startsWith(shown)
}

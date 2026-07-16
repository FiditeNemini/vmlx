export type ReasoningSegments = string[]

export function appendReasoningDelta(
  segments: ReasoningSegments,
  delta: string,
): ReasoningSegments {
  if (!delta) return segments
  const next = [...segments]
  if (next.length === 0) {
    next.push(delta)
  } else {
    next[next.length - 1] += delta
  }
  return next
}

export function markReasoningToolBoundary(
  segments: ReasoningSegments,
): ReasoningSegments {
  if (segments.length === 0) return segments
  const next = [...segments]
  const last = next[next.length - 1]
  if (last && last.trim().length > 0) {
    next.push('')
  }
  return next
}

const RAW_TOOL_CONTROL_MARKUP =
  /<\s*\/?\s*(?:tool_call|function|parameter)(?:\s|=|:|>)|<\s*\/?\s*minimax:tool_call\b|<\|tool_call|\[Calling tool:|\bto=[A-Za-z_][\w.]*\s+code\{/i

/**
 * Reconcile a Responses API reasoning-summary `done` event with the deltas
 * already shown for the current reasoning segment.
 *
 * Local tool streams intentionally stop emitting reasoning deltas once native
 * tool markup begins, but the terminal event can still carry a longer cleaned
 * reasoning summary. Adopt it only when it extends the streamed prefix and has
 * no raw tool-control syntax. Earlier reasoning segments from prior tool
 * iterations remain untouched.
 */
export function reconcileReasoningSummaryDone(
  segments: ReasoningSegments,
  authoritativeText: unknown,
): ReasoningSegments {
  if (typeof authoritativeText !== 'string' || !authoritativeText.trim()) {
    return segments
  }
  if (RAW_TOOL_CONTROL_MARKUP.test(authoritativeText)) return segments

  const next = [...segments]
  if (next.length > 0 && !next[next.length - 1].trim()) {
    next[next.length - 1] = authoritativeText
    return next
  }

  let index = next.length - 1
  while (index >= 0 && !next[index].trim()) index--

  if (index < 0) return [authoritativeText]

  const streamedPrefix = next[index]
  if (
    authoritativeText === streamedPrefix ||
    !authoritativeText.startsWith(streamedPrefix)
  ) {
    return segments
  }

  next[index] = authoritativeText
  return next
}

export function visibleReasoningSegments(
  segments?: ReasoningSegments | null,
): ReasoningSegments {
  return (segments || []).filter((segment) => segment.trim().length > 0)
}

export function reasoningSegmentsForDisplay(
  segments?: ReasoningSegments | null,
  options?: { liveReplace?: boolean },
): ReasoningSegments {
  const visible = visibleReasoningSegments(segments)
  if (options?.liveReplace && visible.length > 1) {
    return [visible[visible.length - 1]]
  }
  return visible
}

export function joinReasoningSegments(
  segments?: ReasoningSegments | null,
): string {
  return visibleReasoningSegments(segments).join('\n\n')
}

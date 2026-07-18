/**
 * Responses API `warnings` field extractor.
 *
 * vMLX extends the OpenAI Responses API schema with a non-fatal
 * `warnings: string[] | null` field on `ResponsesObject`. It surfaces when:
 *   - `previous_response_id` chained a response that produced reasoning only
 *     (no visible message, no tool calls). Chat continuity may be impaired
 *     and prefix-cache reuse may be lower than expected.
 *   - (future) other coherence/observability signals
 *
 * This helper is pure and shared by the main process and renderer so warning
 * parsing stays identical across IPC, persistence, and UI rendering.
 */

export interface ResponsesPayloadLike {
  type?: unknown
  warnings?: unknown
  message?: unknown
  // Other fields are present but not required for warning extraction.
}

export const OUTPUT_TRUNCATED_WARNING =
  'Output truncated because the maximum output-token limit was reached. Increase Max Tokens or send a follow-up message to continue.'

/**
 * Preserve a terminal length stop as response metadata, not assistant text.
 *
 * Both Chat Completions and Responses surface the same `length` finish reason
 * to the panel main process. Persisting one deduplicated warning keeps the
 * terminal state truthful after the renderer reloads the SQLite-backed row.
 */
export function appendOutputTruncationWarning(
  warnings: string[] | null,
  finishReason: unknown,
): string[] | null {
  if (finishReason !== 'length') return warnings
  return Array.from(new Set([...(warnings || []), OUTPUT_TRUNCATED_WARNING]))
}

/**
 * Extract a clean `string[]` of warnings from a Responses API payload.
 *
 * Returns `null` (NOT `[]`) when no warnings are present — caller can use
 * truthy check to decide whether to render any UI. Returns deduplicated,
 * trimmed, non-empty strings only.
 */
export function extractResponsesWarnings(
  payload: ResponsesPayloadLike | null | undefined,
): string[] | null {
  if (!payload) return null
  const raw = payload.warnings
  if (payload.type === 'response.warning' && typeof payload.message === 'string') {
    const text = payload.message.trim()
    return text ? [text] : null
  }
  if (!Array.isArray(raw)) return null
  const seen = new Set<string>()
  const cleaned: string[] = []
  for (const item of raw) {
    if (typeof item !== 'string') continue
    const text = item.trim()
    if (!text) continue
    if (seen.has(text)) continue
    seen.add(text)
    cleaned.push(text)
  }
  return cleaned.length > 0 ? cleaned : null
}

/**
 * Remove warnings that describe an intermediate empty/reasoning-only response
 * after the panel's bounded post-tool recovery produced visible content.
 *
 * This is intentionally narrow. Cache, parser, schema, tool-drop, and
 * previous_response_id warnings remain visible. Only the current-response
 * empty-answer diagnostics are superseded by a successful recovery turn.
 */
export function dropSupersededRecoveryWarnings(
  warnings: string[] | null,
): string[] | null {
  if (!warnings) return null
  const kept = warnings.filter((warning) => {
    const normalized = warning.trim().toLowerCase()
    if (
      normalized.startsWith(
        'this response produced reasoning only (no visible message, no tool calls).',
      ) && normalized.includes('visible answer is empty')
    ) {
      return false
    }
    return normalized !== 'the model produced no visible response.'
  })
  return kept.length > 0 ? kept : null
}

/**
 * Map a Responses warning to a stable category key, useful for UI styling
 * or analytics. Falls back to "other" for unrecognized warnings so the UI
 * always has something to render.
 */
export function categorizeResponsesWarning(warning: string): string {
  const lower = warning.toLowerCase()
  if (lower.includes('reasoning only') || lower.includes('previous_response_id')) {
    return 'chain_reasoning_only'
  }
  if (lower.includes('no visible response') || lower.includes('empty_model_response')) {
    return 'empty_visible_response'
  }
  if (lower.includes('cache') || lower.includes('prefix')) {
    return 'cache_alignment'
  }
  return 'other'
}

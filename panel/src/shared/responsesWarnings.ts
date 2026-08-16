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

export interface ResponsesIncompleteDetailsLike {
  reason?: unknown
}

export const OUTPUT_TRUNCATED_WARNING =
  'Output truncated because the maximum output-token limit was reached. Increase Max Tokens or send a follow-up message to continue.'

/**
 * Translate a Responses terminal envelope without treating every incomplete
 * response as output truncation.
 */
export function responsesTerminalFinishReason(
  status: unknown,
  incompleteDetails?: ResponsesIncompleteDetailsLike | null,
): string | undefined {
  if (status === 'completed') return 'stop'
  if (status === 'incomplete') {
    const reason =
      typeof incompleteDetails?.reason === 'string'
        ? incompleteDetails.reason.trim().toLowerCase()
        : ''
    if (reason === 'max_output_tokens') return 'length'
    if (
      reason === 'cancelled' ||
      reason === 'canceled' ||
      reason === 'aborted' ||
      reason === 'abort'
    ) {
      return 'cancelled'
    }
    if (reason === 'error') return 'error'
    return 'incomplete'
  }
  if (status === 'failed') return 'error'
  return typeof status === 'string' && status.trim() ? status : undefined
}

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
  if (lower.startsWith('reasoning effort ')) {
    return 'effort_substitution'
  }
  if (lower.startsWith('model context limit')) {
    return 'context_exhaustion'
  }
  if (lower.includes('cache') || lower.includes('prefix')) {
    return 'cache_alignment'
  }
  return 'other'
}

/**
 * Structured notice records the engine attaches additively (#175):
 * `effort_substitution` on chat terminal chunks / Responses terminal
 * snapshots, and `context_exhaustion` top-level on chat responses or inside
 * `incomplete_details` on Responses length terminals. These builders turn
 * them into the per-message warning strings the existing rail renders and
 * persists; both return null on malformed/absent records so callers can
 * merge unconditionally.
 */

export interface EffortSubstitutionLike {
  requested_effort?: unknown
  effective_effort?: unknown
  stamped_levels?: unknown
}

export interface ContextExhaustionLike {
  prompt_tokens?: unknown
  requested_max_tokens?: unknown
  clamped_max_tokens?: unknown
  declared_context_tokens?: unknown
  /** Chat-dialect doors set this: true = the clamped budget was consumed
   * (a length stop AT the ceiling). false = clamped at admission but the
   * turn completed inside the budget. The Responses length terminal omits
   * it (that record only surfaces on a length stop — treat as exhausted). */
  exhausted?: unknown
}

export function effortSubstitutionNotice(
  record: EffortSubstitutionLike | null | undefined,
): string | null {
  if (!record) return null
  const requested =
    typeof record.requested_effort === 'string' ? record.requested_effort.trim() : ''
  const effective =
    typeof record.effective_effort === 'string' ? record.effective_effort.trim() : ''
  if (!requested || !effective) return null
  const levels = Array.isArray(record.stamped_levels)
    ? record.stamped_levels.filter((l): l is string => typeof l === 'string' && !!l.trim())
    : []
  const supported = levels.length > 0 ? ` (supported: ${levels.join(', ')})` : ''
  return (
    `Reasoning effort "${requested}" is not supported by this model; ` +
    `it ran at "${effective}"${supported}.`
  )
}

export function contextExhaustionNotice(
  record: ContextExhaustionLike | null | undefined,
): string | null {
  if (!record) return null
  const prompt = Number(record.prompt_tokens)
  const requested = Number(record.requested_max_tokens)
  const clamped = Number(record.clamped_max_tokens)
  const declared = Number(record.declared_context_tokens)
  if (!Number.isFinite(prompt) || !Number.isFinite(declared) || declared <= 0) {
    return null
  }
  const clampedPart =
    Number.isFinite(requested) && Number.isFinite(clamped)
      ? ` Output was capped at ${clamped.toLocaleString('en-US')} tokens ` +
        `(requested ${requested.toLocaleString('en-US')}).`
      : ''
  // Strong wording only when the clamped budget was actually consumed
  // (exhausted true, or absent — the Responses length terminal omits the
  // flag and only surfaces on a length stop). A clamped-but-completed turn
  // gets an informational capacity notice instead of a false alarm.
  if (record.exhausted === false) {
    return (
      `Model context limit pressure: the prompt (${prompt.toLocaleString('en-US')} tokens) ` +
      `leaves limited room in the model's ${declared.toLocaleString('en-US')}-token context.` +
      clampedPart +
      ' This turn completed within the cap; if a later response cuts off there, shorten the chat history or serve a larger-context bundle.'
    )
  }
  return (
    `Model context limit reached: the prompt (${prompt.toLocaleString('en-US')} tokens) ` +
    `fills part of the model's ${declared.toLocaleString('en-US')}-token context.` +
    clampedPart +
    ' Generation ran to the context ceiling — shorten the chat history or serve a larger-context bundle.'
  )
}

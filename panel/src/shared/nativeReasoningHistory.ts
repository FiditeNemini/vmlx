/**
 * Registry family names whose native templates preserve historical reasoning (GLM-5.3: `clear_thinking` defaults to
 * false, so every earlier assistant turn keeps its <think> block). For these, everything the app renders BEFORE the
 * current user message must not depend on the current message's wording: a "without tools" turn that dropped the
 * replayed reasoning or the leading tool-epoch note rewrote the whole history and re-prefilled it (GLM-5.3, 31.6k-token
 * chat: cached 0, 78 s TTFT, 2026-10-12).
 */
const HISTORY_STABLE_FAMILIES = new Set(['naive_n05_flash', 'qwen3.5', 'qwen4-exp', 'glm5-next'])

export function historyIsStableAcrossToolAuthorization(detectedFamily: string | undefined): boolean {
  return detectedFamily !== undefined && HISTORY_STABLE_FAMILIES.has(detectedFamily)
}

/** Current-turn tool authorization must not rewrite those prior assistant tokens. */
export function shouldReplayHistoricalReasoning(
  detectedFamily: string | undefined,
  currentPromptForbidsTools: boolean,
): boolean {
  return historyIsStableAcrossToolAuthorization(detectedFamily) || !currentPromptForbidsTools
}

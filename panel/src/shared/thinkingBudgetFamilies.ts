/**
 * ONE list of which families honour a top-level `max_thinking_tokens` cap on
 * the reasoning pass.
 *
 * This is not cosmetic. The engine's never-empty answer pass is ARMED by a
 * capped first pass: without a cap the model can spend the entire budget inside
 * <think> and return an EMPTY answer at finish=length, and the salvage never
 * runs. The only way a user supplies that cap from the app is the "Max Thinking
 * Tokens" field in Chat Settings — and that field is rendered only when the
 * family's `supportsThinkingBudget` is true. So a family the engine caps but
 * the panel does not list is a family whose fix the user cannot reach.
 *
 * MEASURED 2026-08-12: nemotron / nemotron-h had been in the engine's
 * `_THINKING_BUDGET_CAP_FAMILIES` since 2026-08-11, but no panel row set
 * supportsThinkingBudget, so the Chat Settings drawer for a live Nemotron
 * session showed "Enable Thinking" and NO budget field. Same for nanbeige.
 *
 * KEYS ARE PANEL REGISTRY NAMES, not engine family_name — the registry maps
 * engine qwen3_5 -> "qwen3.5", qwen3_next -> "qwen3-next", nemotron_h ->
 * "nemotron-h", gemma4_text -> "gemma4-text" (model-config-registry.ts
 * MODEL_TYPE_TO_FAMILY), while minimax_m3 and openpangu_v2 keep their
 * underscores because those ARE their registry names. Two earlier fixes in this
 * project were silent no-ops for exactly this reason, so every key below was
 * taken from the registry individually rather than transliterated.
 *
 * tests/test_cli_panel_settings_parity.py asserts this list and the engine's
 * `_THINKING_BUDGET_CAP_FAMILIES` stay in step across that translation.
 *
 * Deliberately ABSENT, and why:
 *  - deepseek-v4 and step-3.7-flash key on reasoning_effort, not a token
 *    budget, so max_thinking_tokens is inert for them; showing the field would
 *    be untruthful. (The engine subtracts deepseek_v4 from its own cap set for
 *    the same reason.)
 *  - muse-glimmer's template reads neither enable_thinking nor a budget marker;
 *    its only live control is `reasoning_strength`.
 */

export const THINKING_BUDGET_FAMILIES: readonly string[] = Object.freeze([
  'gemma4',
  'gemma4-text',
  'hy3',
  'laguna',
  'minimax',
  'minimax_m3',
  'nanbeige',
  'nemotron',
  'nemotron-h',
  'openpangu_v2',
  'qwen3',
  'qwen3.5',
  'qwen3.5-moe',
  'qwen3-next',
])

const THINKING_BUDGET_FAMILY_SET = new Set(THINKING_BUDGET_FAMILIES)

/** Whether the app should offer a Max Thinking Tokens control for a family. */
export function familySupportsThinkingBudget(registryFamily?: string): boolean {
  return !!registryFamily && THINKING_BUDGET_FAMILY_SET.has(registryFamily)
}

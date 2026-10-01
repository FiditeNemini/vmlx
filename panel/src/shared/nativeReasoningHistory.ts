/**
 * Registry family names whose native templates preserve historical reasoning.
 * Current-turn tool authorization must not rewrite those prior assistant tokens.
 */
export function shouldReplayHistoricalReasoning(
  detectedFamily: string | undefined,
  currentPromptForbidsTools: boolean,
): boolean {
  return detectedFamily === 'naive_n05_flash' ||
    detectedFamily === 'qwen3.5' ||
    detectedFamily === 'qwen4-exp' ||
    !currentPromptForbidsTools
}

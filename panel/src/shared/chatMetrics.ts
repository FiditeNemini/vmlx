export interface FinalDecodeTpsInput {
  cumulativeTps: number
  rollingTps: number[]
  lastRollingTps?: number
  maxCumulativeToRollingRatio?: number
}

export interface PrefillTpsInput {
  promptTokens: number
  cachedTokens: number
  ttftSeconds: number
  serverUsageKnown: boolean
}

/**
 * Calculate prompt-processing throughput for the uncached prefill only.
 *
 * Prompt and cache counts must come from authoritative server usage for the
 * same HTTP pass whose TTFT is supplied. Client-estimated prompt counts and
 * exchange-wide tool-loop totals cannot be paired truthfully with one pass's
 * TTFT, so no rate is returned until server usage is known.
 */
export function calculatePrefillTps({
  promptTokens,
  cachedTokens,
  ttftSeconds,
  serverUsageKnown,
}: PrefillTpsInput): string | undefined {
  if (!serverUsageKnown) return undefined
  if (!Number.isFinite(promptTokens) || promptTokens <= 0) return undefined
  if (!Number.isFinite(ttftSeconds) || ttftSeconds <= 0.001) return undefined

  const safeCachedTokens = Number.isFinite(cachedTokens)
    ? Math.min(Math.max(cachedTokens, 0), promptTokens)
    : 0
  const uncachedPromptTokens = Math.max(promptTokens - safeCachedTokens, 0)
  if (uncachedPromptTokens <= 0) return undefined

  return (uncachedPromptTokens / ttftSeconds).toFixed(1)
}

/**
 * Select a truthful final decode rate for a possibly multi-request agent turn.
 *
 * Cumulative delta timing represents every reasoning/tool iteration, but a
 * server-side buffered pass can make it look impossibly fast. The last rolling
 * rate avoids that burst but can represent only a short final-answer tail after
 * a pause. Use the median rolling rate as the representative control: retain
 * cumulative throughput when it agrees with the progressive stream, otherwise
 * reject the buffered burst.
 */
export function selectFinalDecodeTps({
  cumulativeTps,
  rollingTps,
  lastRollingTps = 0,
  maxCumulativeToRollingRatio = 1.5,
}: FinalDecodeTpsInput): number {
  const samples = rollingTps
    .filter(value => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b)

  const representativeRolling = samples.length > 0
    ? samples.length % 2 === 1
      ? samples[Math.floor(samples.length / 2)]
      : (samples[samples.length / 2 - 1] + samples[samples.length / 2]) / 2
    : Number.isFinite(lastRollingTps) && lastRollingTps > 0
      ? lastRollingTps
      : 0

  const validCumulative = Number.isFinite(cumulativeTps) && cumulativeTps > 0
    ? cumulativeTps
    : 0
  if (representativeRolling <= 0) return validCumulative
  if (validCumulative <= 0) return representativeRolling

  return validCumulative <= representativeRolling * maxCumulativeToRollingRatio
    ? validCumulative
    : representativeRolling
}

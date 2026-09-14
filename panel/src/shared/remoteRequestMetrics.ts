/**
 * Remote providers need not send one token per event, incremental usage, or
 * engine-side decode timing. Count only their request-local usage. The fallback
 * rate is observed output / HTTP-request time, NOT model decode throughput:
 * it includes TTFT, network and terminal buffering, but excludes tool execution.
 */
export class RemoteRequestMetrics {
  private passes: Array<{ start: number; end?: number; tokens?: number }> = []

  beginPass(now: number): void {
    this.endPass(now)
    this.passes.push({ start: now })
  }

  recordUsage(usage: unknown): void {
    const pass = this.passes[this.passes.length - 1]
    if (!pass || pass.end !== undefined || !usage || typeof usage !== 'object') return
    const value = usage as Record<string, unknown>
    const count = value.output_tokens ?? value.completion_tokens
    if (typeof count !== 'number' || !Number.isSafeInteger(count) || count < 0) return
    pass.tokens = Math.max(pass.tokens ?? 0, count)
  }

  endPass(now: number): void {
    const pass = this.passes[this.passes.length - 1]
    if (pass && pass.end === undefined) pass.end = Math.max(pass.start, now)
  }

  snapshot(now: number): {
    outputTokens?: number
    tokensPerSecond?: number
    requestSeconds: number
    passes: number
  } {
    const requestSeconds = this.passes.reduce(
      (sum, pass) => sum + Math.max(0, (pass.end ?? now) - pass.start) / 1000, 0,
    )
    // A complete exchange must not present the known subset as its total.
    const known = this.passes.length > 0 && this.passes.every(pass => pass.tokens !== undefined)
    const outputTokens = known
      ? this.passes.reduce((sum, pass) => sum + pass.tokens!, 0)
      : undefined
    return {
      outputTokens,
      tokensPerSecond: outputTokens !== undefined && requestSeconds > 0
        ? outputTokens / requestSeconds : undefined,
      requestSeconds,
      passes: this.passes.length,
    }
  }
}

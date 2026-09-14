import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { RemoteRequestMetrics } from '../src/shared/remoteRequestMetrics'
import { getMetricsItems } from '../src/renderer/src/components/chat/chat-utils'

describe('remote terminal-only usage through the production parser', () => {
  const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
  const marker = 'if (isResponsesTerminalEvent && respUsage) {'
  const start = source.indexOf(marker)
  const end = source.indexOf('if (respUsage.input_tokens != null)', start)
  if (start < 0 || end < 0) throw new Error('Production terminal usage block missing')
  // Exercise the small production block without importing Electron or making
  // a request. The fixture uses retained real per-request usage, not estimated
  // SSE chunk counts. Recorders do not affect its token arithmetic.
  const apply = new Function('respUsage', 'iterationTokenBase', `
    let tokenCount = 0, iterationTokenCount = 0, serverSendsUsage = false, tpsTokenBase = 0;
    const tpsSnapshots = [];
    const recordServerDecodeUsage = () => {};
    const remoteMetrics = undefined;
    ${source.slice(start + marker.length, end)}
    return { tokenCount, iterationTokenCount };
  `)

  it.each([[230, 453], [221, 149], [70, 70], [0, 5]])(
    'sums request-local usage %i + %i without guessing a restart from magnitudes',
    (first, second) => {
      const a = apply({ output_tokens: first }, 0)
      const b = apply({ output_tokens: second }, a.tokenCount)
      expect(a.iterationTokenCount + b.iterationTokenCount).toBe(first + second)
    },
  )
})

describe('remote observed request throughput', () => {
  it('sums usage and HTTP windows while excluding a slow local tool', () => {
    const metrics = new RemoteRequestMetrics()
    metrics.beginPass(100)
    metrics.recordUsage({ output_tokens: 230 })
    metrics.endPass(1100)
    metrics.beginPass(11100)
    metrics.recordUsage({ output_tokens: 453 })
    metrics.endPass(14100)
    expect(metrics.snapshot(99000)).toEqual({
      outputTokens: 683, tokensPerSecond: 683 / 4, requestSeconds: 4, passes: 2,
    })
  })

  it('does not mistake events, omitted usage, or known subsets for total tokens', () => {
    const metrics = new RemoteRequestMetrics()
    metrics.beginPass(0)
    metrics.recordUsage({ prompt_tokens: 100 })
    expect(metrics.snapshot(1000).outputTokens).toBeUndefined()
    metrics.recordUsage({ completion_tokens: 70 })
    metrics.endPass(1000)
    expect(metrics.snapshot(5000).tokensPerSecond).toBe(70)
    metrics.beginPass(5000)
    expect(metrics.snapshot(6000)).toMatchObject({
      outputTokens: undefined, tokensPerSecond: undefined, requestSeconds: 2,
    })
  })

  it('treats incremental usage as cumulative within, not across, HTTP passes', () => {
    const metrics = new RemoteRequestMetrics()
    metrics.beginPass(0)
    for (const completion_tokens of [1, 8, 8, 16, 32]) metrics.recordUsage({ completion_tokens })
    metrics.endPass(1000)
    metrics.beginPass(3000)
    metrics.recordUsage({ completion_tokens: 32 })
    metrics.endPass(4000)
    expect(metrics.snapshot(8000).outputTokens).toBe(64)
    expect(metrics.snapshot(8000).tokensPerSecond).toBe(32)
  })

  it.each([null, {}, { output_tokens: -1 }, { output_tokens: 1.5 },
    { output_tokens: '42' }, { completion_tokens: Number.NaN },
    { output_tokens: Number.POSITIVE_INFINITY }])('rejects invalid or missing usage %j', usage => {
    const metrics = new RemoteRequestMetrics()
    metrics.beginPass(0)
    metrics.recordUsage(usage)
    expect(metrics.snapshot(1000).outputTokens).toBeUndefined()
  })

  it('keeps explicit zero distinct from unknown, and handles abort without new usage', () => {
    const metrics = new RemoteRequestMetrics()
    metrics.beginPass(100)
    metrics.recordUsage({ output_tokens: 0 })
    metrics.endPass(500)
    expect(metrics.snapshot(1500).tokensPerSecond).toBe(0)
    metrics.beginPass(2000)
    metrics.endPass(2500)
    metrics.endPass(9000)
    expect(metrics.snapshot(99000)).toMatchObject({
      outputTokens: undefined, requestSeconds: 0.9, tokensPerSecond: undefined,
    })
  })

  it('cannot turn buffered single-event output into a near-zero-duration decode burst', () => {
    const metrics = new RemoteRequestMetrics()
    metrics.beginPass(0)
    metrics.recordUsage({ output_tokens: 400 })
    metrics.endPass(10000)
    expect(metrics.snapshot(10000).tokensPerSecond).toBe(40)
  })

  it('does not alter the existing local decode or prefill helpers', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    expect(source).toContain('isRemote ? new RemoteRequestMetrics() : undefined')
    expect(source).toContain('if (!remoteMetrics) return {}')
    expect(source.match(/remoteMetrics\?\.beginPass\(/g)).toHaveLength(2)
    expect(source.match(/remoteMetrics\?\.endPass\(/g)).toHaveLength(3)
    expect(source.match(/\.\.\.remoteMetricFields\(\)/g)).toHaveLength(6)
  })
})

describe('remote/local metric presentation', () => {
  const t = (key: string, args?: Record<string, string | number>) => `${key}:${JSON.stringify(args ?? {})}`
  const base = { tokenCount: 40, tokensPerSecond: '20.0', ttft: '0.25' }

  it('retains the existing local rate and label without remote metadata', () => {
    const items = getMetricsItems(base, false, t)
    expect(items[1]).toMatchObject({ label: '20.0 t/s', title: 'chat.metrics.tpsTitle:{}' })
  })

  it('labels the observed remote window and missing provider usage', () => {
    const observed = getMetricsItems({ ...base, decodeMetricSource: 'remote-request' }, false, t)
    expect(observed[1].title).toBe('chat.metrics.remoteRateTitle:{}')
    const unknown = getMetricsItems({ ...base, tokenCountKnown: false,
      tokensPerSecond: '—', decodeMetricSource: 'unavailable' }, true, t)
    expect(unknown[0].value).toBe('—')
    expect(unknown[1].title).toBe('chat.metrics.remoteUsageUnavailable:{}')
  })

  it.each(['en', 'es', 'ja', 'ko', 'zh'])('provides the new labels in %s', locale => {
    const metrics = JSON.parse(readFileSync(`src/renderer/src/i18n/locales/${locale}.json`, 'utf8')).chat.metrics
    for (const key of ['remoteRateLabel', 'remoteRateTitle', 'remoteUsageUnavailable']) {
      expect(typeof metrics[key]).toBe('string')
      expect(metrics[key].length).toBeGreaterThan(8)
    }
  })
})

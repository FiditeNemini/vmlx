import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  calculatePrefillTps,
  selectFinalDecodeTps,
} from '../src/shared/chatMetrics'

describe('chat prefill TPS', () => {
  it('uses only the uncached prompt tail', () => {
    expect(
      calculatePrefillTps({
        promptTokens: 939,
        cachedTokens: 832,
        ttftSeconds: 0.18,
        serverUsageKnown: true,
      }),
    ).toBe('594.4')
  })

  it('clamps cache counts and rejects requests with no uncached prefill', () => {
    expect(
      calculatePrefillTps({
        promptTokens: 128,
        cachedTokens: 256,
        ttftSeconds: 0.25,
        serverUsageKnown: true,
      }),
    ).toBeUndefined()
  })

  it('does not report a rate before authoritative server usage', () => {
    expect(
      calculatePrefillTps({
        promptTokens: 939,
        cachedTokens: 0,
        ttftSeconds: 0.18,
        serverUsageKnown: false,
      }),
    ).toBeUndefined()
  })

  it('rejects invalid token counts and TTFT', () => {
    expect(
      calculatePrefillTps({
        promptTokens: Number.NaN,
        cachedTokens: 0,
        ttftSeconds: 0.18,
        serverUsageKnown: true,
      }),
    ).toBeUndefined()
    expect(
      calculatePrefillTps({
        promptTokens: 100,
        cachedTokens: 0,
        ttftSeconds: 0.001,
        serverUsageKnown: true,
      }),
    ).toBeUndefined()
  })

  it('routes live, final, and abort prefill metrics through the shared helper', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    expect(source.match(/calculatePrefillTps\(\{/g)).toHaveLength(3)
    expect(source).not.toMatch(/promptTokens\s*\/\s*ttft/)
    expect(source).not.toMatch(/promptTokens\s*\/\s*abortTtft/)
    expect(source).toContain('finalStreamCachedTokens')
  })
})

describe('final chat decode TPS', () => {
  it('keeps cumulative multi-iteration throughput when only the final tail is slow', () => {
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 49.6,
        rollingTps: [48.8, 49.1, 50.3, 49.4, 8.3],
        lastRollingTps: 8.3,
      }),
    ).toBe(49.6)
  })

  it('rejects an impossible cumulative burst from buffered output', () => {
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 261,
        rollingTps: [42.7, 42.9, 43.1],
        lastRollingTps: 43.1,
      }),
    ).toBe(42.9)
  })

  it('falls back cleanly when only one timing source is available', () => {
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 0,
        rollingTps: [],
        lastRollingTps: 37.5,
      }),
    ).toBe(37.5)
    expect(
      selectFinalDecodeTps({
        cumulativeTps: 31.25,
        rollingTps: [],
      }),
    ).toBe(31.25)
  })
})

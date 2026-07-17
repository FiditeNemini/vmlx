import { describe, expect, it } from 'vitest'
import { selectFinalDecodeTps } from '../src/shared/chatMetrics'

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

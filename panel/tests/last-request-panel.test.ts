import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { lastRequestItems, type LastRequest } from '../src/renderer/src/components/sessions/PerformancePanel'
import en from '../src/renderer/src/i18n/locales/en.json'

// The Perf panel's "Last request" cards render the SAME per-lane records the engine producer test writes
// (tests/test_request_telemetry.py -> tests/fixtures/last_request/*.json). A key renamed on either side fails one of
// the two tests; that is the contract that keeps this panel from silently breaking per family/quant edits.
const FIXTURES = join(__dirname, '..', '..', 'tests', 'fixtures', 'last_request')

function t(key: string, params?: Record<string, string | number>): string {
  const value = key.split('.').reduce<any>((node, part) => node?.[part], en)
  if (typeof value !== 'string') throw new Error(`missing i18n key ${key}`)
  return value.replace(/\{(\w+)\}/g, (_, name) => String(params?.[name] ?? `{${name}}`))
}

function fixture(lane: string): LastRequest {
  return JSON.parse(readFileSync(join(FIXTURES, `${lane}.json`), 'utf8'))
}

function cards(lane: string): Record<string, string> {
  return Object.fromEntries(lastRequestItems(fixture(lane), t).map(item => [item.key, item.value]))
}

describe('Perf panel last-request cards (shared engine fixtures)', () => {
  it('covers every lane fixture the engine test writes', () => {
    const lanes = readdirSync(FIXTURES).filter(f => f.endsWith('.json')).map(f => f.replace(/\.json$/, '')).sort()
    expect(lanes).toEqual(['batched-llm', 'batched-mllm', 'batched-mllm-mtp', 'simple-dflash2-glm', 'simple-dflash2-qwen', 'simple-llm-no-receipt'])
    for (const lane of lanes) {
      const items = lastRequestItems(fixture(lane), t)
      // Every card always renders (absent figures read "not reported", never hidden).
      expect(items.map(item => item.key)).toEqual(['decode', 'prefill', 'cachedPrefill', 'ttft', 'prompt', 'speculative', 'request'])
      for (const item of items) expect(item.value, `${lane}:${item.key}`).not.toMatch(/undefined|NaN|null/)
    }
  })

  it('shows t/s, cold pp/s, cached pp/s and TTFT for a batched prefix hit', () => {
    const c = cards('batched-llm')
    expect(c.decode).toBe('50.0 t/s (100 tok / 2.00 s)')
    expect(c.prefill).toBe('2048.0 pp/s (4096 tok)')
    expect(c.cachedPrefill).toBe('16384.0 cached pp/s')
    expect(c.ttft).toBe('0.50 s')
    expect(c.prompt).toBe('8192 (4096 block-disk cached)')
    expect(c.speculative).toBe('none')
    expect(c.request).toBe('batched · llm · stream_chat · stop')
  })

  it('shows DFlash2 acceptance, emitted per cycle, AR tokens and widths', () => {
    expect(cards('simple-dflash2-qwen').speculative).toBe('dflash2 (qwen) · accept 23.0% · 3.29/cycle · AR 192 tok · widths 5:11 8:37 16:10')
    expect(cards('simple-dflash2-glm').speculative).toBe('dflash2 (glm) · accept 27.5% · 2.10/cycle · AR 64 tok · widths 4:40')
    expect(cards('simple-dflash2-glm').prefill).toBe('231.6 pp/s (359 tok · glm_dflash2)')
  })

  it('shows native MTP acceptance and final depth (Qwen Flash-Next lane; Naive has no MTP)', () => {
    expect(cards('batched-mllm-mtp').speculative).toBe('mtp · accept 70.0% · D2')
    expect(cards('batched-llm').speculative).toBe('none')
  })

  it('says "not reported" for a lane without a prefill receipt and "no cache hit" when cold', () => {
    const c = cards('simple-llm-no-receipt')
    expect(c.prefill).toBe('not reported')
    expect(c.cachedPrefill).toBe('no cache hit')
    expect(cards('batched-mllm').cachedPrefill).toBe('no cache hit')
  })
})

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const form = readFileSync(
  resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx'), 'utf-8')

/**
 * step3p7_full_sliding_kv was listed as mixed-SWA in three of four places:
 * mixedSwaBlockDiskOnlySupported, subtypeRequiresPagedCache and the ENGINE's
 * own detector (mllm_scheduler returns true for it alongside mixed_swa_kv) —
 * but NOT in mixedSwaCacheActive, which drives the cache codec badge and its
 * description. Step-3.7 therefore showed a generic "AUTO" badge with the
 * engine-native copy, for a family everything else treats as mixed-SWA.
 */
describe('mixed-SWA subtype membership is consistent across the form', () => {
  const blockFor = (name: string) => {
    const start = form.indexOf(`const ${name} =`)
    expect(start, `${name} not found`).toBeGreaterThan(-1)
    return form.slice(start, form.indexOf('\n  const ', start + 10))
  }

  it.each([
    'mixedSwaBlockDiskOnlySupported',
    'subtypeRequiresPagedCache',
    'mixedSwaCacheActive',
  ])('%s includes step3p7_full_sliding_kv', (name) => {
    expect(blockFor(name)).toContain("'step3p7_full_sliding_kv'")
  })

  it.each([
    'mixedSwaBlockDiskOnlySupported',
    'subtypeRequiresPagedCache',
    'mixedSwaCacheActive',
  ])('%s includes mixed_swa_kv', (name) => {
    expect(blockFor(name)).toContain("'mixed_swa_kv'")
  })
})

describe('the engine agrees these are the mixed-SWA subtypes', () => {
  it('mllm_scheduler classifies both the same way', () => {
    const sched = readFileSync(
      resolve(__dirname, '../../vmlx_engine/mllm_scheduler.py'), 'utf-8')
    const start = sched.indexOf('"mixed_swa_kv"')
    const block = sched.slice(start - 200, start + 300)
    expect(block).toContain('"step3p7_full_sliding_kv"')
  })
})

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { compactQuantizationBadgeLabel } from '../src/renderer/src/lib/quantizationBadge'

/** DSV4-Flash's real label, from the Tools tab. */
const DSV4 =
  'JANG_2L_GS64_ProjLayerBits_Ggs64-Dgs32-Ugs64_Attn8g64_Tok8g64_NoMTP_AWQ_DiagImatrix_QAT_GPTQ'

describe('every surface that renders a quantization badge compacts it', () => {
  const surfaces = [
    'src/renderer/src/components/sessions/SessionCard.tsx',
    'src/renderer/src/components/tools/ToolsDashboard.tsx',
  ]

  it.each(surfaces)('%s uses the shared compactor', (rel) => {
    const source = readFileSync(resolve(__dirname, '..', rel), 'utf-8')
    expect(source).toContain('compactQuantizationBadgeLabel(')
  })

  it.each(surfaces)('%s keeps the full value in a tooltip', (rel) => {
    const source = readFileSync(resolve(__dirname, '..', rel), 'utf-8')
    // truncating without a title would DESTROY information rather than fold it
    expect(source).toMatch(/title=\{(model\.quantization|jangLabel)\}/)
  })
})

describe('the compactor on the label that motivated this', () => {
  it('folds the 90-char DSV4 label down', () => {
    expect(DSV4.length).toBeGreaterThan(80)
    const compact = compactQuantizationBadgeLabel(DSV4)
    expect(compact.length).toBeLessThan(24)
    expect(compact.endsWith('…')).toBe(true)
    // it must still start with the part that identifies the bundle
    expect(compact.startsWith('JANG_2L')).toBe(true)
  })

  it('leaves short labels completely alone', () => {
    expect(compactQuantizationBadgeLabel('MXFP4 (4b)')).toBe('MXFP4 (4b)')
    expect(compactQuantizationBadgeLabel('JANG_4M')).toBe('JANG_4M')
  })
})

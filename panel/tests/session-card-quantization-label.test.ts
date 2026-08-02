import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'
import { compactQuantizationBadgeLabel } from '../src/renderer/src/lib/quantizationBadge'

const source = readFileSync(
  join(__dirname, '../src/renderer/src/components/sessions/SessionCard.tsx'),
  'utf8',
)

describe('SessionCard quantization label truth', () => {
  it('replaces the path fallback with the bundle-grounded detector label', () => {
    expect(source).toContain('window.api.models.detectConfig(session.modelPath)')
    expect(source).toContain('detected?.quantizationLabel')
    expect(source).toContain('setJangLabel(detected.quantizationLabel)')
  })

  it('keeps JANGTQ distinct even while the bundle detector is pending', () => {
    expect(source).toContain('name.includes("jangtq")')
    expect(source).toContain('? "JANGTQ"')
  })

  it('does not classify an MXFP bundle from its provider directory name', () => {
    expect(source).toContain('const bundleName = session.modelPath.split("/").filter(Boolean).pop()')
    expect(source).toContain('const name = bundleName.toLowerCase()')
    expect(source).not.toContain('const name = session.modelPath.toLowerCase()')
  })

  it('keeps a long detector label recognizable instead of collapsing it to a codec name', () => {
    const fullLabel = 'JANG_2L_GS64_ProjLayerBits_Ggs64-Dgs32-Ugs64_Attn8g64_Tok8g64_NoMTP_AWQ_DiagImatrix'

    expect(compactQuantizationBadgeLabel(fullLabel)).toBe('JANG_2L_GS64…')
    expect(compactQuantizationBadgeLabel('JANGTQ2 (2b)')).toBe('JANGTQ2 (2b)')

    const compactUnsegmented = compactQuantizationBadgeLabel('FullDetectorLabelWithoutSeparators')
    expect(compactUnsegmented.endsWith('…')).toBe(true)
    expect(compactUnsegmented.length).toBeLessThanOrEqual(24)
  })

  it('keeps the full detector label accessible while constraining the visible badge', () => {
    expect(source).toContain('title={jangLabel}')
    expect(source).toContain('aria-label={`Quantization: ${jangLabel}`}')
    expect(source).toContain('{compactQuantizationBadgeLabel(jangLabel)}')
    expect(source).toContain('flex min-w-0 items-start justify-between gap-2')
    expect(source).toContain('flex-1 min-w-0 overflow-hidden')
    expect(source).toContain('min-w-0 max-w-[9rem] shrink truncate')
    expect(source).toContain('className="flex items-center gap-1.5 flex-shrink-0"')
  })
})

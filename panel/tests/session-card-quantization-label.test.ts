import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

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
})

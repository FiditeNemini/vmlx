import { describe, expect, it } from 'vitest'
import { formatJangQuantizationLabel } from '../src/shared/jangQuantization'

/**
 * MXFP bundles rendered with NO badge in the session list, which is
 * indistinguishable from "unquantized" — for a whole format family, in a list
 * where the badge is how you tell a 4-bit bundle from a ternary one.
 *
 * They are not missing metadata: they carry a real jang_config.json declaring
 * `profile: "MXFP4"` and `quantization.bits: 4`. The formatter only matched
 * jang/jangtq/mxtq/affine weight formats, and MXFP declares
 * `weight_format: "mlx"`, so it fell through to undefined.
 */
describe('MXFP quantization labels', () => {
  it('labels an MXFP4 bundle from its real jang_config shape', () => {
    // verbatim from Nemotron-Omni-Nano-MXFP4-CRACK/jang_config.json
    expect(formatJangQuantizationLabel({
      version: 2,
      weight_format: 'mlx',
      profile: 'MXFP4',
      quantization: { method: 'affine', group_size: 32, bits: 4 },
    } as never)).toBe('MXFP4 (4b)')
  })

  it('labels MXFP8 the same way', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mlx',
      profile: 'MXFP8',
      quantization: { bits: 8 },
    } as never)).toBe('MXFP8 (8b)')
  })

  it('shows the bare profile when no bit count is declared', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mlx', profile: 'MXFP4',
    } as never)).toBe('MXFP4')
  })

  it('still returns undefined with no profile — a blank beats an invented label', () => {
    expect(formatJangQuantizationLabel({ weight_format: 'mlx' } as never)).toBeUndefined()
    expect(formatJangQuantizationLabel({} as never)).toBeUndefined()
    expect(formatJangQuantizationLabel({
      weight_format: 'mlx', quantization: { bits: 4 },
    } as never)).toBeUndefined()
  })
})

describe('the existing formats are unchanged', () => {
  it('JANG affine still wins before the new fallback', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'jang', profile: 'JANG_4M',
      quantization: { actual_bits: 4 },
    } as never)).toBe('JANG_4M (4b)')
  })

  it('JANGTQ still reports its unsupported 1-bit suffix', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'mxtq', profile: 'JANGTQ1',
    } as never)).toBe('JANGTQ1 (1b, unsupported)')
  })

  it('routed-average labelling is preserved', () => {
    expect(formatJangQuantizationLabel({
      weight_format: 'jang', profile: 'JANG_4M',
      quantization: { routed_avg_bits: 4.0959 },
    } as never)).toBe('JANG_4M (4.1b routed)')
  })
})

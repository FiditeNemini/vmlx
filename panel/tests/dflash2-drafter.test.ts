import { describe, expect, it } from 'vitest'
import {
  QWEN38_27B_DFLASH2_REPO,
  configDeclaresDflash2,
  isQwen38Dense27b,
  looksLikeDflash2Drafter,
} from '../src/shared/dflash2Drafter'

describe('DFlash2 drafter helpers', () => {
  it('detects Qwen3.8 27B bundles by name or path', () => {
    for (const id of [
      'Qwen3.8-27B-JANG_4D /Volumes/EricsLLMDrive/jangq-ai/Qwen3.8-27B-JANG_4D',
      'qwen3_8-27b-jang_2d',
      '/models/Qwen3.8-27B-JANG_6D',
      'Qwen3-8-27B-MLX-4bit',
    ]) {
      expect(isQwen38Dense27b(id)).toBe(true)
    }
  })

  it('keeps other families out (no published Qwen3.8 drafter for them)', () => {
    for (const id of [
      'Qwen3.6-27B-JANG_4M-CRACK',
      'Qwen3.8-Flash-Next-JANG_4M',
      '/Users/eric/q38fn-jangh/run_v5/build/bundle',
      'Allosaurus-v0.1-125B-A6B-JANGH2',
      '',
      undefined,
    ]) {
      expect(isQwen38Dense27b(id as string)).toBe(false)
    }
  })

  it('recognizes drafters by config architecture and by name', () => {
    expect(configDeclaresDflash2({ architectures: ['DFlash2DraftModel'] })).toBe(true)
    expect(configDeclaresDflash2({ architectures: ['Qwen3ForCausalLM'] })).toBe(false)
    expect(configDeclaresDflash2(null)).toBe(false)
    expect(looksLikeDflash2Drafter('/Volumes/EricsLLMDrive/z-lab/Qwen3.8-27B-DFlash2')).toBe(true)
    expect(looksLikeDflash2Drafter('mlx-community/small-draft-model')).toBe(false)
    expect(QWEN38_27B_DFLASH2_REPO).toBe('z-lab/Qwen3.8-27B-DFlash2')
  })
})

describe('bundled DFlash2 drafter detection', () => {
  it('finds <bundle>/dflash2 by config architecture and ignores bundles without one', async () => {
    const { mkdtempSync, mkdirSync, writeFileSync } = await import('fs')
    const { join } = await import('path')
    const { tmpdir } = await import('os')
    const { findBundledDflash2Drafter } = await import('../src/main/dflash2Bundle')
    const root = mkdtempSync(join(tmpdir(), 'vmlx-dflash2-'))
    const withDrafter = join(root, 'Qwen3.8-27B-JANG_4D')
    mkdirSync(join(withDrafter, 'dflash2'), { recursive: true })
    writeFileSync(join(withDrafter, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }))
    writeFileSync(join(withDrafter, 'dflash2', 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }))
    expect(findBundledDflash2Drafter(withDrafter)).toBe(join(withDrafter, 'dflash2'))

    const without = join(root, 'Qwen3.8-27B-JANG_2D')
    mkdirSync(join(without, 'mtp_draft'), { recursive: true })
    writeFileSync(join(without, 'mtp_draft', 'config.json'), JSON.stringify({ architectures: ['Qwen3ForCausalLM'] }))
    expect(findBundledDflash2Drafter(without)).toBeNull()
    expect(findBundledDflash2Drafter(join(root, 'missing'))).toBeNull()
    expect(findBundledDflash2Drafter('')).toBeNull()
  })
})

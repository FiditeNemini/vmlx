import { mkdtempSync, mkdirSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { describe, expect, it } from 'vitest'
import { configDeclaresDspark, findBundledDsparkDrafter } from '../src/main/dsparkBundle'
import { buildCommandPreview } from '../src/renderer/src/components/sessions/SessionSettings'
import { DEFAULT_CONFIG } from '../src/renderer/src/components/sessions/SessionConfigForm'

function bundle(drafter: Record<string, unknown> | null, dir = 'dspark'): string {
  const root = mkdtempSync(join(tmpdir(), 'dspark-bundle-'))
  writeFileSync(join(root, 'config.json'), JSON.stringify({ model_type: 'minicpmv4_7' }))
  if (drafter) {
    mkdirSync(join(root, dir))
    writeFileSync(join(root, dir, 'config.json'), JSON.stringify(drafter))
  }
  return root
}

describe('bundled DSpark drafter detection', () => {
  it('recognizes the openbmb (MiniCPM-V-4.7) and SpecForge (Naive N0.5) drafter configs', () => {
    expect(configDeclaresDspark({ architectures: ['Qwen3DSparkModel'] })).toBe(true)
    expect(configDeclaresDspark({ architectures: ['DSparkDraftModel'] })).toBe(true)
    expect(configDeclaresDspark({ dflash_config: { projector_type: 'dspark' } })).toBe(true)
    expect(configDeclaresDspark({ architectures: ['DFlash2DraftModel'] })).toBe(false)
    expect(configDeclaresDspark(null)).toBe(false)
  })

  it('finds <bundle>/dspark and ignores bundles without one', () => {
    const root = bundle({ architectures: ['Qwen3DSparkModel'] })
    expect(findBundledDsparkDrafter(root)).toBe(join(root, 'dspark'))
    expect(findBundledDsparkDrafter(bundle(null))).toBeNull()
    expect(findBundledDsparkDrafter(bundle({ architectures: ['DFlash2DraftModel'] }, 'dflash2'))).toBeNull()
    expect(findBundledDsparkDrafter('')).toBeNull()
  })
})

const flat = (cmd: string) => cmd.replace(/\s*\\\n\s*/g, ' ')

describe('command preview with a bundled DSpark drafter', () => {
  const base: any = { ...DEFAULT_CONFIG, continuousBatching: true }
  const detected = { family: 'minicpm-v4.7', isMultimodal: true }

  it('defaults on: single-request lane and --dspark on', () => {
    const cmd = flat(buildCommandPreview('/m', { ...base }, detected, '', false, undefined, true))
    expect(cmd).toContain('--dspark on')
    expect(cmd).toContain('--no-continuous-batching')
    expect(cmd).not.toContain('--continuous-batching ')
  })

  it('unchecked: --dspark off and the configured batching', () => {
    const cmd = flat(buildCommandPreview('/m', { ...base, useBundledDspark: false }, detected, '', false, undefined, true))
    expect(cmd).toContain('--dspark off')
    expect(cmd).toContain('--continuous-batching')
  })

  it('bundles without a drafter get no --dspark flag', () => {
    expect(buildCommandPreview('/m', { ...base }, detected, '', false, undefined, false)).not.toContain('--dspark')
  })
})

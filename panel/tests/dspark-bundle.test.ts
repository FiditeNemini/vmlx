import { mkdtempSync, mkdirSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { describe, expect, it } from 'vitest'
import { bundledDsparkDefault, configDeclaresDspark, findBundledDsparkDrafter } from '../src/main/dsparkBundle'
import { buildCommandPreview } from '../src/renderer/src/components/sessions/SessionSettings'
import { DEFAULT_CONFIG } from '../src/renderer/src/components/sessions/SessionConfigForm'

function bundle(drafter: Record<string, unknown> | null, dir = 'dspark', stamp?: Record<string, unknown>): string {
  const root = mkdtempSync(join(tmpdir(), 'dspark-bundle-'))
  writeFileSync(join(root, 'config.json'), JSON.stringify({ model_type: 'minicpmv4_7' }))
  if (stamp) writeFileSync(join(root, 'jang_config.json'), JSON.stringify({ speculative: stamp }))
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

  it("reads the bundle's measured default from jang_config speculative.default", () => {
    const drafter = { architectures: ['Qwen3DSparkModel'] }
    expect(bundledDsparkDefault(bundle(drafter, 'dspark', { method: 'dspark', default: false }))).toBe(false)
    expect(bundledDsparkDefault(bundle(drafter, 'dspark', { method: 'dspark', default: true }))).toBe(true)
    expect(bundledDsparkDefault(bundle(drafter, 'dspark', { method: 'dspark' }))).toBe(true)
    expect(bundledDsparkDefault(bundle(drafter))).toBe(true)
  })
})

const flat = (cmd: string) => cmd.replace(/\s*\\\n\s*/g, ' ')

describe('command preview with a bundled DSpark drafter', () => {
  const base: any = { ...DEFAULT_CONFIG, continuousBatching: true }
  const detected = { family: 'minicpm-v4.7', isMultimodal: true }

  const on = { default: true }
  const off = { default: false }
  const preview = (cfg: any, dspark: any) => flat(buildCommandPreview('/m', { ...base, ...cfg }, detected, '', false, undefined, dspark))

  it('unset follows the stamp: stamped on = single-request lane with --dspark auto', () => {
    const cmd = preview({}, on)
    expect(cmd).toContain('--dspark auto')
    expect(cmd).toContain('--no-continuous-batching')
    expect(cmd).not.toContain('--continuous-batching ')
  })

  it('unset follows the stamp: stamped off = the configured batching with --dspark auto', () => {
    const cmd = preview({}, off)
    expect(cmd).toContain('--dspark auto')
    expect(cmd).toContain('--continuous-batching')
  })

  it('an explicit choice wins over the stamp', () => {
    const forcedOn = preview({ useBundledDspark: true }, off)
    expect(forcedOn).toContain('--dspark on')
    expect(forcedOn).toContain('--no-continuous-batching')
    const forcedOff = preview({ useBundledDspark: false }, on)
    expect(forcedOff).toContain('--dspark off')
    expect(forcedOff).toContain('--continuous-batching')
  })

  it('bundles without a drafter get no --dspark flag', () => {
    expect(preview({}, false)).not.toContain('--dspark')
  })
})

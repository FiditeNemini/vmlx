import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildNativeMtpLaunchArgs, resolveNativeMtpStartupMode } from '../src/shared/nativeMtpLaunchArgs'
import { applyEffectiveSessionGenerationDefaults, applyMtpSamplerOverrides } from '../src/shared/effectiveGenerationDefaults'

describe('Flash Next MTP opt-in startup default', () => {
  it.each(['qwen4-exp', 'qwen4_exp', 'qwen4_exp_text'])('starts %s Off, independent of quant tier or dormant depth', family => {
    for (const depth of [1, 2, 3]) {
      const mode = resolveNativeMtpStartupMode(family)
      expect(mode).toBe('off')
      expect(buildNativeMtpLaunchArgs({
        supported: true, mode, configuredDepth: depth, depthOverride: true,
      })).toEqual(['--disable-native-mtp'])
    }
  })

  it.each(['auto', 'deterministic', 'off'] as const)('preserves an explicitly saved %s mode', mode => {
    expect(resolveNativeMtpStartupMode('qwen4-exp', mode)).toBe(mode)
  })

  it('leaves 27B, GLM and unknown family defaults unchanged', () => {
    for (const family of ['qwen3.5', 'qwen3_5', 'glm5-next', 'hy3', undefined]) {
      expect(resolveNativeMtpStartupMode(family)).toBe('auto')
    }
  })

  it('does not pin bundle sampling or rewrite explicit API/chat overrides while Off', () => {
    const config = { nativeMtpMode: resolveNativeMtpStartupMode('qwen4-exp'), nativeMtpDepth: 3, nativeMtpDepthOverride: true }
    const bundle = { temperature: 1, topP: 0.95, topK: 20, maxTokens: 16384 }
    const explicit = { temperature: 0.7, topP: 0.9, maxTokens: 4096 }
    expect(applyEffectiveSessionGenerationDefaults(bundle, config, { supported: true })).toBe(bundle)
    expect(applyMtpSamplerOverrides(explicit, config, { supported: true })).toBe(explicit)
  })

  it('uses the shared policy for fresh Chat/Server, Reset, and missing persisted fields', () => {
    const read = (path: string) => readFileSync(resolve(__dirname, '../src', path), 'utf8')
    const create = read('renderer/src/components/sessions/CreateSession.tsx')
    expect(create).toContain('nativeMtpMode: resolveNativeMtpStartupMode(')
    expect(create).toContain('!defaultsOnly && nativeMtpModeEditedRef.current ? prev.nativeMtpMode : undefined')
    expect(create).toContain('resolveNativeMtpStartupMode(det?.family, stored.nativeMtpMode)')
    for (const component of ['CreateSession', 'SessionSettings', 'ServerSettingsDrawer']) {
      expect(read(`renderer/src/components/sessions/${component}.tsx`))
        .toContain('base.nativeMtpMode = resolveNativeMtpStartupMode(detected.family)')
    }
    const main = read('main/sessions.ts')
    expect(main).toContain("effectiveFamily === 'qwen4-exp' && (config as any).nativeMtpMode === undefined")
    expect(main).toContain('(config as any).nativeMtpMode = resolveNativeMtpStartupMode(effectiveFamily)')
    expect(main).toContain('requestedNativeMtpMode === undefined && existingConfig.nativeMtpMode !== undefined')
    expect(main).toContain('(config as any).nativeMtpMode = existingConfig.nativeMtpMode')
  })
})

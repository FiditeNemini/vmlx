import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildNativeMtpLaunchArgs, resolveNativeMtpStartupMode } from '../src/shared/nativeMtpLaunchArgs'

describe('Native MTP startup default (two product modes)', () => {
  it.each(['qwen4-exp', 'qwen4_exp', 'qwen4_exp_text', 'qwen3.5', 'glm5-next', undefined])(
    'starts %s Adaptive when the bundle passed the gate and declares no measured-off default', family => {
      const mode = resolveNativeMtpStartupMode(family)
      expect(mode).toBe('adaptive')
      expect(buildNativeMtpLaunchArgs({ supported: true, mode })).toEqual([
        '--native-mtp-depth-policy', 'adaptive', '--native-mtp-sampling-policy', 'compatible-only',
      ])
    },
  )

  it('starts AR for a bundle whose measured verifier declares defaultMode off', () => {
    expect(resolveNativeMtpStartupMode('glm5-next', undefined, 'off')).toBe('off')
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'off' })).toEqual(['--disable-native-mtp'])
  })

  it.each(['adaptive', 'off'] as const)('preserves an explicitly saved %s mode', mode => {
    expect(resolveNativeMtpStartupMode('qwen4-exp', mode)).toBe(mode)
    expect(resolveNativeMtpStartupMode('qwen4-exp', mode, 'off')).toBe(mode)
  })

  it('uses the shared policy for fresh Chat/Server, Reset, and missing persisted fields', () => {
    const read = (path: string) => readFileSync(resolve(__dirname, '../src', path), 'utf8')
    const create = read('renderer/src/components/sessions/CreateSession.tsx')
    expect(create).toContain('nativeMtpMode: resolveNativeMtpStartupMode(')
    expect(create).toContain('!defaultsOnly && nativeMtpModeEditedRef.current ? prev.nativeMtpMode : undefined')
    expect(create).toContain('resolveNativeMtpStartupMode(det?.family, stored.nativeMtpMode, det?.nativeMtp?.defaultMode)')
    for (const component of ['CreateSession', 'SessionSettings', 'ServerSettingsDrawer']) {
      expect(read(`renderer/src/components/sessions/${component}.tsx`))
        .toContain('base.nativeMtpMode = resolveNativeMtpStartupMode(detected.family, undefined, detected.nativeMtp?.defaultMode)')
    }
    const main = read('main/sessions.ts')
    expect(main).toContain("detectedNativeMtp?.supported && (config as any).nativeMtpMode === undefined")
    expect(main).toContain('resolveNativeMtpStartupMode(')
    expect(main).toContain('requestedNativeMtpMode === undefined && existingConfig.nativeMtpMode !== undefined')
    expect(main).toContain('(config as any).nativeMtpMode = existingConfig.nativeMtpMode')
    // The retired fixed-D3 family fill must not come back.
    expect(main).not.toContain('nativeMtpDepthOverride = true')
    expect(main).not.toContain('nativeMtpDepth = 3')
  })

  it('no product surface emits a fixed depth, greedy-only or deterministic-defaults', () => {
    const read = (path: string) => readFileSync(resolve(__dirname, '../src', path), 'utf8')
    for (const file of ['main/sessions.ts', 'renderer/src/components/sessions/SessionSettings.tsx', 'shared/nativeMtpLaunchArgs.ts']) {
      const source = read(file)
      // The additional-args blocklists legitimately still name the retired
      // flags so a user cannot smuggle them back in; the launch producer must
      // not emit them.
      const launchStart = source.indexOf('buildNativeMtpLaunchArgs({')
      const launch = launchStart === -1 ? source : source.slice(launchStart, launchStart + 1200)
      expect(launch).not.toContain("'--native-mtp-depth',")
      expect(launch).not.toContain("configuredDepth")
      expect(launch).not.toContain("depthOverride")
      // Legacy process adoption still parses the retired policies from argv/
      // health; the launch producer must not emit them.
      expect(launch).not.toContain('greedy-only')
      expect(launch).not.toContain('deterministic-defaults')
    }
    const form = read('renderer/src/components/sessions/SessionConfigForm.tsx')
    expect(form).not.toContain('settingKey="nativeMtpDepth"')
    expect(form).not.toContain('nativeMtpDepthPolicy')
    expect(form).toContain("{ value: 'adaptive', label: t('sessions.config.nativeMtpModeAdaptive') }")
    expect(form).toContain("{ value: 'off', label: t('sessions.config.nativeMtpModeAr') }")
    expect(form).not.toContain("value: 'deterministic'")
    expect(form).not.toContain("{ value: 'auto', label: t('sessions.config.mtpAutoBundleDefaults') }")
  })
})

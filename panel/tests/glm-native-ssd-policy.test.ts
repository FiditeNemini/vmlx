import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { detectModelConfigFromDir } from '../src/main/model-config-registry'
import { usesExactTypedPromptDiskCache } from '../src/shared/detectedFamilyNames'
import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'

const dirs: string[] = []
afterEach(() => {
  vi.unstubAllEnvs()
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

describe('GLM native SSD opt-in, not a saved default migration', () => {
  it.each(['1', '0', 'true'])('propagates only the explicit runtime opt-in %s', (flag) => {
    vi.stubEnv('VMLX_GLM5_NATIVE_SSD', flag)
    const dir = mkdtempSync(join(tmpdir(), 'vmlx-glm-native-policy-'))
    dirs.push(dir)
    writeFileSync(join(dir, 'config.json'), JSON.stringify({
      model_type: 'glm5_next', text_config: { model_type: 'glm5_next_text' },
    }))
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('glm5-next')
    expect(detected.nativeGlmSsd).toBe(flag === '1' ? true : undefined)
    expect(usesExactTypedPromptDiskCache(detected.family, detected.nativeGlmSsd)).toBe(flag !== '1')
    // The setting is main-process runtime metadata, never stamped into a bundle.
    expect(detected.cacheSubtype).toBe('glm5_next_native_v2')
  })

  it.each(['glm5_next', 'glm5_next_text', 'glm5-next'])('preserves the explicit SSD switch and cap for %s', (family) => {
    for (const enabled of [true, false]) {
      const legacy = usesExactTypedPromptDiskCache(family, true)
      const result = buildCacheLaunchArgs({
        continuousBatching: true, enablePrefixCache: true, usePagedCache: false,
        enableDiskCache: false, enableBlockDiskCache: legacy ? false : enabled,
        forceMemoryAwareCache: legacy, blockDiskCacheDir: '/test/owned-native-pool',
        blockDiskCacheMaxGb: 3, blockDiskCacheMaxPercent: 15,
      })
      expect(result.args).toContain('--no-paged-cache')
      expect(result.args).not.toContain('--use-paged-cache')
      expect(result.policy.enableBlockDiskCache).toBe(enabled)
      expect(result.args.includes('--disable-block-disk-cache')).toBe(!enabled)
      if (enabled) {
        expect(result.args).toContain('/test/owned-native-pool')
        const cap = result.args.indexOf('--block-disk-cache-max-gb')
        expect(cap).toBeGreaterThanOrEqual(0)
        expect(result.args[cap + 1]).toBe('3')
        // The engine resolves explicit GB ahead of percent. Preserve both
        // user inputs; the native scheduler receives that resolved GB cap.
        const percent = result.args.indexOf('--block-disk-cache-max-percent')
        expect(percent).toBeGreaterThanOrEqual(0)
        expect(result.args[percent + 1]).toBe('15')
        expect(result.args).not.toContain('--enable-disk-cache')
      }
    }
  })

  it('does not change normal GLM or other family policy', () => {
    expect(usesExactTypedPromptDiskCache('glm5-next')).toBe(true)
    expect(usesExactTypedPromptDiskCache('openpangu_v2', true)).toBe(true)
    expect(usesExactTypedPromptDiskCache('qwen4_exp', true)).toBe(false)
  })

  it('hides non-applicable RAM and generic block controls only for native GLM', () => {
    const form = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
    expect(form).toContain("normalizedDetectedFamily === 'glm5-next' && detectedNativeGlmSsd === true")
    expect(form).toContain('!nativeGlmSsdActive && !dsv4Active && !blockDiskOnly')
    expect(form).toContain('!nativeGlmSsdActive && (effectiveUsePagedCache || cachePolicy.blockDiskCacheChecked)')
    expect(form).toContain('!nativeGlmSsdActive && (exactTypedPromptDiskCache || cachePolicy.legacyDiskCacheChecked)')
    expect(form).toContain('data-vmlx-section="nativeGlmSsd"')
    for (const language of ['en', 'zh', 'ko', 'ja', 'es']) {
      const locale = JSON.parse(readFileSync(`src/renderer/src/i18n/locales/${language}.json`, 'utf8'))
      expect(locale.sessions.config.glmNativeSsdNote.length).toBeGreaterThan(40)
    }
  })
})

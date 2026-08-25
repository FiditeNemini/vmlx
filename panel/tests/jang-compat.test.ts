import { describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { getJangCompatWarning, KNOWN_INCOMPATIBLE_FAMILIES } from '../src/renderer/src/lib/jangCompat'
import { validateJangBundleMetadataForLaunch } from '../src/main/model-bundle-validation'

describe('getJangCompatWarning', () => {
  it('returns null when no model_type provided', () => {
    expect(getJangCompatWarning(undefined, 'JANG_2L')).toBeNull()
    expect(getJangCompatWarning(null, 'JANG_2L')).toBeNull()
    expect(getJangCompatWarning('', 'JANG_2L')).toBeNull()
  })

  it('returns null for non-hybrid families regardless of profile', () => {
    expect(getJangCompatWarning('llama', 'JANG_2L')).toBeNull()
    expect(getJangCompatWarning('mistral', 'JANG_2S')).toBeNull()
    expect(getJangCompatWarning('qwen3', 'JANG_1L')).toBeNull()
    expect(getJangCompatWarning('gemma', 'JANG_2M')).toBeNull()
  })

  it('warns for hybrid families across JANG profiles because layout compatibility is profile-independent', () => {
    expect(getJangCompatWarning('qwen3_next', 'JANG_4M')).not.toBeNull()
    expect(getJangCompatWarning('qwen3_next', 'JANG_6M')).not.toBeNull()
    expect(getJangCompatWarning('nemotron_h', 'JANG_4S')).not.toBeNull()
    expect(getJangCompatWarning('mamba2', 'JANG_3M')).not.toBeNull()
  })

  it('warns on qwen3_next + low JANG profile (2-bit/1-bit tier)', () => {
    const warn = getJangCompatWarning('qwen3_next', 'JANG_2L')
    expect(warn).not.toBeNull()
    // Message is family-agnostic (DRY across hybrid families) but must mention
    // the actionable signals so users know it's about external mlx_lm load.
    expect(warn).toMatch(/conv1d/i)
    expect(warn).toMatch(/jang.*2\.5\.27/i)
    expect(warn).toMatch(/mlx_lm/i)
    expect(warn).toMatch(/not a coherence guarantee/i)
  })

  it('warns on all hybrid families at any JANG profile', () => {
    const jangProfiles = ['JANG_1L', 'JANG_2S', 'JANG_2M', 'JANG_2L', 'JANG_3M', 'JANG_4M', 'JANG_6M']
    for (const family of Object.keys(KNOWN_INCOMPATIBLE_FAMILIES)) {
      for (const profile of jangProfiles) {
        const warn = getJangCompatWarning(family, profile)
        expect(warn, `${family} + ${profile} should warn`).not.toBeNull()
      }
    }
  })

  it('is case-insensitive on model_type', () => {
    expect(getJangCompatWarning('QWEN3_NEXT', 'JANG_2L')).not.toBeNull()
    expect(getJangCompatWarning('Qwen3_Next', 'JANG_2L')).not.toBeNull()
  })

  it('returns null when profile is unknown or undefined', () => {
    expect(getJangCompatWarning('qwen3_next', undefined)).toBeNull()
    expect(getJangCompatWarning('qwen3_next', '')).toBeNull()
    expect(getJangCompatWarning('qwen3_next', 'NOT_A_PROFILE')).toBeNull()
  })

  it('does NOT warn for MLX presets (only JANG_*)', () => {
    expect(getJangCompatWarning('qwen3_next', 'mlx_4bit')).toBeNull()
    expect(getJangCompatWarning('qwen3_next', 'balanced')).toBeNull()
  })
})

describe('KNOWN_INCOMPATIBLE_FAMILIES schema', () => {
  it('covers all hybrid families per vMLX engine mamba_cache.py:244', () => {
    // Families per the deep audit doc section 2 (mamba_cache.py):
    // qwen3_5, qwen3_5_moe, qwen3_next, nemotron_h, mamba2, lfm2_moe, granitemoehybrid
    // Plus aliases per capabilities.py
    const required = ['qwen3_next', 'nemotron_h', 'mamba2', 'lfm2_moe', 'granitemoehybrid']
    for (const family of required) {
      expect(KNOWN_INCOMPATIBLE_FAMILIES[family], `${family} missing from map`).toBeDefined()
    }
  })

  it('each entry has profiles_warn array', () => {
    for (const [family, entry] of Object.entries(KNOWN_INCOMPATIBLE_FAMILIES)) {
      expect(Array.isArray(entry.profiles_warn), `${family} missing profiles_warn array`).toBe(true)
      expect(entry.profiles_warn.length, `${family} profiles_warn empty`).toBeGreaterThan(0)
      expect(typeof entry.message, `${family} missing message`).toBe('string')
    }
  })
})

describe('authoritative JANG bundle launch metadata', () => {
  function withBundle(
    files: Record<string, unknown | string>,
    check: (path: string) => void,
  ): void {
    const path = mkdtempSync(join(tmpdir(), 'vmlx-jang-contract-'))
    try {
      for (const [name, value] of Object.entries(files)) {
        writeFileSync(
          join(path, name),
          typeof value === 'string' ? value : JSON.stringify(value),
        )
      }
      check(path)
    } finally {
      rmSync(path, { recursive: true, force: true })
    }
  }

  it('preserves config-only and pre-capabilities legacy bundles', () => {
    withBundle(
      { 'config.json': { model_type: 'qwen3' } },
      (path) => expect(validateJangBundleMetadataForLaunch(path)).toMatchObject({
        ok: true,
        schema: 'legacy',
      }),
    )
    withBundle(
      {
        'config.json': { model_type: 'qwen3' },
        'jang_config.json': { format: 'JANG', quantization: { bits: 4 } },
      },
      (path) => expect(validateJangBundleMetadataForLaunch(path)).toMatchObject({
        ok: true,
        schema: 'legacy',
      }),
    )
  })

  it('accepts current capabilities and legacy chat-authoritative schemas', () => {
    withBundle(
      {
        'config.json': { model_type: 'qwen3' },
        'jang_config.json': { capabilities: { family: 'qwen3' } },
      },
      (path) => expect(validateJangBundleMetadataForLaunch(path)).toMatchObject({
        ok: true,
        schema: 'capabilities',
        family: 'qwen3',
      }),
    )
    withBundle(
      {
        'config.json': { model_type: 'deepseek_v4' },
        'jang_config.json': { model_family: 'deepseek_v4', chat: {} },
      },
      (path) => expect(validateJangBundleMetadataForLaunch(path)).toMatchObject({
        ok: true,
        schema: 'chat',
        family: 'deepseek_v4',
      }),
    )
  })

  it('rejects malformed or family-less authoritative stamps before spawn', () => {
    withBundle(
      {
        'config.json': { model_type: 'lfm2_moe' },
        'jang_config.json': { capabilities: { tools: true, reasoning: true } },
      },
      (path) => {
        const result = validateJangBundleMetadataForLaunch(path)
        expect(result.ok).toBe(false)
        expect(result.error).toMatch(/capabilities\.family/)
      },
    )
    withBundle(
      {
        'config.json': { model_type: 'deepseek_v4' },
        'jang_config.json': { chat: {} },
      },
      (path) => {
        const result = validateJangBundleMetadataForLaunch(path)
        expect(result.ok).toBe(false)
        expect(result.error).toMatch(/model_family/)
      },
    )
    withBundle(
      {
        'config.json': { model_type: 'qwen3' },
        'jang_config.json': '{not-json',
      },
      (path) => expect(validateJangBundleMetadataForLaunch(path).ok).toBe(false),
    )
    withBundle(
      {
        'config.json': '{not-json',
        'jang_config.json': { capabilities: { family: 'qwen3' } },
      },
      (path) => expect(validateJangBundleMetadataForLaunch(path).ok).toBe(false),
    )
  })
})

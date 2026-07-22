import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { applyBundleGenerationDefaultsToSessionConfig } from '../src/shared/sessionGenerationDefaults'

describe('session generation-default hydration', () => {
  it('preserves an explicit do_sample=false declaration', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { doSample: false })).toMatchObject({
      defaultDoSample: false,
      defaultSamplingDefaultsDeclared: true,
    })
  })

  it('preserves an unusually large model-derived top-k', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { topK: 151552 }).defaultTopK)
      .toBe(151552)
  })

  it('hydrates the model-derived maximum output-token default', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { maxNewTokens: 32768 }).defaultMaxNewTokens)
      .toBe(32768)
  })

  it('resets absent bundle defaults to neutral inheritance sentinels', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({ unrelated: true }, null)).toEqual({
      unrelated: true,
      defaultTemperature: 0,
      defaultTopP: 0,
      defaultTopK: 0,
      defaultMinP: 0,
      defaultRepetitionPenalty: 0,
      defaultMaxNewTokens: 0,
      defaultDoSample: undefined,
      defaultSamplingDefaultsDeclared: false,
    })
  })

  it('wires both settings surfaces on initial load and Reset', () => {
    for (const sourcePath of [
      'src/renderer/src/components/sessions/SessionSettings.tsx',
      'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
    ]) {
      const source = readFileSync(sourcePath, 'utf8')
      expect(source.match(/getGenerationDefaults\(/g)).toHaveLength(2)
      expect(source.match(/applyBundleGenerationDefaultsToSessionConfig\(/g)).toHaveLength(2)
      expect(source).toContain('return () => { active = false }')
      expect(source).toContain('resetStillCurrent')
      expect(source).toContain('setConfig(current => applyBundleGenerationDefaultsToSessionConfig(current, generationDefaults))')
    }
  })

  it('uses one shared mapper for fresh and previously launched sessions', () => {
    const source = readFileSync(
      'src/renderer/src/components/sessions/CreateSession.tsx',
      'utf8',
    )
    expect(source.match(/applyBundleGenerationDefaultsToSessionConfig\(/g)).toHaveLength(3)
    expect(source).not.toContain('function applyGenerationDefaultsToConfig')
    expect(source).not.toContain('function applyGenerationDefaultsToStoredConfig')
    expect(source).toContain('Promise.all([')
  })
})

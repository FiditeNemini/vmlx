import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { describeDsv4ActivationQat } from '../src/renderer/src/components/sessions/dsv4QatStatus'

describe('DSV4 activation-QAT status presentation', () => {
  it('keeps pre-forward observation pending instead of reporting a mismatch', () => {
    expect(describeDsv4ActivationQat({
      requested: false,
      runtime_requested: false,
      effective: false,
      observed: null,
      attested: false,
      matches_request: false,
      paths: {
        attention_kv_and_pool_e4m3: null,
        indexer_hadamard128_fp4_e2m1: null,
      },
      fused_kernels: {
        e4m3_available: true,
        indexer_available: true,
      },
      supported: true,
    })).toEqual({
      requestedEffective: 'saved off · runtime off · effective off · supported',
      observedAttestation: 'observed pending · not attested · match pending',
      paths: 'KV/pool pending · indexer pending',
      fusedKernels: 'E4M3 ready · indexer ready',
    })
  })

  it('reports an attested enabled runtime and both observed transform paths', () => {
    expect(describeDsv4ActivationQat({
      requested: true,
      runtime_requested: true,
      effective: true,
      observed: true,
      attested: true,
      matches_request: true,
      paths: {
        attention_kv_and_pool_e4m3: true,
        indexer_hadamard128_fp4_e2m1: true,
      },
      fused_kernels: {
        e4m3_available: true,
        indexer_available: false,
      },
      supported: true,
    })).toEqual({
      requestedEffective: 'saved on · runtime on · effective on · supported',
      observedAttestation: 'observed on · attested · match',
      paths: 'KV/pool on · indexer on',
      fusedKernels: 'E4M3 ready · indexer missing',
    })
  })

  it('surfaces an attested requested-versus-observed mismatch', () => {
    expect(describeDsv4ActivationQat({
      requested: true,
      effective: true,
      observed: false,
      attested: true,
      matches_request: false,
    }).observedAttestation).toBe('observed off · attested · mismatch')
  })

  it('renders the same four live truth cards in Cache and Performance', () => {
    for (const file of [
      'src/renderer/src/components/sessions/CachePanel.tsx',
      'src/renderer/src/components/sessions/PerformancePanel.tsx',
    ]) {
      const source = readFileSync(file, 'utf8')
      expect(source).toContain('describeDsv4ActivationQat')
      // Both panels render the four cards through the shared i18n keys; the
      // English copy lives in the locale catalog now.
      expect(source).toContain("label={t('sessions.cache.dsv4QatRequested')}")
      expect(source).toContain("label={t('sessions.cache.dsv4QatObserved')}")
      expect(source).toContain("label={t('sessions.cache.dsv4QatPaths')}")
      expect(source).toContain("label={t('sessions.cache.dsv4QatKernels')}")
    }
    const catalog = readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8')
    expect(catalog).toContain('DSV4 QAT Requested')
    expect(catalog).toContain('DSV4 QAT Observed')
    expect(catalog).toContain('DSV4 QAT Paths')
    expect(catalog).toContain('DSV4 QAT Kernels')
  })
})

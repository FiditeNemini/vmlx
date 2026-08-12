import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { formatCacheStorageBytes } from '../src/renderer/src/components/sessions/CachePanel'

const source = readFileSync(
  'src/renderer/src/components/sessions/CachePanel.tsx',
  'utf8',
)

describe('CachePanel last-request truthfulness', () => {
  it('invalidates stale refreshes and clears state when session identity changes', () => {
    expect(source).toContain('requestGuard.beginLatest(expectedIdentity)')
    expect(source).toContain('requestGuard.isCurrent(requestToken)')
    expect(source).toContain('requestGuard.invalidateRequests()')
    expect(source).toContain('requestGuard.resetIdentity()')
    expect(source).toContain('requestGuard.beginAction(identity)')
    expect(source).toContain('requestGuard.finishAction(actionToken)')
    expect(source).toContain('identityKeyRef.current !== identityKey')
    expect(source).toContain('warmInputGenerationRef.current === submittedInputGeneration')
    expect(source).toContain('disabled={actionBusy}')
    expect(source).toContain('setStats(null)')
    expect(source).toContain('setEntries(null)')
    expect(source).toContain('setError(null)')
  })

  it('reads cache execution telemetry from scheduler and batch-generator shapes', () => {
    expect(source).toContain('schedulerStats?.last_cache_execution')
    expect(source).toContain('schedulerStats?.batch_generator?.last_cache_execution')
    const cachePanelCopy = readFileSync(
        'src/renderer/src/i18n/locales/en.json',
        'utf-8',
    )
    // copy lives in the locale catalog now that the panel is translated
    expect(cachePanelCopy).toContain('Last Cache Execution')
  })

  it('renders prompt, cached, uncached, prefill, block, timing, and fallback fields', () => {
    for (const field of [
      'prompt_tokens',
      'cached_tokens',
      'uncached_prompt_tokens',
      'prefill_tokens',
      'generation_prompt_suffix_tokens',
      'blocks',
      'disk_blocks',
      'reconstruction_seconds',
      'dequantization_seconds',
      'total_worker_cache_seconds',
      'cache_reuse_applied',
      'fallback_reason',
    ]) {
      expect(source).toContain(`lastCacheExecution.${field}`)
    }
  })

  it('describes only longest causal-prefix reuse and rejects arbitrary suffix claims', () => {
    // The explainer moved into i18n, so the claim now lives in the catalogs —
    // checking those is strictly stronger than checking the component, because
    // it verifies what users in every language actually read.
    expect(source).toContain("t('sessions.cachePanel.reuseExplainer')")
    const en = JSON.parse(
      readFileSync(
        join(__dirname, '..', 'src/renderer/src/i18n/locales/en.json'),
        'utf8',
      ),
    )
    const copy: string = en.sessions.cachePanel.reuseExplainer
    expect(copy).toContain('longest continuous causal token prefix from token 0')
    expect(copy).toContain('Only the unmatched tail is prefilled')
    expect(copy).toMatch(/arbitrary suffix or interior sequences are never reused/)
    // ...and it must NOT stop there: unqualified, that claim is false for
    // path-dependent families, where a SHORTER prompt can reuse nothing.
    expect(copy.toLowerCase()).toContain('shorter')
  })

  it('separates persistent namespace occupancy from current-engine activity', () => {
    // The labels render through i18n keys; assert the wiring in the source and
    // the English copy in the locale catalog.
    const catalog = readFileSync(
      join(__dirname, '..', 'src/renderer/src/i18n/locales/en.json'),
      'utf8',
    )
    for (const [key, marker] of [
      ["t('sessions.cache.persistedBlockReads')", 'Persisted Block Reads'],
      ["t('sessions.cache.thisEngineReads')", 'This Engine Reads H / M'],
      ["t('sessions.cache.thisEngineWrites')", 'This Engine Writes'],
      ["t('sessions.cache.thisEngineEvictions')", 'This Engine Evictions'],
      ["t('sessions.cache.writerPendingInFlight')", 'Writer Pending / In Flight'],
      ["t('sessions.cache.offThreadWrites')", 'Off-thread Writes Q / C / F'],
      ["t('sessions.cache.lastReconciliationTrim')", 'Last Local Reconciliation Trim'],
    ]) {
      expect(source).toContain(key)
      expect(catalog).toContain(marker)
    }
    expect(source).toContain('!blockDiskCache && schedulerCache.disk_hits')
    expect(source).toContain('blockDiskCache.disk_size_bytes')
  })

  it('does not round small nonzero namespaces down to 0.00 GB', () => {
    expect(formatCacheStorageBytes(512)).toBe('512 B')
    expect(formatCacheStorageBytes(512 * 1024)).toBe('512.0 KB')
    expect(formatCacheStorageBytes(32 * 1024 ** 2)).toBe('32.0 MB')
    expect(formatCacheStorageBytes(1.5 * 1024 ** 3)).toBe('1.50 GB')
  })
})

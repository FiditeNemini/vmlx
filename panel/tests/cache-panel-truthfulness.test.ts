import { readFileSync } from 'node:fs'
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
    expect(source).toContain('Last Cache Execution')
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
    expect(source).toContain(
      'longest continuous causal token prefix from token 0',
    )
    expect(source).toContain('Only the unmatched tail is sent through prefill')
    expect(source).toMatch(
      /arbitrary suffix or\s+interior token sequences are never reused/,
    )
  })

  it('separates persistent namespace occupancy from current-engine activity', () => {
    for (const marker of [
      'Persisted Block Reads',
      'This Engine Reads H / M',
      'This Engine Writes',
      'This Engine Evictions',
      'Writer Pending / In Flight',
      'Off-thread Writes Q / C / F',
      'Last Local Reconciliation Trim',
    ]) {
      expect(source).toContain(marker)
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

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = readFileSync(
  'src/renderer/src/components/sessions/CachePanel.tsx',
  'utf8',
)

describe('CachePanel last-request truthfulness', () => {
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
})

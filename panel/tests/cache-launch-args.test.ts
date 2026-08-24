import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'

const SSD_ONLY = {
  continuousBatching: true,
  enablePrefixCache: true,
  usePagedCache: true, // stale persisted value must be harmless
  enableDiskCache: false,
  enableBlockDiskCache: true,
  cacheMemoryMb: 8192,
  cacheMemoryPercent: 35,
  cacheTtlMinutes: 60,
  effectivePagedCacheBlockSize: 64,
  maxCacheBlocks: 4097,
  blockDiskCacheMaxGb: 0,
  blockDiskCacheMaxPercent: 10,
}

describe('authoritative cache launch arguments', () => {
  it('builds SSD-only argv with no RAM tier or RAM budget even from stale paged state', () => {
    const result = buildCacheLaunchArgs(SSD_ONLY)

    expect(result.policy.effectiveUsePagedCache).toBe(false)
    expect(result.args).toEqual([
      '--no-paged-cache',
      '--no-vision-memory-cache',
      '--ssm-state-cache-size', '0',
      '--ssm-state-cache-mb', '0',
      '--paged-cache-block-size', '64',
      '--max-cache-blocks', '4097',
      '--enable-block-disk-cache',
      '--block-disk-cache-max-percent', '10',
    ])
    expect(result.args).not.toContain('--use-paged-cache')
    expect(result.args).not.toContain('--enable-vision-memory-cache')
    expect(result.args).not.toContain('--cache-memory-mb')
    expect(result.args).not.toContain('--cache-memory-percent')
    expect(result.args).not.toContain('--block-disk-cache-max-gb')
  })

  it('emits the engine-default override when the visible SSD toggle is off', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      usePagedCache: false,
      enableBlockDiskCache: false,
    })

    expect(result.args).toContain('--no-paged-cache')
    expect(result.args).toContain('--disable-block-disk-cache')
    expect(result.args).not.toContain('--enable-block-disk-cache')
  })

  it('keeps the prefix master switch authoritative', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      enablePrefixCache: false,
    })

    expect(result.args).toEqual([
      '--no-paged-cache',
      '--no-vision-memory-cache',
      '--ssm-state-cache-size', '0',
      '--ssm-state-cache-mb', '0',
      '--disable-prefix-cache',
    ])
  })

  it('keeps openPangu-style exact prompt L2 memory-aware and non-paged', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      enableDiskCache: true,
      enableBlockDiskCache: false,
      noMemoryAwareCache: true,
      forceMemoryAwareCache: true,
      diskCacheDir: '/tmp/prompt-l2',
      diskCacheMaxGb: 10,
    })

    expect(result.args).toContain('--no-paged-cache')
    expect(result.args).not.toContain('--no-memory-aware-cache')
    expect(result.args).toContain('--enable-disk-cache')
    expect(result.args).toContain('--disable-block-disk-cache')
  })

  it('is the cache argv source for both the visual preview and real spawn path', () => {
    const renderer = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
    const launcher = readFileSync('src/main/sessions.ts', 'utf8')

    expect(renderer).toContain("import { buildCacheLaunchArgs }")
    expect(renderer).toContain('const cacheLaunch = buildCacheLaunchArgs({')
    expect(renderer).toContain('parts.push(...cacheLaunch.args)')
    expect(launcher).toContain("import { buildCacheLaunchArgs }")
    expect(launcher).toContain('const cacheLaunch = buildCacheLaunchArgs({')
    expect(launcher).toContain('args.push(...cacheLaunch.args)')
    expect(renderer).not.toContain("parts.push('--use-paged-cache')")
    expect(launcher).not.toContain("args.push('--use-paged-cache')")
  })
})

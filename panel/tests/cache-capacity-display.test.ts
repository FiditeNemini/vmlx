import { describe, expect, it } from 'vitest'

import {
  pagedCacheCapacityText,
  pagedCacheControlsState,
  pagedCacheMemoryIgnoredText,
} from '../src/shared/cacheCapacityDisplay'

describe('cache capacity display helpers', () => {
  it('shows effective paged-cache capacity as block size times max blocks', () => {
    expect(pagedCacheCapacityText({ blockSize: 64, maxBlocks: 1000 })).toBe(
      'Effective paged capacity: 64 tokens/block x 1000 blocks = 64,000 tokens',
    )
    expect(pagedCacheCapacityText({ blockSize: 256, maxBlocks: 64 })).toBe(
      'Effective paged capacity: 256 tokens/block x 64 blocks = 16,384 tokens',
    )
  })

  it('explains that MB/percent set the paged L1 RAM ceiling and only TTL is ignored', () => {
    // #98/H1: under paged cache the memory-budget controls are LIVE — they set the
    // L1 RAM byte ceiling for the block pool. Only Cache TTL is inapplicable.
    expect(pagedCacheMemoryIgnoredText).toContain('Cache TTL is ignored while paged cache is active')
    expect(pagedCacheMemoryIgnoredText).toContain('Cache Memory Limit / Cache Memory %')
    expect(pagedCacheMemoryIgnoredText).toContain('L1 RAM byte ceiling for the paged block pool')
    expect(pagedCacheMemoryIgnoredText).toContain('Max Cache Blocks')
  })

  it('derives disabled/ignored control state from effective paged cache state', () => {
    // #98/H1: memory-budget controls stay enabled+live under paged; only TTL disabled.
    expect(pagedCacheControlsState(true)).toEqual({
      memoryBudgetControlsDisabled: false,
      cacheTtlDisabled: true,
      memoryBudgetIgnored: false,
    })
    expect(pagedCacheControlsState(false)).toEqual({
      memoryBudgetControlsDisabled: false,
      cacheTtlDisabled: false,
      memoryBudgetIgnored: false,
    })
  })
})

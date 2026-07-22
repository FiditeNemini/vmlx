import { describe, expect, it } from 'vitest'

import {
  pagedCacheCapacityText,
  pagedCacheControlsState,
  pagedCacheMemoryIgnoredText,
} from '../src/shared/cacheCapacityDisplay'

describe('cache capacity display helpers', () => {
  it('subtracts the permanently reserved null block from effective capacity', () => {
    expect(pagedCacheCapacityText({ blockSize: 64, maxBlocks: 1000 })).toBe(
      'Effective in-memory cache capacity: 64 tokens/block x 999 usable blocks (1000 configured; 1 reserved) = 63,936 tokens',
    )
    expect(pagedCacheCapacityText({ blockSize: 256, maxBlocks: 64 })).toBe(
      'Effective in-memory cache capacity: 256 tokens/block x 63 usable blocks (64 configured; 1 reserved) = 16,128 tokens',
    )
    expect(pagedCacheCapacityText({ blockSize: 64, maxBlocks: 4 })).toBe(
      'Effective in-memory cache capacity: 64 tokens/block x 3 usable blocks (4 configured; 1 reserved) = 192 tokens',
    )
  })

  it('explains that MB/percent set the paged L1 RAM ceiling and only TTL is ignored', () => {
    // #98/H1: under paged cache the memory-budget controls are LIVE — they set the
    // L1 RAM byte ceiling for the block pool. Only Cache TTL is inapplicable.
    expect(pagedCacheMemoryIgnoredText).toContain('Cache TTL does not apply while In-Memory Paged Cache is on')
    expect(pagedCacheMemoryIgnoredText).toContain('Cache Memory Limit / Cache Memory %')
    expect(pagedCacheMemoryIgnoredText).toContain('L1 use of Apple unified memory')
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
    expect(pagedCacheControlsState(false, true)).toEqual({
      memoryBudgetControlsDisabled: true,
      cacheTtlDisabled: true,
      memoryBudgetIgnored: true,
    })
  })
})

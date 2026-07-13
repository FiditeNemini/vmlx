export interface PagedCacheCapacityInput {
  blockSize: number
  maxBlocks: number
  defaultBlockSize?: number
  defaultMaxBlocks?: number
}

export interface PagedCacheControlsState {
  memoryBudgetControlsDisabled: boolean
  cacheTtlDisabled: boolean
  memoryBudgetIgnored: boolean
}

const DEFAULT_PAGED_BLOCK_SIZE = 64
const DEFAULT_MAX_CACHE_BLOCKS = 1000

function finitePositiveInteger(value: number | undefined, fallback: number): number {
  if (Number.isFinite(value) && Math.floor(value as number) > 0) {
    return Math.floor(value as number)
  }
  return fallback
}

function formatInteger(value: number): string {
  return Math.floor(value).toLocaleString('en-US')
}

export const pagedCacheMemoryIgnoredText =
  'Cache TTL is ignored while paged cache is active. Cache Memory Limit / Cache Memory % set the L1 RAM byte ceiling for the paged block pool (they bound RAM and evict free blocks); Max Cache Blocks and Block Size set token capacity.'

export function resolvePagedCacheCapacity(input: PagedCacheCapacityInput): {
  blockSize: number
  maxBlocks: number
  capacityTokens: number
} {
  const blockSize = finitePositiveInteger(
    input.blockSize,
    finitePositiveInteger(input.defaultBlockSize, DEFAULT_PAGED_BLOCK_SIZE),
  )
  const maxBlocks = finitePositiveInteger(
    input.maxBlocks,
    finitePositiveInteger(input.defaultMaxBlocks, DEFAULT_MAX_CACHE_BLOCKS),
  )
  return {
    blockSize,
    maxBlocks,
    capacityTokens: blockSize * maxBlocks,
  }
}

export function pagedCacheCapacityText(input: PagedCacheCapacityInput): string {
  const resolved = resolvePagedCacheCapacity(input)
  return `Effective paged capacity: ${resolved.blockSize} tokens/block x ${resolved.maxBlocks} blocks = ${formatInteger(resolved.capacityTokens)} tokens`
}

export function pagedCacheControlsState(effectiveUsePagedCache: boolean): PagedCacheControlsState {
  // Under paged cache the memory budget controls (Cache Memory Limit / %) stay
  // LIVE: they set the L1 RAM byte ceiling for the paged block pool (#98), so
  // the UI value must reach the engine. Only Cache TTL is inapplicable to the
  // paged backend.
  return {
    memoryBudgetControlsDisabled: false,
    cacheTtlDisabled: effectiveUsePagedCache,
    memoryBudgetIgnored: false,
  }
}

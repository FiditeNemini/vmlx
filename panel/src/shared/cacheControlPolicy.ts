export type CacheControlKey =
  | 'enablePrefixCache'
  | 'usePagedCache'
  | 'enableDiskCache'
  | 'enableBlockDiskCache'

export type CacheControlUpdate = [CacheControlKey, boolean]

export interface CacheControlState {
  continuousBatching: boolean
  enablePrefixCache: boolean
  usePagedCache: boolean
  enableDiskCache: boolean
  enableBlockDiskCache: boolean
  architectureRequiresPagedCache?: boolean
  /**
   * The architecture has non-KV companion state, but the engine can pair the
   * disk-only KV block index with a typed companion SSD store/rederive path.
   * This is true for hybrid/Mamba SSM caches and native rotating/mixed-SWA
   * caches whose typed metadata is carried in each block record. This includes
   * DSV4, M3, and hybrid/Mamba cache contracts with a complete
   * typed block-disk representation. It remains false for ZAYA/openPangu contracts
   * that still require their architecture-specific in-memory or prompt-level
   * cache lane.
   */
  architectureSupportsBlockDiskOnly?: boolean
}

export interface CacheControlPolicy {
  batchingOff: boolean
  prefixOff: boolean
  architectureRequiresPagedCache: boolean
  architectureForcedPagedActive: boolean
  userPagedCacheActive: boolean
  effectiveUsePagedCache: boolean
  pagedCacheDisabled: boolean
  legacyDiskCacheDisabled: boolean
  blockDiskCacheVisible: boolean
  blockDiskCacheDisabled: boolean
  legacyDiskCacheChecked: boolean
  blockDiskCacheChecked: boolean
  legacyDiskCacheUnavailableReason?: 'batching-off' | 'paged-cache-active' | 'architecture-requires-paged-cache'
}

export interface CacheLaunchPolicy {
  prefixCacheOff: boolean
  effectiveUsePagedCache: boolean
  enableLegacyDiskCache: boolean
  enableBlockDiskCache: boolean
}

export function resolveCacheControlPolicy(state: CacheControlState): CacheControlPolicy {
  const batchingOff = !state.continuousBatching
  const prefixOff = !state.enablePrefixCache
  // In-RAM paged cache is OFF for EVERY family; SSD block-disk L2 is the only
  // tier. The control stays visible but is never active and never checked, so
  // the UI cannot disagree with what resolveCacheLaunchPolicy() launches.
  const architectureRequiresPagedCache = false
  const architectureForcedPagedActive = false
  const userPagedCacheActive = false
  const effectiveUsePagedCache = false
  const blockDiskCacheActive = !batchingOff && !prefixOff && !!state.enableBlockDiskCache
  const pagedCacheDisabled = true
  const legacyDiskCacheDisabled = batchingOff || blockDiskCacheActive
  const blockDiskCacheVisible = !batchingOff
  const blockDiskCacheDisabled = batchingOff
  const legacyDiskCacheChecked = !!state.enableDiskCache && !legacyDiskCacheDisabled && !prefixOff
  const blockDiskCacheChecked = blockDiskCacheActive
  const legacyDiskCacheUnavailableReason = batchingOff
    ? 'batching-off'
    : architectureRequiresPagedCache
      ? 'architecture-requires-paged-cache'
      : effectiveUsePagedCache
        ? 'paged-cache-active'
        : undefined

  return {
    batchingOff,
    prefixOff,
    architectureRequiresPagedCache,
    architectureForcedPagedActive,
    userPagedCacheActive,
    effectiveUsePagedCache,
    pagedCacheDisabled,
    legacyDiskCacheDisabled,
    blockDiskCacheVisible,
    blockDiskCacheDisabled,
    legacyDiskCacheChecked,
    blockDiskCacheChecked,
    legacyDiskCacheUnavailableReason,
  }
}

export function resolveCacheLaunchPolicy(state: CacheControlState): CacheLaunchPolicy {
  const batchingOff = !state.continuousBatching
  const prefixEnabled = !batchingOff && !!state.enablePrefixCache
  // In-RAM paged cache is OFF for EVERY family. SSD block-disk L2 is the only
  // cache tier this product ships. Neither a per-family registry capability
  // (`detected.usePagedCache`) nor an architecture "requires paged" claim may
  // reintroduce a RAM tier here -- both previously did, which is how 18
  // families ended up launching with --use-paged-cache.
  const effectiveUsePagedCache = false
  const enableBlockDiskCache = !!state.enableBlockDiskCache && prefixEnabled

  return {
    prefixCacheOff: !prefixEnabled,
    effectiveUsePagedCache,
    enableLegacyDiskCache: !!state.enableDiskCache && prefixEnabled && !enableBlockDiskCache,
    enableBlockDiskCache,
  }
}

export function cacheControlUpdatesForPagedToggle(_enabled: boolean, _state: CacheControlState): CacheControlUpdate[] {
  // The checkbox is disabled, but keep the handler fail-closed too. A stale
  // renderer event, automation click, or future styling regression must not be
  // able to persist a value that the launcher will ignore.
  return [['usePagedCache', false]]
}

export function cacheControlUpdatesForDiskToggle(enabled: boolean, state: CacheControlState): CacheControlUpdate[] {
  const updates: CacheControlUpdate[] = []
  if (enabled && !state.enablePrefixCache) updates.push(['enablePrefixCache', true])
  if (enabled && state.usePagedCache) updates.push(['usePagedCache', false])
  if (enabled && state.enableBlockDiskCache) updates.push(['enableBlockDiskCache', false])
  updates.push(['enableDiskCache', enabled])
  return updates
}

export function cacheControlUpdatesForBlockDiskToggle(enabled: boolean, state: CacheControlState): CacheControlUpdate[] {
  const updates: CacheControlUpdate[] = []
  if (enabled && !state.enablePrefixCache) updates.push(['enablePrefixCache', true])
  if (enabled && state.enableDiskCache) updates.push(['enableDiskCache', false])
  updates.push(['enableBlockDiskCache', enabled])
  return updates
}

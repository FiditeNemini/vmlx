import {
  resolveCacheLaunchPolicy,
  type CacheControlState,
  type CacheLaunchPolicy,
} from './cacheControlPolicy'
import {
  finiteNonNegativeNumber,
  finitePositiveInteger,
  finitePositiveNumber,
} from './launchArgValues'

/**
 * Every input used to build the prefix-cache tier arguments shared by the
 * renderer command preview and the main-process engine launcher.
 *
 * Keep this interface deliberately independent of ServerConfig/SessionConfig:
 * importing either UI or main-process types here would make the shared policy
 * depend on one of the two consumers it exists to keep in agreement.
 */
export interface CacheLaunchArgsInput extends CacheControlState {
  noMemoryAwareCache?: boolean
  /** Typed runtimes such as openPangu must keep their memory-aware clone path. */
  forceMemoryAwareCache?: boolean
  prefixCacheSize?: unknown
  prefixCacheMaxBytes?: unknown
  cacheMemoryMb?: unknown
  cacheMemoryPercent?: unknown
  cacheTtlMinutes?: unknown
  effectivePagedCacheBlockSize?: unknown
  maxCacheBlocks?: unknown
  diskCacheDir?: string
  diskCacheMaxGb?: unknown
  blockDiskCacheDir?: string
  blockDiskCacheMaxGb?: unknown
  blockDiskCacheMaxPercent?: unknown
}

export interface CacheLaunchArgsResult {
  args: string[]
  policy: CacheLaunchPolicy
  blockDiskOnly: boolean
}

/**
 * Build the authoritative cache-tier argv fragment.
 *
 * This is intentionally consumed by BOTH SessionSettings' visible command
 * preview and sessions.ts' spawned argv. Before this helper existed those two
 * hand-maintained blocks had already drifted in two user-visible ways:
 *
 *  - the preview emitted `--block-disk-cache-max-gb 0`, which means unlimited,
 *    while the launcher omitted the unset legacy value and honored the percent;
 *  - the preview could omit `--disable-block-disk-cache` while the launcher
 *    emitted it to override the engine's default-on SSD tier.
 *
 * In-RAM paged cache is retired product-wide. A stale saved `usePagedCache`
 * value is accepted only so old configs can flow through one migration-safe
 * API; it can never produce `--use-paged-cache`.
 */
export function buildCacheLaunchArgs(input: CacheLaunchArgsInput): CacheLaunchArgsResult {
  const policy = resolveCacheLaunchPolicy(input)
  // These three tokens are the product-wide retained-RAM contract. Keep them
  // ahead of every early return so disabled prefix caching cannot accidentally
  // fall back to an engine default that re-enables a hidden RAM tier.
  const args: string[] = [
    '--no-paged-cache',
    '--ssm-state-cache-size', '0',
    '--ssm-state-cache-mb', '0',
  ]
  const blockDiskOnly = policy.enableBlockDiskCache && !policy.effectiveUsePagedCache

  if (policy.prefixCacheOff) {
    args.push('--disable-prefix-cache')
    return { args, policy, blockDiskOnly }
  }

  if (input.noMemoryAwareCache && !input.forceMemoryAwareCache) {
    args.push('--no-memory-aware-cache')
    const prefixCacheSize = finitePositiveInteger(input.prefixCacheSize)
    if (prefixCacheSize != null) args.push('--prefix-cache-size', prefixCacheSize.toString())
    const prefixCacheMaxBytes = finitePositiveInteger(input.prefixCacheMaxBytes)
    if (prefixCacheMaxBytes != null) args.push('--prefix-cache-max-bytes', prefixCacheMaxBytes.toString())
  } else if (!blockDiskOnly) {
    // SSD-only block mode retains no L1 payload, so RAM-budget and TTL controls
    // are intentionally absent there. They remain meaningful for the separate
    // memory-aware/prompt-L2 lane used by exact typed runtimes.
    const cacheMemoryMb = finitePositiveInteger(input.cacheMemoryMb)
    if (cacheMemoryMb != null) args.push('--cache-memory-mb', cacheMemoryMb.toString())
    const cacheMemoryPercent = finitePositiveNumber(input.cacheMemoryPercent)
    if (cacheMemoryPercent != null) {
      args.push('--cache-memory-percent', (cacheMemoryPercent / 100).toString())
    }
    const cacheTtlMinutes = finitePositiveNumber(input.cacheTtlMinutes)
    if (cacheTtlMinutes != null) args.push('--cache-ttl-minutes', cacheTtlMinutes.toString())
  }

  if (policy.enableBlockDiskCache) {
    const blockSize = finitePositiveInteger(input.effectivePagedCacheBlockSize)
    if (blockSize != null) args.push('--paged-cache-block-size', blockSize.toString())
    const maxCacheBlocks = finitePositiveInteger(input.maxCacheBlocks)
    if (maxCacheBlocks != null) args.push('--max-cache-blocks', maxCacheBlocks.toString())
  }

  if (policy.enableLegacyDiskCache) {
    args.push('--enable-disk-cache')
    if (input.diskCacheDir) args.push('--disk-cache-dir', input.diskCacheDir)
    const diskCacheMaxGb = finiteNonNegativeNumber(input.diskCacheMaxGb)
    if (diskCacheMaxGb != null) args.push('--disk-cache-max-gb', diskCacheMaxGb.toString())
  }

  if (policy.enableBlockDiskCache) {
    args.push('--enable-block-disk-cache')
    if (input.blockDiskCacheDir) args.push('--block-disk-cache-dir', input.blockDiskCacheDir)

    // A stale zero in the removed GB control means "unset" in product state,
    // but means UNLIMITED to the engine. Only a positive legacy GB value is an
    // explicit cap. Zero/unlimited is represented by the still-visible percent
    // control, whose zero has the same engine meaning without overriding itself.
    const blockDiskCacheMaxGb = finiteNonNegativeNumber(input.blockDiskCacheMaxGb)
    if (blockDiskCacheMaxGb != null && blockDiskCacheMaxGb > 0) {
      args.push('--block-disk-cache-max-gb', blockDiskCacheMaxGb.toString())
    }
    const blockDiskCacheMaxPercent = finiteNonNegativeNumber(input.blockDiskCacheMaxPercent)
    if (blockDiskCacheMaxPercent != null) {
      args.push('--block-disk-cache-max-percent', blockDiskCacheMaxPercent.toString())
    }
  } else {
    // The engine defaults compatible SSD L2 on. A visible user opt-out must be
    // explicit in both the preview and the spawned command.
    args.push('--disable-block-disk-cache')
  }

  return { args, policy, blockDiskOnly }
}

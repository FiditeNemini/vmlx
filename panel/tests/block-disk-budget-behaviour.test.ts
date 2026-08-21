import { describe, expect, it } from 'vitest'
import { finiteNonNegativeNumber } from '../src/shared/launchArgValues'
import { DEFAULT_BLOCK_DISK_CACHE_PERCENT } from '../src/shared/cacheDefaults'

/**
 * The SSD budget, tested as BEHAVIOUR rather than as source text.
 *
 * The previous test in `block-disk-percent-reaches-argv.test.ts` grepped
 * sessions.ts for string literals. It passed while the shipped launcher handed
 * the entire disk to the cache, because:
 *
 *   1. fresh configs seeded `blockDiskCacheMaxGb: 0`
 *   2. `finiteNonNegativeNumber(0)` returns 0, NOT undefined  (0 >= 0)
 *   3. buildArgs emitted `--block-disk-cache-max-gb 0` on the `!= null` check
 *   4. the engine resolves `if explicit_gb is not None: return float(explicit_gb)`
 *   5. BlockDiskStore documents `max_size_gb: 0 = unlimited`
 *
 * So the UI showed "10%" and the engine ran with no cap at all. A grep-based
 * test cannot catch that, because every individual string it asserts was
 * present and correct. These tests model the actual decision chain.
 */

/** The emission rule buildArgs implements, isolated so it can be exercised. */
function emittedBudgetFlags(config: {
  blockDiskCacheMaxGb?: number
  blockDiskCacheMaxPercent?: number
}): string[] {
  const args: string[] = []
  const gb = finiteNonNegativeNumber(config.blockDiskCacheMaxGb)
  if (gb != null && gb > 0) args.push('--block-disk-cache-max-gb', gb.toString())
  const pct = finiteNonNegativeNumber(config.blockDiskCacheMaxPercent)
  if (pct != null) args.push('--block-disk-cache-max-percent', pct.toString())
  return args
}

/** The engine's resolver, mirrored from vmlx_engine/cli.py. */
function engineResolvesGb(args: string[], volumeTotalGb: number): number {
  const gbIdx = args.indexOf('--block-disk-cache-max-gb')
  if (gbIdx >= 0) return Number(args[gbIdx + 1])          // explicit-GB-wins
  const pctIdx = args.indexOf('--block-disk-cache-max-percent')
  const pct = pctIdx >= 0 ? Number(args[pctIdx + 1]) : DEFAULT_BLOCK_DISK_CACHE_PERCENT
  if (pct <= 0) return 0                                   // 0 == unlimited
  return (volumeTotalGb * pct) / 100
}

const VOLUME_GB = 3721   // the drive these defaults were measured on

describe('the budget the engine actually ends up with', () => {
  it('a fresh config (no GB seeded) gets the percent, NOT unlimited', () => {
    const args = emittedBudgetFlags({ blockDiskCacheMaxPercent: 10 })
    expect(args).not.toContain('--block-disk-cache-max-gb')
    expect(engineResolvesGb(args, VOLUME_GB)).toBeCloseTo(372.1, 0)
  })

  it('a stored 0 GB does NOT become an unlimited cache', () => {
    // The exact shape that shipped: seeded 0, emitted, read as unlimited.
    const args = emittedBudgetFlags({ blockDiskCacheMaxGb: 0, blockDiskCacheMaxPercent: 10 })
    expect(args).not.toContain('--block-disk-cache-max-gb')
    const resolved = engineResolvesGb(args, VOLUME_GB)
    expect(resolved).not.toBe(0)
    expect(resolved).toBeCloseTo(372.1, 0)
  })

  it('a user-chosen GB cap still wins over the percent', () => {
    const args = emittedBudgetFlags({ blockDiskCacheMaxGb: 40, blockDiskCacheMaxPercent: 10 })
    expect(engineResolvesGb(args, VOLUME_GB)).toBe(40)
  })

  it('unlimited is reachable — via the percent slider, which is the only control', () => {
    const args = emittedBudgetFlags({ blockDiskCacheMaxPercent: 0 })
    expect(engineResolvesGb(args, VOLUME_GB)).toBe(0)
  })

  it('the percent scales with the volume instead of being a flat number', () => {
    const args = emittedBudgetFlags({ blockDiskCacheMaxPercent: 10 })
    expect(engineResolvesGb(args, 256)).toBeCloseTo(25.6, 1)
    expect(engineResolvesGb(args, 4000)).toBeCloseTo(400, 1)
  })

  it('finiteNonNegativeNumber(0) returns 0 — the trap this all hinged on', () => {
    // Pinned explicitly: a `!= null` guard does NOT filter 0, and every
    // "0 means unset" assumption built on that guard is wrong.
    expect(finiteNonNegativeNumber(0)).toBe(0)
    expect(finiteNonNegativeNumber(0) != null).toBe(true)
    expect(finiteNonNegativeNumber(undefined)).toBeUndefined()
  })
})

describe('the launcher and the stored config agree', () => {
  const sessions = readSessions()

  function readSessions() {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require('fs').readFileSync('src/main/sessions.ts', 'utf-8') as string
  }

  it('buildArgs guards the GB flag on > 0, not on != null', () => {
    expect(sessions).toContain('blockDiskCacheMaxGb != null && blockDiskCacheMaxGb > 0')
  })

  it('no code path seeds a GB value nobody chose', () => {
    expect(sessions).not.toContain("setConfigValue(mutable, 'blockDiskCacheMaxGb', 0)")
    expect(sessions).not.toContain("setConfigValue(mutable, 'blockDiskCacheMaxGb', 10)")
    // The adopted-session path used an object literal, which is why the
    // previous grep-based guard missed it.
    expect(sessions).not.toMatch(/blockDiskCacheMaxGb:\s*10\b/)
  })

  it('the migration DELETES the legacy GB rather than writing 0', () => {
    expect(sessions).toContain('delete config.blockDiskCacheMaxGb')
  })

  it('changing the percent requires a restart, like every other launch flag', () => {
    const idx = sessions.indexOf('RESTART_REQUIRED_KEYS')
    const block = sessions.slice(idx, idx + 1200)
    expect(block).toContain("'blockDiskCacheMaxPercent'")
  })
})

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'

/**
 * The SSD budget slider must reach the engine.
 *
 * v1.6.35 moved the block-disk budget from a flat GB number to a percent of the
 * volume. The engine resolves explicit-GB-wins, so passing only
 * `--block-disk-cache-max-gb` pins the budget and makes the percent slider
 * decorative — which is exactly what shipped in the first cut: fresh configs
 * seeded `blockDiskCacheMaxGb: 10` and buildArgs emitted only the GB flag, so
 * the 10%-of-disk default could never take effect.
 *
 * These assertions read the REAL sessions.ts, deliberately. `settings-flow.test.ts`
 * builds args through a hand-written mirror of buildArgs; a mirror agrees with
 * itself no matter what the shipped launcher does, and that is why it stayed
 * green through the defect. Anything asserting "the user's setting reaches the
 * process" has to look at the source that spawns the process.
 */
describe('block-disk budget percent reaches the engine argv', () => {
  const sessions = readFileSync('src/main/sessions.ts', 'utf-8')

  it('buildArgs emits --block-disk-cache-max-percent', () => {
    expect(sessions).toContain("args.push('--block-disk-cache-max-percent', blockDiskCacheMaxPercent.toString())")
  })

  it('emits the percent alongside the GB flag, not as an either/or', () => {
    const gbIdx = sessions.indexOf("args.push('--block-disk-cache-max-gb', blockDiskCacheMaxGb.toString())")
    const pctIdx = sessions.indexOf("args.push('--block-disk-cache-max-percent', blockDiskCacheMaxPercent.toString())")
    expect(gbIdx).toBeGreaterThan(-1)
    expect(pctIdx).toBeGreaterThan(gbIdx)
    // Between them there must be no `else` — an else would mean a session
    // carrying any GB value silently suppresses the percent.
    expect(sessions.slice(gbIdx, pctIdx)).not.toMatch(/\belse\b/)
  })

  it('fresh configs seed 0 GB so the percent owns the budget', () => {
    expect(sessions).toContain("setConfigValue(mutable, 'blockDiskCacheMaxGb', 0)")
    expect(sessions).toContain(
      "setConfigValue(mutable, 'blockDiskCacheMaxPercent', DEFAULT_BLOCK_DISK_CACHE_PERCENT)",
    )
    // The old flat seed is what made the percent dead on arrival.
    expect(sessions).not.toContain("setConfigValue(mutable, 'blockDiskCacheMaxGb', 10)")
  })

  it('both arg allow-lists know the percent flag', () => {
    // Two lists gate which flags survive engine adoption / arg comparison. A
    // flag missing from either is silently dropped when an existing engine is
    // reused, so the setting appears to work and then does not.
    const occurrences = sessions.split("'--block-disk-cache-max-percent',").length - 1
    expect(occurrences).toBeGreaterThanOrEqual(2)
  })

  it('the default percent has exactly one definition', () => {
    const shared = readFileSync('src/shared/cacheDefaults.ts', 'utf-8')
    expect(shared).toContain('export const DEFAULT_BLOCK_DISK_CACHE_PERCENT = 10')
    const form = readFileSync(
      'src/renderer/src/components/sessions/SessionConfigForm.tsx',
      'utf-8',
    )
    expect(form).toContain('blockDiskCacheMaxPercent: DEFAULT_BLOCK_DISK_CACHE_PERCENT')
    expect(sessions).toContain('DEFAULT_BLOCK_DISK_CACHE_PERCENT')
  })
})

describe('the v17 SSD-first default runs as a post-pass', () => {
  const sessions = readFileSync('src/main/sessions.ts', 'utf-8')

  it('applies after the legacy tuple migrations, not before', () => {
    const legacyIdx = sessions.indexOf('const legacyChanged = applyLegacyCacheStackMigrations(config, modelPath)')
    const ssdIdx = sessions.indexOf('const ssdFirstChanged = applySsdFirstCacheDefaults(')
    expect(legacyIdx).toBeGreaterThan(-1)
    expect(ssdIdx).toBeGreaterThan(legacyIdx)
  })

  it('skips passes where the version stamp declined', () => {
    // markCacheStackStartupDefaultsCurrent refuses to stamp while the bundle is
    // unreachable so the migration can retry later. Mutating usePagedCache or
    // blockDiskCacheMaxGb on such a pass rewrites the exact-tuple fingerprint
    // the retry matches on, and the retry then never fires.
    expect(sessions).toContain(
      'Number(config.cacheStackStartupDefaultsVersion || 0) !==\n    CACHE_STACK_STARTUP_DEFAULTS_VERSION',
    )
  })

  it('exempts the families whose runtime policy contradicts SSD-only', () => {
    const fn = sessions.slice(
      sessions.indexOf('function applySsdFirstCacheDefaults('),
      sessions.indexOf('function applyCacheStackStartupDefaultMigration('),
    )
    // Persisting a value the launcher overrules shows the checkbox Off while
    // the engine runs the RAM tier.
    expect(fn).toContain('isZayaCacheStackMigrationTarget')
    expect(fn).toContain("'openpangu_v2'")
  })
})

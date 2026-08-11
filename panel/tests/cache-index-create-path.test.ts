import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'

/**
 * An adversarial audit found the cache-index fix covered only the adopted-process
 * creation site. The Create Session form — how nearly every session is born —
 * still shipped `maxCacheBlocks: 1000` and sent it explicitly, and the main
 * process then stamped the session current so no migration could ever lift it.
 * The merge-over-existing path was worse: it ran the migration and then let the
 * incoming 1000 clobber the freshly lifted value.
 *
 * 1000 blocks x 64 tokens addresses only 63,936 tokens, silently capping prefix
 * reuse below the context window. These are structural invariants, so they are
 * asserted against the source rather than a spun-up session.
 */
const root = join(__dirname, '..')
const sessions = readFileSync(join(root, 'src/main/sessions.ts'), 'utf8')
const form = readFileSync(
  join(root, 'src/renderer/src/components/sessions/SessionConfigForm.tsx'),
  'utf8',
)

describe('cache index is never created at the stale flat 1000', () => {
  it('the Create Session form default is capacity-derived, not 1000', () => {
    expect(form).not.toMatch(/^\s*maxCacheBlocks:\s*1000,\s*$/m)
    expect(form).toMatch(/^\s*maxCacheBlocks:\s*4097,\s*$/m)
  })

  it('the main process backstops every create path', () => {
    expect(sessions).toContain('function liftStaleFlatCacheIndex(')
    // fresh create + merge-over-existing both call it
    const calls = sessions.match(/liftStaleFlatCacheIndex\(/g) || []
    expect(calls.length).toBeGreaterThanOrEqual(3) // definition + 2 call sites
  })

  it('the backstop runs BEFORE the version is stamped current', () => {
    // Stamping first would mark the session migrated while still holding 1000.
    const lift = sessions.indexOf('liftStaleFlatCacheIndex(config, modelPath)')
    const mark = sessions.indexOf('markCacheStackStartupDefaultsCurrent(config, modelPath)')
    expect(lift).toBeGreaterThan(-1)
    expect(mark).toBeGreaterThan(-1)
    expect(lift).toBeLessThan(mark)
  })

  it('the version stamp is withheld while the bundle is unreachable', () => {
    // Models live on an external drive, so a launch with the drive unmounted is
    // routine. Stamping then consumes a detection-dependent migration without
    // performing it — permanently. The retry guard must cover every version
    // from 12 up to current, not just 12.
    expect(sessions).toMatch(
      /pendingVersion >= 12 && pendingVersion < CACHE_STACK_STARTUP_DEFAULTS_VERSION/,
    )
  })
})

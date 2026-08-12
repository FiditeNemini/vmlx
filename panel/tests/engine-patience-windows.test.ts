import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import {
  ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS,
  HEALTH_POLL_INTERVAL_SECONDS,
  MIN_HEALTH_FAIL_COUNT,
  healthFailureToleranceCount,
} from '../src/shared/enginePatienceWindows'

/**
 * A large fresh prefill blocks the engine's event loop, so /health does not
 * answer while it runs. MEASURED on the box (DSV4-Flash, 249k cached + 83k
 * fresh): no /health response within 60s, process R at 77% CPU, RSS flat.
 *
 * Two components then judge that silence independently. The engine's streaming
 * guard waits `timeout` and, because it cannot read prefill progress, spends
 * further grace windows. The panel's health poll declares the session down
 * after N failures. When the panel is less patient than the engine, the user
 * watches the session die while the engine is still computing their answer.
 */
describe('panel health tolerance vs engine patience', () => {
  it('waits at least as long as the engine will, measured as ELAPSED time', () => {
    for (const timeout of [300, 900, 1800, 2400]) {
      // The caller acts ON the Nth failure, so N failures span (N-1) intervals.
      // Measuring N*interval overstates the wait and hid a one-interval early
      // give-up against an endpoint that refuses immediately.
      const failures = healthFailureToleranceCount(timeout)
      const elapsedBeforeGivingUp = (failures - 1) * HEALTH_POLL_INTERVAL_SECONDS
      const enginePatience = timeout * (1 + ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS)
      expect(elapsedBeforeGivingUp).toBeGreaterThanOrEqual(enginePatience)
    }
  })

  it('covers the full engine patience at the shipped 900s slow-family timeout', () => {
    // Old rule was ceil(timeout / 5) polls = 900s of tolerance.
    const failures = healthFailureToleranceCount(900)
    expect((failures - 1) * HEALTH_POLL_INTERVAL_SECONDS).toBe(2700)
  })

  it('keeps a floor when no timeout is configured', () => {
    for (const bad of [undefined, 0, -1, Number.NaN, 'x' as unknown as number]) {
      expect(healthFailureToleranceCount(bad as number)).toBe(MIN_HEALTH_FAIL_COUNT)
    }
  })

  it('mirrors the engine constant it is derived from', () => {
    // If the Python grace changes and this does not, the panel silently goes
    // back to being the impatient one.
    const server = readFileSync(
      resolve(__dirname, '../../vmlx_engine/server.py'),
      'utf-8',
    )
    const match = server.match(/_UNKNOWN_PROGRESS_GRACE_WINDOWS\s*=\s*(\d+)/)
    expect(match).not.toBeNull()
    expect(Number(match![1])).toBe(ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS)
  })

  it('sessions.ts uses the shared helper, not its own arithmetic', () => {
    const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf-8')
    expect(source).toContain('healthFailureToleranceCount(')
    expect(source).not.toContain('Math.ceil(cfg.timeout / 5)')
  })
})

/**
 * The regression that made the fix above only half-work: the health poll read
 * `config.timeout`, but that is not what the engine was given. Slow families
 * keep the generic 300 in storage and are lifted to 900 at launch, so the poll
 * was scaling its patience from a number the engine never saw — for exactly
 * the families whose prefills run longest.
 */
describe('health tolerance resolves the family timeout, not the stored one', () => {
  const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf-8')

  it('the poll and the --timeout arg call the same resolver', () => {
    expect(source).toContain('healthFailureToleranceCount(resolvedEngineTimeoutSeconds(cfg))')
    expect(source).toContain("args.push('--timeout', resolvedEngineTimeoutSeconds(config).toString())")
    expect(source).not.toContain('healthFailureToleranceCount(cfg.timeout)')
  })

  it('every slow family is more patient than the generic default', async () => {
    const { SLOW_FAMILY_TIMEOUTS, GENERIC_DEFAULT_TIMEOUT_SECONDS, resolveSlowFamilyTimeoutSeconds } =
      await import('../src/shared/slowFamilyTimeouts')
    const generic = healthFailureToleranceCount(GENERIC_DEFAULT_TIMEOUT_SECONDS)
    for (const family of Object.keys(SLOW_FAMILY_TIMEOUTS)) {
      // a session still on the generic stored value, as these families are
      const resolved = resolveSlowFamilyTimeoutSeconds(GENERIC_DEFAULT_TIMEOUT_SECONDS, family)
      expect(resolved).toBe(SLOW_FAMILY_TIMEOUTS[family])
      expect(healthFailureToleranceCount(resolved)).toBeGreaterThan(generic)
      // and it must still cover the engine's full patience for that timeout
      expect(healthFailureToleranceCount(resolved) * HEALTH_POLL_INTERVAL_SECONDS)
        .toBeGreaterThanOrEqual(resolved * (1 + ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS))
    }
  })
})

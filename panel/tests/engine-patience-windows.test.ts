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
  it('waits at least as long as the engine will', () => {
    for (const timeout of [300, 900, 1800, 2400]) {
      const tolerated = healthFailureToleranceCount(timeout) * HEALTH_POLL_INTERVAL_SECONDS
      const enginePatience = timeout * (1 + ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS)
      expect(tolerated).toBeGreaterThanOrEqual(enginePatience)
    }
  })

  it('is 3x the old behaviour at the shipped 900s slow-family timeout', () => {
    // Old rule was ceil(timeout / 5) polls = 900s of tolerance.
    expect(healthFailureToleranceCount(900) * HEALTH_POLL_INTERVAL_SECONDS).toBe(2700)
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
    expect(source).toContain('healthFailureToleranceCount(cfg.timeout)')
    expect(source).not.toContain('Math.ceil(cfg.timeout / 5)')
  })
})

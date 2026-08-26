/**
 * ONE definition of how long a silent engine may be presumed alive.
 *
 * A large fresh prefill blocks the engine's event loop, so `/health` does not
 * answer at all while it runs. MEASURED on the box (DSV4-Flash, 249k cached +
 * 83k fresh): the process is `R`, 77% CPU, RSS flat — and `/health` does not
 * respond within 60s. Cold prefills of that span took ~780s.
 *
 * Two components decide independently whether that silence means "busy" or
 * "dead", and they must not disagree:
 *
 *   - the ENGINE's streaming guard (`_stream_with_keepalive`) waits
 *     `timeout`, then spends `_UNKNOWN_PROGRESS_GRACE_WINDOWS` further windows
 *     when it cannot read progress — which is always, during prefill, because
 *     `request_progress` counts output tokens only.
 *   - the PANEL's health poll marks the session down after N consecutive
 *     failures.
 *
 * Before this was shared, the engine would patiently wait 3 x timeout for a
 * prefill the panel had already declared dead at 1 x timeout. The user would
 * see the session go down while the engine was still computing their answer.
 */

/**
 * Extra timeout windows the engine spends when it cannot read progress.
 *
 * Mirrors `_UNKNOWN_PROGRESS_GRACE_WINDOWS` in vmlx_engine/server.py. If that
 * constant changes, this must change with it — the Python test
 * `test_stream_keepalive_timeout.py` and the panel test
 * `engine-patience-windows.test.ts` both pin the relationship.
 */
export const ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS = 2

/** Health poll interval, in seconds. */
export const HEALTH_POLL_INTERVAL_SECONDS = 5

/** Floor, for sessions with no configured timeout. 60 polls = 5 minutes. */
export const MIN_HEALTH_FAIL_COUNT = 60

/**
 * After the configured startup window, a model that is still making observable
 * load progress gets this much quiet time before the panel calls it stalled.
 * This is deliberately an idle window, not an unbounded timeout extension.
 */
export const STARTUP_PROGRESS_IDLE_GRACE_MS = 120_000

/** Absolute ceiling for a progress-extended startup. */
export const STARTUP_HARD_TIMEOUT_MULTIPLIER = 3

/**
 * Decide whether a startup poll may continue.
 *
 * The configured timeout remains the normal deadline. Beyond it, only recent
 * resident/log progress can keep the wait alive, and never beyond the hard
 * multiple. This prevents an 84 GB model that is visibly loading from being
 * marked Error at 900 seconds while still ensuring a hung process terminates.
 */
export function shouldContinueStartupWait(
  elapsedMs: number,
  configuredTimeoutMs: number,
  lastProgressAgeMs?: number,
): boolean {
  const timeoutMs = Math.max(1, configuredTimeoutMs)
  if (elapsedMs < timeoutMs) return true
  if (elapsedMs >= timeoutMs * STARTUP_HARD_TIMEOUT_MULTIPLIER) return false
  return lastProgressAgeMs != null
    && lastProgressAgeMs >= 0
    && lastProgressAgeMs <= STARTUP_PROGRESS_IDLE_GRACE_MS
}

/**
 * How many consecutive health failures to tolerate before declaring a session
 * down, given its configured request timeout in seconds.
 *
 * The engine will keep a silent request alive for `timeout * (1 + grace)`, so
 * the panel waits at least that long before contradicting it.
 */
export function healthFailureToleranceCount(timeoutSeconds: number | undefined): number {
  const timeout = Number(timeoutSeconds)
  if (!Number.isFinite(timeout) || timeout <= 0) return MIN_HEALTH_FAIL_COUNT
  const enginePatienceSeconds = timeout * (1 + ENGINE_UNKNOWN_PROGRESS_GRACE_WINDOWS)
  // +1 because the caller acts ON the Nth failure, so N failures only span
  // (N-1) intervals of elapsed time. Without it, an endpoint that refuses
  // immediately (rather than hanging) is declared down one interval EARLY --
  // 2695s against 2700s of engine patience at the 900s slow-family timeout.
  // Small, but the whole point of this function is "at least as long", and
  // that has to be true rather than nearly true.
  return Math.max(
    MIN_HEALTH_FAIL_COUNT,
    Math.ceil(enginePatienceSeconds / HEALTH_POLL_INTERVAL_SECONDS) + 1,
  )
}

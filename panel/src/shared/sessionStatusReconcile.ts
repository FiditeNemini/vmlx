/**
 * Boot-time reconciliation of stale session runtime status.
 *
 * 2026-08-17 (ledger row 275): reproduced live in the dev app — the Sessions
 * view showed a red **Error** badge on two models that then loaded cleanly on
 * the first try (qwen3_5 and nemotron_h both reached `model_loaded: true`), and
 * one of those cards additionally displayed a DIFFERENT model's name than the
 * bundle at its path. Nothing ever cleared those rows, so the badge survived
 * app restarts until the user happened to start that session again. Users read
 * that as "the app logs errors when loading my model".
 *
 * `running` / `loading` / `standby` / `error` all describe the CURRENT
 * process's runtime. After the app exits no engine child survives, so on the
 * next boot none of them can still be true — a start has not even been
 * attempted yet. They are reset to `stopped` with the stale pid dropped.
 *
 * Lives in shared/ (no better-sqlite3 import) so it is testable: the native
 * module is built for Electron's ABI and cannot load under vitest.
 */

/** Statuses that describe this process's runtime and cannot survive a restart. */
export const TRANSIENT_SESSION_STATUSES = [
  "running",
  "loading",
  "standby",
  "error",
] as const

/** Statuses that are durable and must NOT be rewritten at boot. */
export const DURABLE_SESSION_STATUSES = ["stopped"] as const

export function isTransientSessionStatus(status: string): boolean {
  return (TRANSIENT_SESSION_STATUSES as readonly string[]).includes(status)
}

function statusList(): string {
  return TRANSIENT_SESSION_STATUSES.map((s) => `'${s}'`).join(",")
}

/** Counts rows a boot reconciliation would touch. */
export function staleSessionCountSql(): string {
  return `SELECT COUNT(*) as cnt FROM sessions WHERE status IN (${statusList()})`
}

/**
 * Resets stale rows to `stopped` and drops the stale pid.
 * model_name is deliberately NOT rewritten — the start path re-resolves it
 * from the bundle, which is how the mis-titled card corrected itself live.
 */
export function staleSessionResetSql(): string {
  return `UPDATE sessions SET status = 'stopped', pid = NULL WHERE status IN (${statusList()})`
}

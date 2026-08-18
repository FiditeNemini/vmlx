import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  DURABLE_SESSION_STATUSES,
  TRANSIENT_SESSION_STATUSES,
  isTransientSessionStatus,
  staleSessionCountSql,
  staleSessionResetSql,
} from '../src/shared/sessionStatusReconcile'

// 2026-08-17 (ledger row 275): reproduced live in the dev app — the Sessions
// view showed a red Error badge on two models that then loaded cleanly on the
// first try, and one card displayed a DIFFERENT model's name than the bundle at
// its path. Nothing cleared those rows, so the stale badge survived app
// restarts until the user happened to start that session again. Users read that
// as "the app logs errors when loading my model".
//
// better-sqlite3 is built for Electron's ABI and cannot load under vitest, so
// the rule lives in shared/ and is exercised here directly; a source assertion
// keeps database.ts wired to it.

describe('stale session status reconciliation', () => {
  it('treats every runtime status as transient and stopped as durable', () => {
    expect([...TRANSIENT_SESSION_STATUSES].sort()).toEqual(
      ['error', 'loading', 'running', 'standby'].sort(),
    )
    expect([...DURABLE_SESSION_STATUSES]).toEqual(['stopped'])

    // the badge users complained about
    expect(isTransientSessionStatus('error')).toBe(true)
    // impossible after the app exits
    expect(isTransientSessionStatus('running')).toBe(true)
    expect(isTransientSessionStatus('loading')).toBe(true)
    expect(isTransientSessionStatus('standby')).toBe(true)
    // must never be rewritten
    expect(isTransientSessionStatus('stopped')).toBe(false)
  })

  it('both SQL statements target exactly the transient set, derived from one constant', () => {
    const countSql = staleSessionCountSql()
    const resetSql = staleSessionResetSql()

    for (const status of TRANSIENT_SESSION_STATUSES) {
      expect(countSql).toContain(`'${status}'`)
      expect(resetSql).toContain(`'${status}'`)
    }
    // a durable status must never appear in the IN(...) list
    for (const status of DURABLE_SESSION_STATUSES) {
      expect(countSql).not.toContain(`IN ('${status}'`)
      expect(resetSql.split('WHERE')[1]).not.toContain(`'${status}'`)
    }
    // the reset clears BOTH the badge and the stale pid
    expect(resetSql).toContain("SET status = 'stopped'")
    expect(resetSql).toContain('pid = NULL')
    // model_name is deliberately untouched: the start path re-resolves it from
    // the bundle, which is how the mis-titled card corrected itself live.
    expect(resetSql).not.toContain('model_name')
    // the two statements cannot drift: same WHERE clause
    expect(countSql.split('WHERE')[1]).toBe(resetSql.split('WHERE')[1])
  })

  it('database.ts runs the reconciliation on every boot via the shared rule', () => {
    const source = readFileSync('src/main/database.ts', 'utf8')
    expect(source).toContain('staleSessionCountSql()')
    expect(source).toContain('staleSessionResetSql()')
    expect(source).toContain('this.reconcileStaleSessionStatuses()')
    // must NOT be gated behind a one-time migration key: a crash can strand
    // the state at any time, so it has to run on every boot.
    const fn = source.slice(source.indexOf('reconcileStaleSessionStatuses(): void'))
    expect(fn.slice(0, 1200)).not.toContain('FROM settings WHERE key')
  })
})

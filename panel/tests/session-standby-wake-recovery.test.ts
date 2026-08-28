import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { resolve } from 'path'

/**
 * A failed wake must not pin the session at status 'loading' forever.
 *
 * The /health handler branches on the reply in order. The recovery for "server
 * says standby but our DB still says loading" used to sit AFTER a bare
 * `if (isStandby)`, which swallowed every standby reply first — so the
 * recovery was unreachable. The visible result was a progress bar frozen
 * mid-fill with no error and no way to retry, because the branch that ran
 * instead deletes the fail counter, so the session could never time out
 * either.
 *
 * This asserts the ORDER, which is the whole defect: the loading-qualified
 * standby test must be evaluated before the general one.
 */
describe('standby wake recovery', () => {
  const src = readFileSync(
    resolve(__dirname, '../src/main/sessions.ts'),
    'utf-8',
  )

  it('checks standby+loading before the general standby branch', () => {
    const qualified = src.indexOf("isStandby && session.status === 'loading'")
    const general = src.indexOf('} else if (isStandby) {')

    expect(qualified, 'the standby+loading recovery branch is missing').toBeGreaterThan(-1)
    expect(general, 'the general standby branch is missing').toBeGreaterThan(-1)
    expect(
      qualified,
      'the standby+loading recovery sits after the general isStandby branch, ' +
        'so it is unreachable and a failed wake pins the session at "loading"',
    ).toBeLessThan(general)
  })

  it('still reverts the session to standby so the user can retry', () => {
    const start = src.indexOf("isStandby && session.status === 'loading'")
    const body = src.slice(start, start + 2000)
    expect(body).toContain("status: 'standby'")
    expect(body).toContain('session:standby')
  })
})

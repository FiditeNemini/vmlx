import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

/**
 * A downed session must release the chat locks held against its endpoint.
 *
 * chat:sendMessage keeps one activeRequests entry per chat and rejects a second
 * message with "a message is already being generated for this chat". The
 * renderer shows that rejection as a toast, which auto-dismisses -- so a lock
 * that outlives its engine makes the chat look silently broken: the user's
 * message renders, nothing answers, and the engine is idle.
 *
 * MEASURED in the dev app before this fix: no reply for 300s with the engine at
 * 0.3% CPU; a brand-new chat answered the same prompt in 4.95s.
 */
describe('handleSessionDown aborts in-flight inference', () => {
  const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf-8')
  const body = source.slice(source.indexOf('private handleSessionDown') >= 0
    ? source.indexOf('private handleSessionDown')
    : source.indexOf('handleSessionDown('))

  it('emits the abort before the running/loading status check', () => {
    const abortAt = body.indexOf("this.emit('session:abortInference'")
    const statusGate = body.indexOf("session.status === 'running'")
    expect(abortAt).toBeGreaterThan(-1)
    expect(statusGate).toBeGreaterThan(-1)
    // Aborting must come FIRST: a session already marked `error` never enters
    // the status branch, and would otherwise keep its locks forever.
    expect(abortAt).toBeLessThan(statusGate)
  })

  it('emits the abort before the remote branch returns', () => {
    const abortAt = body.indexOf("this.emit('session:abortInference'")
    const remoteBranch = body.indexOf("session.type === 'remote'")
    expect(remoteBranch).toBeGreaterThan(-1)
    // The remote branch returns early; if the abort sat after it, an
    // unreachable remote endpoint would strand every chat bound to it.
    expect(abortAt).toBeLessThan(remoteBranch)
  })

  it('emits it exactly once', () => {
    const occurrences = body.split("this.emit('session:abortInference'").length - 1
    expect(occurrences).toBe(1)
  })
})

describe('the chat lock cannot outlive an aborted request', () => {
  const chat = readFileSync(resolve(__dirname, '../src/main/ipc/chat.ts'), 'utf-8')

  it('clears a lock whose controller is already aborted', () => {
    expect(chat).toContain('existing.controller.signal.aborted')
    expect(chat).toContain('if (abandoned || age > staleLockMs)')
  })

  it('does not claim a 10-minute cap while coding a 30-minute one', () => {
    const window = chat.slice(chat.indexOf('const staleLockMs') - 300,
                              chat.indexOf('const staleLockMs') + 200)
    expect(window).not.toMatch(/Cap at 10 minutes/)
    expect(window).toContain('30 * 60 * 1000')
  })
})

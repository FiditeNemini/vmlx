import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import ts from 'typescript'

// Regression guard for the stuck "Waiting for response..." composer state.
//
// chat:send-message registers an activeRequests entry BEFORE endpoint
// resolution so the concurrency guard and Stop button work during setup.
// Setup can throw (renderer endpoint validation, resolveServerEndpoint /
// resolveUrl rejections, health-check exhaustion while an engine is still
// loading, request building). Observed live 2026-08-07: a setup throw leaked
// the entry, chat:isStreaming stayed true, and the chat was locked in
// "streaming" state until the stale-lock window (up to 30min) or a manual
// Stop. Every one of those throw sites must sit inside a try whose finally
// removes the entry.

const source = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')
const sf = ts.createSourceFile('chat.ts', source, ts.ScriptTarget.Latest, true)

function collectTries(): ts.TryStatement[] {
  const tries: ts.TryStatement[] = []
  const walk = (n: ts.Node) => {
    if (ts.isTryStatement(n)) tries.push(n)
    n.forEachChild(walk)
  }
  walk(sf)
  return tries
}

const tries = collectTries()

function coveredByEntryCleanup(pos: number): boolean {
  return tries.some(
    (t) =>
      t.tryBlock.getStart() <= pos &&
      t.tryBlock.end >= pos &&
      (t.finallyBlock?.getText() ?? '').includes('activeRequests.delete(chatId)'),
  )
}

describe('activeRequests entry lifecycle in chat:send-message', () => {
  it('covers every setup-phase throw site with an entry-deleting finally', () => {
    const setupThrowMarkers = [
      'not allowed — must be localhost', // renderer endpoint validation
      'Cannot reach server on port', // health-check exhaustion (engine loading)
      'no native Thinking Off/Instruct mode', // instruct-mode capability throw
    ]
    for (const marker of setupThrowMarkers) {
      const pos = source.indexOf(marker)
      expect(pos, `marker not found: ${marker}`).toBeGreaterThan(-1)
      expect(coveredByEntryCleanup(pos), `throw not covered by entry cleanup: ${marker}`).toBe(true)
    }
  })

  it('covers endpoint resolution awaits with an entry-deleting finally', () => {
    // resolveServerEndpoint / resolveUrl rejections must not leak the entry
    const pos = source.indexOf('await resolveServerEndpoint(chat.modelPath)')
    expect(pos).toBeGreaterThan(-1)
    expect(coveredByEntryCleanup(pos)).toBe(true)
  })

  it('registers the entry before setup so Stop works during endpoint resolution', () => {
    const setPos = source.indexOf('activeRequests.set(chatId, {')
    const resolvePos = source.indexOf('await resolveServerEndpoint(chat.modelPath)')
    expect(setPos).toBeGreaterThan(-1)
    expect(setPos).toBeLessThan(resolvePos)
  })

  it('guards the delete so a newer request entry is never removed by a stale one', () => {
    const guarded = tries.filter((t) =>
      (t.finallyBlock?.getText() ?? '').includes('activeRequests.delete(chatId)'),
    )
    expect(guarded.length).toBeGreaterThan(0)
    for (const t of guarded) {
      expect(t.finallyBlock!.getText()).toContain('current.controller === abortController')
    }
  })
})

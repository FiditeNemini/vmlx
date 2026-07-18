import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

// CHAT-SESSION-MODELPATH-IDENTITY-MISMATCH regression: a chat pinned to a
// stopped duplicate session (same model identity under a different path
// prefix) must resolve to the USABLE twin for endpoint/status display, so
// the composer/banner never claim not-running while the model's server runs.
describe('active session usable-twin fallback', () => {
  const source = readFileSync(
    new URL('../src/renderer/src/App.tsx', import.meta.url),
    'utf8',
  )

  it('falls back from a non-usable pinned session to a usable same-identity session', () => {
    expect(source).toContain('const pinnedSession = sessions.find(s => s.id === state.activeSessionId)')
    expect(source).toContain(
      'sessions.find(s => sessionMatchesModelPath(s.modelPath, pinnedSession.modelPath) && sessionUsable(s)) || pinnedSession',
    )
  })

  it('usable means running, loading, or standby', () => {
    expect(source).toContain(
      "(s.status === 'running' || s.status === 'loading' || s.status === 'standby')",
    )
  })
})

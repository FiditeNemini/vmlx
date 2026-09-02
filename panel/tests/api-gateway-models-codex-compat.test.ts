import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'

// codex 0.150's model-manager deserializes GET /v1/models expecting a
// top-level `models` array and logs a refresh error against a pure OpenAI
// `data` body. The gateway therefore returns BOTH: `data` (the standard field
// every OpenAI client reads) and an additive `models` mirror for codex. This
// is a source-contract test — handleListModels is private and the alternative
// would be booting a full gateway over a socket.
describe('gateway /v1/models codex compatibility', () => {
  const source = readFileSync('src/main/api-gateway.ts', 'utf8')

  it('returns both the standard data field and the additive models mirror', () => {
    expect(source).toContain('{ object: "list", data: models, models }')
  })

  it('keeps data as the primary OpenAI field', () => {
    // Guard against a future edit dropping `data` in favor of only `models`,
    // which would break every standard OpenAI client.
    const call = source.slice(source.indexOf('handleListModels'))
    expect(call).toContain('data: models')
  })
})

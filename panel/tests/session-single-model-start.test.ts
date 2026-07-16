import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('manual session single-model enforcement', () => {
  it('serializes UI starts and unloads every other local engine before launch', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('async startSession(sessionId: string)')
    const end = source.indexOf('private async _startSessionInner', start)
    const block = source.slice(start, end)

    expect(source).toContain('private singleModelStartTransitionPending: Promise<void>')
    expect(block).toContain("db.getSetting('gateway_single_model_mode') === 'true'")
    expect(block).toContain("other.type !== 'remote'")
    expect(block).toContain("['running', 'loading', 'standby'].includes(other.status)")
    expect(block).toContain('await this.stopSession(other.id)')
    expect(block.indexOf('await this.stopSession(other.id)')).toBeLessThan(
      block.indexOf('this._startSessionInner(sessionId)'),
    )
    expect(block).toContain('const previous = this.singleModelStartTransitionPending.catch')
    expect(block).toContain('this.singleModelStartTransitionPending = previous.then')
  })
})

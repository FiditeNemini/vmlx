import { describe, expect, it } from 'vitest'
import { isTypewriterAppend } from '../src/renderer/src/components/chat/typewriter'

describe('typewriter append detection', () => {
  it('treats ordinary streaming growth as an append', () => {
    expect(isTypewriterAppend('', 'Hel')).toBe(true)
    expect(isTypewriterAppend('Hel', 'Hello')).toBe(true)
    expect(isTypewriterAppend('Hello', 'Hello')).toBe(true)
  })

  it('snaps a same-length rewrite (live repro: stream showed "\\n\\n4", completion was "482")', () => {
    expect(isTypewriterAppend('\n\n4', '482')).toBe(false)
  })

  it('snaps shrinks and longer non-prefix rewrites', () => {
    expect(isTypewriterAppend('\n\n482', '482')).toBe(false)
    expect(isTypewriterAppend('abc', 'xbcdef')).toBe(false)
  })
})

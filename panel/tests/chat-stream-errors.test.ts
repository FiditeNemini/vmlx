import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  ChatStreamServerEventError,
  shouldRethrowChatStreamLineError,
} from '../src/shared/chatStreamErrors'

describe('chat SSE error propagation', () => {
  it('distinguishes intentional server failures from malformed ordinary lines', () => {
    expect(
      shouldRethrowChatStreamLineError(
        new ChatStreamServerEventError('Server error: vision prefill failed'),
        false,
      ),
    ).toBe(true)
    expect(shouldRethrowChatStreamLineError(new Error('bad optional field'), false)).toBe(false)
    expect(shouldRethrowChatStreamLineError(new Error('socket closed'), true)).toBe(true)
  })

  it('wires both Responses and Chat Completions error chunks through the rethrow gate', () => {
    const source = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')

    expect(source.match(/throw new ChatStreamServerEventError/g)).toHaveLength(2)
    expect(source).toContain('shouldRethrowChatStreamLineError(')
    expect(source).toContain('isExpectedChatBackendDisconnectError(e)')
    expect(source.indexOf('shouldRethrowChatStreamLineError(')).toBeLessThan(
      source.indexOf('// Skip malformed JSON lines'),
    )
  })
})

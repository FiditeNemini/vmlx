import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  ChatStreamServerEventError,
  chatStreamServerEventErrorDetail,
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

  it('extracts nested response.failed errors without losing the terminal payload', () => {
    expect(
      chatStreamServerEventErrorDetail(
        {
          type: 'response.failed',
          response: {
            status: 'failed',
            error: { type: 'server_error', message: 'engine failed after output' },
            usage: { input_tokens: 5, output_tokens: 7, total_tokens: 12 },
          },
        },
        'response.failed',
      ),
    ).toBe('engine failed after output')
  })

  it('wires both Responses and Chat Completions error chunks through the rethrow gate', () => {
    const source = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')

    expect(source.match(/pendingStreamServerError = new ChatStreamServerEventError/g)).toHaveLength(2)
    expect(source.match(/if \(pendingStreamServerError\) throw pendingStreamServerError/g)).toHaveLength(2)
    expect(source).toContain('responsesEventType === "response.failed"')
    expect(source).toContain('shouldRethrowChatStreamLineError(')
    expect(source).toContain('isExpectedChatBackendDisconnectError(e)')
    expect(source.indexOf('shouldRethrowChatStreamLineError(')).toBeLessThan(
      source.indexOf('// Skip malformed JSON lines'),
    )
  })
})

describe('chat timeout liveness', () => {
  const source = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')

  it('treats the saved session timeout as inactivity rather than total wall time', () => {
    expect(source).toContain('const fetchInactivitySeconds = isRemote')
    expect(source).toContain(': Math.max(timeoutSeconds, 30)')
    expect(source.indexOf('const isRemote = chatSession?.type === "remote"')).toBeLessThan(
      source.indexOf('const fetchInactivitySeconds = isRemote'),
    )
    expect(source).toContain('const armFetchInactivityTimeout = () =>')
    expect(source).toContain('if (value && value.byteLength > 0) armFetchInactivityTimeout()')
    expect(source).toContain('active.startedAt = Date.now()')
    expect(source).toContain('}, fetchInactivitySeconds * 1000)')
    expect(source).toContain('timeoutMs: fetchInactivitySeconds * 1000')
    expect(source).not.toContain('timeoutMs: timeoutSeconds * 1000')
  })

  it('does not turn the saved timeout into an explicit hard engine deadline', () => {
    expect(source).not.toMatch(/obj\.timeout\s*=\s*timeoutSeconds/)
  })
})

describe('detail keeps the machine code (ledger row 152)', () => {
  it('appends [code] when the message lacks it', () => {
    expect(
      chatStreamServerEventErrorDetail({
        error: {
          message: 'prefill admission rejected chunk [0:2048)',
          code: 'prefill_admission_declined',
        },
      }),
    ).toBe('prefill admission rejected chunk [0:2048) [prefill_admission_declined]')
  })

  it('does not duplicate a code already present', () => {
    expect(
      chatStreamServerEventErrorDetail({
        error: { message: 'x prefill_admission_declined y', code: 'prefill_admission_declined' },
      }),
    ).toBe('x prefill_admission_declined y')
  })
})

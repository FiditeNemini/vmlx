import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({
  ipcMain: { handle: vi.fn() },
}))
vi.mock('../src/main/ipc/utils', () => ({
  resolveBaseUrl: vi.fn(),
  getAuthHeaders: vi.fn(),
}))

import { isExpectedCacheEndpointDisconnectError } from '../src/main/ipc/cache'

describe('cache endpoint transport error classification', () => {
  it.each([
    'ECONNREFUSED',
    'ETIMEDOUT',
    'EHOSTUNREACH',
    'ENETUNREACH',
    'UND_ERR_SOCKET',
    'UND_ERR_CONNECT_TIMEOUT',
    'UND_ERR_HEADERS_TIMEOUT',
    'UND_ERR_BODY_TIMEOUT',
    'UND_ERR_DESTROYED',
    'UND_ERR_ABORTED',
  ])('classifies %s as an expected endpoint transport failure', (code) => {
    expect(isExpectedCacheEndpointDisconnectError({ code })).toBe(true)
  })

  it('finds an undici code nested under TypeError("fetch failed")', () => {
    const error: any = new TypeError('fetch failed')
    error.cause = { code: 'UND_ERR_SOCKET', message: 'other side closed' }

    expect(isExpectedCacheEndpointDisconnectError(error)).toBe(true)
  })

  it('finds ECONNREFUSED through aggregate and wrapped error shapes', () => {
    const error = {
      reason: {
        errors: [
          { code: 'SOMETHING_ELSE' },
          { cause: { code: 'ECONNREFUSED' } },
        ],
      },
    }

    expect(isExpectedCacheEndpointDisconnectError(error)).toBe(true)
  })

  it('does not classify application or HTTP errors as transport disconnects', () => {
    expect(isExpectedCacheEndpointDisconnectError({ code: 'ERR_INVALID_ARG_TYPE' })).toBe(false)
    expect(isExpectedCacheEndpointDisconnectError(new Error('Cache stats failed: 500'))).toBe(false)
  })
})

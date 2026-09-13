import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { remoteServerBaseUrl } from '../src/shared/remoteApiUrl'

describe('remote endpoint API base URLs', () => {
  it.each([
    ['http://127.0.0.1:8009', 'http://127.0.0.1:8009'],
    [' http://127.0.0.1:8009/v1/ ', 'http://127.0.0.1:8009'],
    ['https://gateway.test/api/v1', 'https://gateway.test/api'],
    ['https://gateway.test/tenant/model/v1///', 'https://gateway.test/tenant/model'],
    ['https://gateway.test/tenant-v1', 'https://gateway.test/tenant-v1'],
  ])('uses exactly one API version for %s', (input, expected) => {
    expect(remoteServerBaseUrl(input)).toBe(expected)
    expect(`${remoteServerBaseUrl(input)}/v1/responses`).toBe(`${expected}/v1/responses`)
  })

  it('uses the shared normalization in all endpoint-owning surfaces', () => {
    for (const file of [
      'src/main/sessions.ts',
      'src/main/ipc/chat.ts',
      'src/shared/remoteModelCapabilities.ts',
      'src/renderer/src/components/chat/ChatSettings.tsx',
    ]) {
      const source = readFileSync(file, 'utf8')
      expect(source, file).toContain('remoteServerBaseUrl(')
      expect(source, file).not.toMatch(/remoteUrl!?\.replace\(/)
    }
  })
})

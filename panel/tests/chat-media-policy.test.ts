import { describe, expect, it } from 'vitest'
import { resolveChatMediaPolicy } from '../src/shared/chatMediaPolicy'

describe('chat request media policy', () => {
  it('keeps Force Off ahead of detected vision when selecting tools', () => {
    expect(resolveChatMediaPolicy({ configuredMode: false, detectedMultimodal: true }).multimodal).toBe(false)
  })

  it('rejects media attachments under Force Off instead of changing mode or dropping media', () => {
    const result = resolveChatMediaPolicy({ configuredMode: false, detectedMultimodal: true, hasMediaAttachments: true })
    expect(result.multimodal).toBe(false)
    expect(result.attachmentError).toContain('Force Off')
  })

  it.each([{ forceTextOnly: true }, { smelt: true }])('keeps a text-only route authoritative: %j', (mode) => {
    const result = resolveChatMediaPolicy({ ...mode, configuredMode: true, detectedMultimodal: true, hasMediaAttachments: true })
    expect(result.multimodal).toBe(false)
    expect(result.attachmentError).toContain('text-only')
  })

  it.each([undefined, true])('forwards explicit media in enabled/unknown mode %s for server validation', (configuredMode) => {
    expect(resolveChatMediaPolicy({ configuredMode, hasMediaAttachments: true })).toEqual({ multimodal: true })
  })

  it('does not reject text-file attachments in a text-only session', () => {
    expect(resolveChatMediaPolicy({ configuredMode: false, detectedMultimodal: true, hasMediaAttachments: false })).toEqual({ multimodal: false })
  })

  it('does not turn an unknown artifact into a multimodal model without media', () => {
    expect(resolveChatMediaPolicy({})).toEqual({ multimodal: false })
  })
})

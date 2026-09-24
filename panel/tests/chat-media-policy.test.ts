import { describe, expect, it } from 'vitest'
import { hasPersistedUserMedia, resolveChatMediaPolicy } from '../src/shared/chatMediaPolicy'

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

  it('rejects a text follow-up with media history under Force Off instead of silently stripping history', () => {
    const input = { configuredMode: false, hasMediaHistory: true }
    expect(resolveChatMediaPolicy(input).attachmentError).toContain('conversation already contains')
  })
})

describe('persisted media admission', () => {
  it.each(['image_url', 'video_url', 'input_audio'])('finds prior %s when the new message is text-only', (type) => {
    const history = [
      { role: 'user', content: JSON.stringify([{ type: 'text', text: 'Describe this' }, { type, [type]: {} }]) },
      { role: 'assistant', content: 'A previous answer' },
      { role: 'user', content: 'Explain that again' },
    ]
    expect(hasPersistedUserMedia(history)).toBe(true)
    expect(resolveChatMediaPolicy({ configuredMode: false, hasMediaHistory: hasPersistedUserMedia(history) }).attachmentError).toContain('start a new text-only chat')
  })

  it('preserves media history in Auto and On for runtime validation', () => {
    for (const configuredMode of [undefined, true]) {
      expect(resolveChatMediaPolicy({ configuredMode, detectedMultimodal: true, hasMediaHistory: true }).attachmentError).toBeUndefined()
    }
  })

  it.each([
    '[not JSON mentioning image_url',
    'Explain input_audio and video_url',
    JSON.stringify([{ type: 'text', text: '[Attached file: example.json]\n{"type":"image_url"}' }]),
    JSON.stringify([null, 12, { type: 'input_text', text: 'image_url' }]),
  ])('does not misclassify text or malformed history: %s', (content) => {
    expect(hasPersistedUserMedia([{ role: 'user', content }])).toBe(false)
  })

  it('does not interpret assistant prose as user attachment history', () => {
    expect(hasPersistedUserMedia([{ role: 'assistant', content: '[{"type":"image_url"}]' }])).toBe(false)
  })
})

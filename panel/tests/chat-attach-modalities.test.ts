import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { attachmentAllowed } from '../src/renderer/src/components/chat/InputBox'

// Ledger H: the app let a user attach audio to a model whose runtime rejects audio (Gemma-4 26B A4B: text/vision/
// video); the turn then failed with a 400. A media file is refused at attach time only when the running engine
// ADVERTISES its modalities and that kind is not among them. Unknown or failed detection never restricts.
describe('attachmentAllowed', () => {
  it('refuses a media kind the engine does not advertise', () => {
    expect(attachmentAllowed('audio', ['text', 'image', 'video'])).toBe(false)
    expect(attachmentAllowed('video', ['text'])).toBe(false)
    expect(attachmentAllowed('image', ['TEXT', 'IMAGE'])).toBe(true)
    // The server's real vocabulary: images are advertised as "vision".
    expect(attachmentAllowed('image', ['text', 'vision', 'video'])).toBe(true)
    expect(attachmentAllowed('video', ['text', 'vision', 'video'])).toBe(true)
    expect(attachmentAllowed('audio', ['text', 'vision', 'video'])).toBe(false)
    expect(attachmentAllowed('image', ['text', 'video'])).toBe(false)
    expect(attachmentAllowed('audio', ['text', 'audio'])).toBe(true)
  })
  it('never restricts on unknown modalities or text files', () => {
    expect(attachmentAllowed('audio', null)).toBe(true)
    expect(attachmentAllowed('audio', [])).toBe(true)
    expect(attachmentAllowed('text', ['image'])).toBe(true)
  })
  it('has the notice text in every shipped locale', () => {
    for (const loc of ['en', 'ko', 'zh', 'ja', 'es']) {
      const d = JSON.parse(readFileSync(join(__dirname, '../src/renderer/src/i18n/locales', `${loc}.json`), 'utf8'))
      const msg: string = d.chat.input.unsupportedAttachment
      expect(msg).toContain('{{kind}}')
      expect(msg).toContain('{{name}}')
    }
  })
})

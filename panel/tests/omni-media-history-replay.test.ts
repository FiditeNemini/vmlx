import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { replayPersistedUserContentParts } from '../src/shared/mediaHistoryReplay'

const chatSource = readFileSync(resolve(__dirname, '../src/main/ipc/chat.ts'), 'utf8')

describe('multimodal historical media replay', () => {
  const parts = [
    { type: 'text', text: 'inspect these in order' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,IMAGE' } },
    { type: 'video_url', video_url: { url: 'data:video/mp4;base64,VIDEO' } },
    { type: 'input_audio', input_audio: { format: 'wav', data: 'AUDIO' } },
  ]

  it('keeps prior image, video, and audio bytes for every multimodal route', () => {
    expect(replayPersistedUserContentParts(parts, true)).toBe(parts)
    expect(chatSource).toContain('replayPersistedUserContentParts(')
    expect(chatSource).toContain('chatIsMultimodal || isRemote')
    expect(chatSource).not.toContain('stripHistoricalMediaPartsForReplay')
  })

  it('strips media only for a genuinely text-only route', () => {
    expect(replayPersistedUserContentParts(parts, false)).toBe('inspect these in order')
    expect(replayPersistedUserContentParts(parts.slice(1), false)).toBe('[Image omitted]')
  })
})

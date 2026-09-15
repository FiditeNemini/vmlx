import type { MediaAttachment } from './InputBox'
import { parseContentArray } from './chat-utils'

/** Reconstruct the input to the normal send path from a persisted user turn.
 * Both Edit and Regenerate must retain the media, not just the bubble's text.
 * File attachments are already persisted as text parts: expose ALL of them
 * in the editor rather than silently discarding everything after the first.
 * Family-specific content ordering remains owned by the main send path.
 */
export function restoreUserMessageContent(stored: string): {
  content: string
  attachments?: MediaAttachment[]
} {
  const parts = parseContentArray(stored)
  if (!parts) return { content: stored }

  const content = parts
    .filter(p => p.type === 'text' && typeof p.text === 'string')
    .map(p => p.text)
    .join('\n\n')
  const attachments: MediaAttachment[] = []
  for (const part of parts) {
    let kind: MediaAttachment['kind']
    let url: string | undefined
    if (part.type === 'input_audio' && part.input_audio?.data) {
      kind = 'audio'
      const { data, format = 'wav' } = part.input_audio
      const mime = format === 'mp3' ? 'audio/mpeg' : `audio/${format}`
      url = data.startsWith('data:') ? data : `data:${mime};base64,${data}`
    } else if (part.type === 'image_url') {
      kind = 'image'
      url = part.image_url?.url
    } else if (part.type === 'video_url') {
      kind = 'video'
      url = part.video_url?.url
    } else {
      continue
    }
    if (typeof url !== 'string' || !url) continue
    // Composer-only metadata is not stored in historical content arrays.
    // Do not change the payload/URL or infer a new encoding from its name.
    attachments.push({
      id: `replay-${attachments.length}`,
      kind,
      dataUrl: url,
      name: kind,
      type: url.match(/^data:([^;]+);/)?.[1]
        ?? (kind === 'image' ? 'image/png' : kind === 'video' ? 'video/mp4' : 'audio/wav'),
      size: 0,
    })
  }
  return { content, attachments: attachments.length ? attachments : undefined }
}

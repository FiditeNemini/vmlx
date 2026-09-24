export interface ChatMediaPolicyInput {
  configuredMode?: boolean
  detectedMultimodal?: boolean
  forceTextOnly?: boolean
  smelt?: boolean
  hasMediaAttachments?: boolean
  hasMediaHistory?: boolean
}

/** Inspect persisted content parts, never infer media from words in chat text. */
export function hasPersistedUserMedia(messages: ReadonlyArray<{ role: string; content: unknown }>): boolean {
  return messages.some((message) => {
    if (message.role !== 'user') return false
    let parts = message.content
    if (typeof parts === 'string') {
      if (!parts.trimStart().startsWith('[')) return false
      try { parts = JSON.parse(parts) } catch { return false }
    }
    return Array.isArray(parts) && parts.some((part) =>
      part && typeof part === 'object'
      && ['image_url', 'video_url', 'input_audio', 'input_image', 'input_video', 'image', 'video', 'audio'].includes(part.type),
    )
  })
}

/** Resolve request routing, not artifact capability; the server validates media. */
export function resolveChatMediaPolicy(input: ChatMediaPolicyInput): {
  multimodal: boolean
  attachmentError?: string
} {
  if (input.configuredMode === false && input.hasMediaHistory) {
    return {
      multimodal: false,
      attachmentError: 'This conversation already contains image, video, or audio content, but Multimodal Support is set to Force Off. Enable multimodal support in Server Settings and restart the session, or start a new text-only chat.',
    }
  }
  const disabled = input.configuredMode === false || input.smelt || input.forceTextOnly
  if (disabled) {
    if (input.hasMediaAttachments) {
      return {
        multimodal: false,
        attachmentError: input.configuredMode === false
          ? 'Multimodal Support is set to Force Off. Remove image, video, or audio attachments, or enable multimodal support in Server Settings and restart the session.'
          : 'This session uses a text-only runtime. Remove image, video, or audio attachments, or select a model session with media support.',
      }
    }
    return { multimodal: false }
  }
  return {
    multimodal: input.detectedMultimodal === true
      || input.configuredMode === true
      || input.hasMediaAttachments === true,
  }
}

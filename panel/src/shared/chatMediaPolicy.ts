export interface ChatMediaPolicyInput {
  configuredMode?: boolean
  detectedMultimodal?: boolean
  forceTextOnly?: boolean
  smelt?: boolean
  hasMediaAttachments?: boolean
}

/** Resolve request routing, not artifact capability; the server validates media. */
export function resolveChatMediaPolicy(input: ChatMediaPolicyInput): {
  multimodal: boolean
  attachmentError?: string
} {
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

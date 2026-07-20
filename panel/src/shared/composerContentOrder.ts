export type ComposerContentPart = {
  type: string
  text?: string
  [key: string]: unknown
}

function isGemma4Family(family?: string): boolean {
  const normalized = String(family || '').toLowerCase().replace(/_/g, '-')
  return normalized === 'gemma4' || normalized === 'gemma-4'
}

/**
 * Build a local Electron composer turn without changing caller-provided API
 * history. Gemma 4's bundle contract requires visual media before text and
 * audio after text; other families retain the panel's historical text-first
 * order.
 */
export function orderComposerContentParts(
  text: string,
  attachmentParts: ComposerContentPart[],
  detectedFamily?: string,
): ComposerContentPart[] {
  const textPart = text.trim() ? [{ type: 'text', text }] : []
  if (!isGemma4Family(detectedFamily)) return [...textPart, ...attachmentParts]

  const visualParts = attachmentParts.filter(
    (part) => part.type === 'image_url' || part.type === 'video_url',
  )
  const textAttachmentParts = attachmentParts.filter((part) => part.type === 'text')
  const audioParts = attachmentParts.filter((part) => part.type === 'input_audio')
  const otherParts = attachmentParts.filter(
    (part) =>
      part.type !== 'image_url' &&
      part.type !== 'video_url' &&
      part.type !== 'text' &&
      part.type !== 'input_audio',
  )

  return [
    ...visualParts,
    ...otherParts,
    ...textPart,
    ...textAttachmentParts,
    ...audioParts,
  ]
}

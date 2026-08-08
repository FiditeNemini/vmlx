import { appendMetalWiredLimitGuidance } from './metalWiredLimit'

const PROJECTED_METAL_HEADROOM_RE =
  /Requested max output tokens exceed projected safe Metal headroom/i

export function projectedMetalHeadroomChatErrorContent(
  message: string | undefined | null,
): string | null {
  const raw = String(message || "").trim()
  if (!PROJECTED_METAL_HEADROOM_RE.test(raw)) return null

  const detail = raw
    .replace(/^Failed to send message:\s*/i, "")
    .replace(/^API error:\s*413\s*-\s*/i, "")
    .trim()

  return appendMetalWiredLimitGuidance(`Generation blocked: ${detail}`)
}

const PROMPT_TOO_LONG_RE = /prompt[_\s]too[_\s]long|prompt too long/i

export function promptTooLongChatErrorContent(
  message: string | undefined | null,
): string | null {
  const raw = String(message || "").trim()
  if (!PROMPT_TOO_LONG_RE.test(raw)) return null

  let detail = raw
    .replace(/^Failed to send message:\s*/i, "")
    .replace(/^API error:\s*4\d\d\s*-\s*/i, "")
    .trim()

  // Older harvest paths pass the raw engine JSON body through — unwrap the
  // OpenAI-style error envelope so the bubble shows the human sentence.
  try {
    const parsed = JSON.parse(detail)
    const nested = parsed?.error?.message ?? parsed?.detail
    if (typeof nested === "string" && nested.trim()) detail = nested.trim()
  } catch {
    /* already plain text */
  }

  return (
    `Message not sent — ${detail}\n\n` +
    "In the app: shorten this message, remove large attachments, raise the " +
    "session's max context in Settings, or start a new chat."
  )
}

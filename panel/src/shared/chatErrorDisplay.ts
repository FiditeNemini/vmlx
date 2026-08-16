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

  return appendMetalWiredLimitGuidance(`${METAL_HEADROOM_BUBBLE_PREFIX}${detail}`)
}

const PROMPT_TOO_LONG_RE = /prompt[_\s]too[_\s]long|prompt too long/i

export function promptTooLongChatErrorContent(
  message: string | undefined | null,
): string | null {
  const raw = String(message || "").trim()
  // A valve decline must qualify on its OWN patterns: the engine's message
  // ("prefill admission rejected chunk ... exceeds the device working-set
  // limit") contains neither "prompt too long" nor the machine code, so
  // gating the admission branch behind PROMPT_TOO_LONG_RE left the decline
  // with no persistent chat surface (only a vanishing toast) — reproduced
  // twice in the dots3 live CDP proof.
  if (!PROMPT_TOO_LONG_RE.test(raw) && !PREFILL_ADMISSION_DECLINED_RE.test(raw)) return null

  let detail = raw
    .replace(/^Failed to send message:\s*/i, "")
    .replace(/^Server error:\s*/i, "")
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

  // A turn-peak admission decline shares the 413/prompt_too_long envelope
  // but has the OPPOSITE remedy: the conversation is too DEEP for this
  // process (Metal peak walks up across turns), not too large per message —
  // "shorten this message" cannot fix it, while restarting the model can
  // (the conversation's cache restores from disk and the fresh process
  // serves the same turn; proven live at 101k tokens).
  if (PREFILL_ADMISSION_DECLINED_RE.test(raw) || PREFILL_ADMISSION_DECLINED_RE.test(detail)) {
    return (
      `${PROMPT_TOO_LONG_BUBBLE_PREFIX}${detail}\n\n` +
      "In the app: restart this model from the Server tab and resend — the " +
      "conversation restores from the disk cache — or start a new chat."
    )
  }

  return (
    `${PROMPT_TOO_LONG_BUBBLE_PREFIX}${detail}\n\n` +
    "In the app: shorten this message, remove large attachments, raise the " +
    "session's max context in Settings, or start a new chat."
  )
}

const PREFILL_ADMISSION_DECLINED_RE =
  /prefill_admission_declined|turn-peak admission declined|prefill admission rejected/i

export const PROMPT_TOO_LONG_BUBBLE_PREFIX = "Message not sent — "
export const METAL_HEADROOM_BUBBLE_PREFIX = "Generation blocked: "

// Panel-synthesized error bubbles are UI-only annotations. They must never be
// replayed to the model as assistant history — and for prompt-too-long, the
// user message that produced the bubble is itself the oversized payload, so
// replaying it poisons every later turn in the chat with the same 413.
export function isPromptTooLongBubbleContent(content: unknown): boolean {
  return (
    typeof content === "string" &&
    content.startsWith(PROMPT_TOO_LONG_BUBBLE_PREFIX)
  )
}

export function isMetalHeadroomBubbleContent(content: unknown): boolean {
  return (
    typeof content === "string" &&
    content.startsWith(METAL_HEADROOM_BUBBLE_PREFIX)
  )
}

export class ChatStreamServerEventError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ChatStreamServerEventError'
  }
}

export function chatStreamServerEventErrorDetail(
  payload: any,
  eventType?: string,
): string | null {
  const isResponsesError =
    eventType === 'error' ||
    eventType === 'response.error' ||
    eventType === 'response.failed'
  if (!isResponsesError && !payload?.error) return null

  const error = payload?.response?.error || payload?.error
  const message = String(
    error?.message ||
      error?.code ||
      payload?.detail ||
      JSON.stringify(payload),
  )
  // Keep the machine code visible alongside the prose: the persistent
  // in-chat bubble classifiers match on codes ("prefill_admission_declined",
  // "prompt_too_long"), and a prose-only detail rendered a device-capacity
  // valve decline as a vanishing toast with NO persistent surface at all
  // (live dots3 CDP proof, ledger row 150 finding 1).
  const code = error?.code ? String(error.code) : ''
  if (code && !message.includes(code)) {
    return `${message} [${code}]`
  }
  return message
}

export function shouldRethrowChatStreamLineError(
  error: unknown,
  expectedBackendDisconnect: boolean,
): boolean {
  return error instanceof ChatStreamServerEventError || expectedBackendDisconnect
}

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
  return String(
    error?.message ||
      error?.code ||
      payload?.detail ||
      JSON.stringify(payload),
  )
}

export function shouldRethrowChatStreamLineError(
  error: unknown,
  expectedBackendDisconnect: boolean,
): boolean {
  return error instanceof ChatStreamServerEventError || expectedBackendDisconnect
}

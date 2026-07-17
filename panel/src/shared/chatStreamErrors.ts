export class ChatStreamServerEventError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ChatStreamServerEventError'
  }
}

export function shouldRethrowChatStreamLineError(
  error: unknown,
  expectedBackendDisconnect: boolean,
): boolean {
  return error instanceof ChatStreamServerEventError || expectedBackendDisconnect
}

export interface StreamedChatToolCall {
  id: string
  function: { name: string; arguments: string }
}

/** OpenAI tool deltas may deliver id, name and arguments in separate chunks. */
export function accumulateChatToolCallDelta(
  calls: StreamedChatToolCall[],
  delta: any,
  fallbackId: () => string,
): void {
  if (!delta || typeof delta !== 'object') return
  const id = typeof delta.id === 'string' && delta.id ? delta.id : undefined
  const name = typeof delta.function?.name === 'string' ? delta.function.name : ''
  const args = typeof delta.function?.arguments === 'string' ? delta.function.arguments : ''
  if (!id && !name && !args) return
  const index = Number.isInteger(delta.index) && delta.index >= 0 ? delta.index : calls.length
  const call = calls[index] ?? { id: id ?? fallbackId(), function: { name: '', arguments: '' } }
  // A provider id arriving after a function fragment supersedes the existing
  // legacy fallback. Never replace an already received id on a name-only chunk.
  if (id) call.id = id
  call.function.name += name
  call.function.arguments += args
  calls[index] = call
}

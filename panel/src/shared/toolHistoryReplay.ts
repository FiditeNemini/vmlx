export interface PersistedAssistantToolHistory {
  content?: unknown
  reasoningContent?: unknown
  reasoningSegmentsJson?: unknown
  toolCallsJson?: unknown
  toolCallsOaiJson?: unknown
  toolResultsOaiJson?: unknown
}

interface OaiToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface OaiToolResult {
  tool_call_id: string
  content: string
}

function parseArray(value: unknown): any[] {
  if (typeof value !== 'string' || value.length === 0) return []
  try {
    const parsed = JSON.parse(value)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function validCalls(value: unknown): OaiToolCall[] {
  return parseArray(value).filter(
    (call: any) =>
      call &&
      typeof call.id === 'string' &&
      call.id.length > 0 &&
      call.function &&
      typeof call.function.name === 'string' &&
      typeof call.function.arguments === 'string',
  )
}

function validResults(value: unknown): OaiToolResult[] {
  return parseArray(value)
    .filter(
      (result: any) =>
        result &&
        typeof result.tool_call_id === 'string' &&
        typeof result.content === 'string',
    )
    .map((result: any) => ({
      tool_call_id: result.tool_call_id,
      content: result.content,
    }))
}

function persistedReasoningSegments(message: PersistedAssistantToolHistory): string[] {
  const segments = parseArray(message.reasoningSegmentsJson).map((segment) =>
    typeof segment === 'string' ? segment : '',
  )
  if (segments.length > 0) return segments
  return typeof message.reasoningContent === 'string' && message.reasoningContent
    ? [message.reasoningContent]
    : []
}

function reasoningItem(text: string): any {
  return {
    type: 'reasoning',
    content: [{ type: 'reasoning', text }],
  }
}

/**
 * Rebuild one persisted assistant row into model-visible history.
 *
 * Tool-loop rows collapse several model generations into one SQLite row.  The
 * UI already stores the native calls/results, per-generation reasoning
 * segments, and call iteration numbers.  Re-expand those fields in their real
 * order so Responses and Chat Completions templates see:
 * reasoning -> call -> result -> reasoning -> answer.
 */
export function replayPersistedAssistantHistory(
  message: PersistedAssistantToolHistory,
  useResponsesApi: boolean,
): any[] {
  const calls = validCalls(message.toolCallsOaiJson)
  const results = validResults(message.toolResultsOaiJson)
  const content = typeof message.content === 'string' ? message.content : ''
  const segments = persistedReasoningSegments(message)

  const callIterations = new Map<string, number>()
  for (const status of parseArray(message.toolCallsJson)) {
    if (
      status?.phase === 'calling' &&
      typeof status.toolCallId === 'string' &&
      Number.isInteger(status.iteration)
    ) {
      callIterations.set(status.toolCallId, Math.max(0, status.iteration))
    }
  }

  if (calls.length === 0) {
    if (useResponsesApi) {
      const items: any[] = []
      if (segments[0]?.trim()) items.push(reasoningItem(segments[0]))
      if (content.trim()) items.push({ type: 'output_text', text: content })
      return items
    }
    if (!content.trim() && !segments[0]?.trim()) return []
    return [
      {
        role: 'assistant',
        content,
        ...(segments[0]?.trim()
          ? { reasoning_content: segments[0] }
          : {}),
      },
    ]
  }

  const groups = new Map<number, OaiToolCall[]>()
  calls.forEach((call, index) => {
    const iteration = callIterations.get(call.id) ?? index
    const group = groups.get(iteration) || []
    group.push(call)
    groups.set(iteration, group)
  })
  const iterations = [...groups.keys()].sort((a, b) => a - b)
  const resultByCall = new Map(results.map((result) => [result.tool_call_id, result]))
  const replay: any[] = []

  for (const iteration of iterations) {
    const group = groups.get(iteration) || []
    const reasoning = segments[iteration]
    if (useResponsesApi) {
      if (reasoning?.trim()) replay.push(reasoningItem(reasoning))
      for (const call of group) {
        replay.push({
          type: 'function_call',
          call_id: call.id,
          name: call.function.name,
          arguments: call.function.arguments,
        })
      }
      for (const call of group) {
        const result = resultByCall.get(call.id)
        if (result) {
          replay.push({
            type: 'function_call_output',
            call_id: result.tool_call_id,
            output: result.content,
          })
        }
      }
    } else {
      replay.push({
        role: 'assistant',
        content: null,
        tool_calls: group,
        ...(reasoning?.trim() ? { reasoning_content: reasoning } : {}),
      })
      for (const call of group) {
        const result = resultByCall.get(call.id)
        if (result) {
          replay.push({
            role: 'tool',
            tool_call_id: result.tool_call_id,
            content: result.content,
          })
        }
      }
    }
  }

  const lastIteration =
    iterations.length > 0 ? iterations[iterations.length - 1] : -1
  const finalReasoning = segments[lastIteration + 1]
  if (useResponsesApi) {
    if (finalReasoning?.trim()) replay.push(reasoningItem(finalReasoning))
    if (content.trim()) replay.push({ type: 'output_text', text: content })
  } else if (content.trim() || finalReasoning?.trim()) {
    replay.push({
      role: 'assistant',
      content,
      ...(finalReasoning?.trim()
        ? { reasoning_content: finalReasoning }
        : {}),
    })
  }
  return replay
}

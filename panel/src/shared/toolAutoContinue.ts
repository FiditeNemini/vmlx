export interface ToolAutoContinueInput {
  content: string
  iterationTokenCount: number
  finishReason?: string | null
  thresholdTokens: number
}

export function shouldAutoContinueAfterToolUse({
  content,
  iterationTokenCount,
  finishReason,
  thresholdTokens,
}: ToolAutoContinueInput): boolean {
  if (!content.trim()) return true
  return finishReason === 'length' && iterationTokenCount < thresholdTokens
}

export function shouldFinishZayaAppleScriptToolRound(
  isAppleScriptToolBundle: boolean,
  toolNames: string[],
): boolean {
  return (
    isAppleScriptToolBundle &&
    toolNames.length > 0 &&
    toolNames.every((name) => name === 'run_applescript')
  )
}

export function requestedExactFinalToolNames(text: string): string[] {
  // This optimization removes the tool catalog from the first follow-up, so
  // recognize only unambiguous singular call directives. A multi-tool contract
  // remains agentic until every explicitly named tool has completed.
  // A broad `exactly once` check misclassified multi-tool requests such as
  // "call file_info ... after that result call run_command ... after both
  // results reply exactly ..." and made the requested second call impossible.
  const names = Array.from(
    text.matchAll(
      /\bcall\s+(?:the\s+)?(?:built-in\s+)?([a-z][\w-]*)(?:\s+tool)?\s+exactly\s+once\b/gi,
    ),
    match => match[1].toLowerCase(),
  )
  if (names.length === 0 || new Set(names).size !== names.length) return []

  // Keep this bounded to the final-result clause, but allow ordinary
  // modifiers from the exact contract ("the real tool result", "both tool
  // results"). The previous literal-only match left tools enabled and a live
  // Bonsai turn executed the same file_info call five times.
  const exactFinalAfterResults =
    /\bafter\b[^.!?\n]{0,96}\b(?:the\s+(?:real\s+)?tool|its|that|both\s+tool)\s+results?\b[^.!?\n]{0,64}\breply exactly\b/i.test(
      text,
    )
  return exactFinalAfterResults ? names : []
}

export function requestsDirectAnswerAfterSingleTool(text: string): boolean {
  return requestedExactFinalToolNames(text).length === 1
}

export function requestsExactTextOnlyWithoutToolUse(text: string): boolean {
  if (!/\breply exactly\b/i.test(text)) return false
  if (requestedExactFinalToolNames(text).length > 0) return false

  // A previous chat/profile may leave builtin tools enabled. For strict
  // exact-answer probes that do not ask for tool use, sending the whole tool
  // catalog changes the prompt and lets small/native models answer from schema
  // text instead of the current user turn. Keep this directive-shaped so normal
  // agentic coding chats still receive tools.
  const explicitToolRequest =
    /\b(?:call|use|invoke|run)\s+(?:the\s+)?(?:built[- ]in\s+)?[a-z][\w-]*(?:\s+tool|\s+function)?\b/i
  const toolResultContract =
    /\bafter\b[^.!?\n]{0,120}\b(?:tool|function)\s+results?\b/i
  const mustUseTool =
    /\bmust\s+(?:call|use|invoke|run)\s+(?:the\s+)?(?:built[- ]in\s+)?(?:[a-z][\w-]*\s+)?(?:tool|function)\b/i

  return !(
    explicitToolRequest.test(text) ||
    toolResultContract.test(text) ||
    mustUseTool.test(text)
  )
}

export function requestsNoToolCalls(text: string): boolean {
  // Keep these directive-shaped so quoted discussion of tool policy does not
  // silently disable the catalog. The UI omits tool schemas entirely when
  // this returns true; that is the stable no-tool request contract for both
  // Responses and Chat Completions.
  const explicitProhibition =
    /(?:^|[.!?\]\n])\s*(?:please\s+)?(?:do not|don['’]?t|dont|never)\s+(?:call|use)\s+(?:any\s+)?tools?\b(?!\s+unless)/i
  const explicitWithoutTools =
    /(?:^|[.!?\]\n])\s*(?:please\s+)?without\s+(?:using\s+)?(?:any\s+)?tools?\b/i
  return explicitProhibition.test(text) || explicitWithoutTools.test(text)
}

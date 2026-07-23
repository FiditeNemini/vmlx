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

export function requestedOnceToolNames(text: string): string[] {
  // An explicit per-tool `exactly once` directive is an execution invariant,
  // independent of how the user phrases the final-answer request. Retire each
  // named tool after its first call while leaving other tools available for
  // genuinely agentic multi-tool work.
  const names = Array.from(
    text.matchAll(
      /\bcall\s+(?:the\s+)?(?:built-in\s+)?([a-z][\w-]*)(?:\s+tool)?\s+exactly\s+once\b/gi,
    ),
    match => match[1].toLowerCase(),
  )
  if (names.length === 0 || new Set(names).size !== names.length) return []
  return names
}

export function requestedExactFinalToolNames(text: string): string[] {
  // This optimization removes the whole tool catalog from the final-answer
  // follow-up, so keep it narrower than the per-tool exactly-once invariant.
  // A multi-tool contract remains agentic until every explicitly named tool
  // has completed.
  const names = requestedOnceToolNames(text)
  if (names.length === 0) return []

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

function containsExplicitToolRequest(text: string): boolean {
  const explicitToolRequest =
    /\b(?:call|use|invoke|run)\s+(?:the\s+)?(?:built[- ]in\s+)?[a-z][\w-]*(?:\s+tool|\s+function)?\b/i
  const toolResultContract =
    /\bafter\b[^.!?\n]{0,120}\b(?:tool|function)\s+results?\b/i
  const mustUseTool =
    /\bmust\s+(?:call|use|invoke|run)\s+(?:the\s+)?(?:built[- ]in\s+)?(?:[a-z][\w-]*\s+)?(?:tool|function)\b/i

  return (
    explicitToolRequest.test(text) ||
    toolResultContract.test(text) ||
    mustUseTool.test(text) ||
    requestedExactFinalToolNames(text).length > 0
  )
}

export function requestsExactTextOnlyWithoutToolUse(text: string): boolean {
  const strictTextAnswer =
    /\breply exactly\b/i.test(text) ||
    /\banswer exactly\b/i.test(text) ||
    /\bvisible answer\s+(?:must\s+be\s+|is\s+|exactly\s+)/i.test(text) ||
    /\bonly visible answer\s+(?:must\s+be\s+|is\s+|exactly\s+)/i.test(text) ||
    /\byour only visible answer\s+(?:must\s+be\s+|is\s+|exactly\s+)/i.test(text)
  if (!strictTextAnswer) return false
  if (requestedExactFinalToolNames(text).length > 0) return false

  // A previous chat/profile may leave builtin tools enabled. For strict
  // exact-answer probes that do not ask for tool use, sending the whole tool
  // catalog changes the prompt and lets small/native models answer from schema
  // text instead of the current user turn. Keep this directive-shaped so normal
  // agentic coding chats still receive tools.
  return !containsExplicitToolRequest(text)
}

export function requestsPrivateReasoningWithoutToolUse(text: string): boolean {
  if (containsExplicitToolRequest(text)) return false

  const privateReasoningDirective =
    /\b(?:privately|private(?:ly)?\s+reason|private\s+(?:calculation|check|solution)|in\s+private|do\s+not\s+expose\s+(?:the\s+)?reasoning|without\s+showing\s+(?:the\s+)?reasoning|silent(?:ly)?\s+(?:reason|think)|think\s+privately)\b/i
  if (!privateReasoningDirective.test(text)) return false

  const textOnlyReasoningTask =
    /\b(?:calculate|calculation|compute|solve|solution|double[- ]?check|check\s+(?:whether|if)|verify|reason\s+through|derive|evaluate)\b/i
  return textOnlyReasoningTask.test(text)
}

export function requestsNoToolCalls(text: string): boolean {
  // Keep these directive-shaped so quoted discussion of tool policy does not
  // silently disable the catalog. The UI omits tool schemas entirely when
  // this returns true; that is the stable no-tool request contract for both
  // Responses and Chat Completions.
  const explicitProhibition =
    /(?:^|[.!?\]\n])\s*(?:please\s+)?(?:do not|don['’]?t|dont|never)\s+(?:call|use)\s+(?:any\s+)?tools?\b(?!\s+unless)/i
  const explicitWithoutTools =
    /(?:^|[.!?\]\n])\s*(?:please\s+)?without\s+(?:(?:using|calling)\s+)?(?:any\s+)?tools?\b/i
  return explicitProhibition.test(text) || explicitWithoutTools.test(text)
}

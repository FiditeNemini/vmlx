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

export function requestsDirectAnswerAfterSingleTool(text: string): boolean {
  return (
    /\bexactly once\b/i.test(text) &&
    // Keep this bounded to one clause, but allow ordinary modifiers from the
    // user's exact contract (for example "after the real tool result"). The
    // previous literal-only match left tools enabled on that follow-up and a
    // live Bonsai turn executed the same file_info call five times.
    /\bafter\b[^.!?\n]{0,64}\b(?:the\s+(?:real\s+)?tool|its|that)\s+result\b/i.test(text) &&
    /\breply exactly\b/i.test(text)
  )
}

export function requestsNoToolCalls(text: string): boolean {
  // Map an explicit current-turn user directive onto the standard API
  // `tool_choice: "none"` contract. Keep this directive-shaped so quoted
  // discussion of tool policy does not silently disable the catalog.
  return /(?:^|[.!?\]\n])\s*(?:please\s+)?(?:do not|don['’]?t|dont|never)\s+(?:call|use)\s+(?:any\s+)?tools?\b(?!\s+unless)/i.test(
    text,
  )
}

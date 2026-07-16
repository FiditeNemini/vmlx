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
    /\bafter\b[^.!?\n]{0,64}\btool result\b/i.test(text) &&
    /\breply exactly\b/i.test(text)
  )
}

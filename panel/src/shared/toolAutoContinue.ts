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
    /\bafter (?:the )?tool result\b/i.test(text) &&
    /\breply exactly\b/i.test(text)
  )
}

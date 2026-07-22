export interface ToolLaunchArgsInput {
  toolParser?: string | null
  enableAutoToolChoice?: boolean | null
}

/**
 * Build the tool-related engine arguments from the already-resolved session
 * policy. Keeping this tiny contract shared prevents the launcher, command
 * preview, and tests from drifting on Auto/On/Off semantics.
 */
export function buildToolLaunchArgs({
  toolParser,
  enableAutoToolChoice,
}: ToolLaunchArgsInput): string[] {
  const args: string[] = []

  if (toolParser === 'none') {
    args.push('--tool-call-parser', 'none')
    return args
  }

  if (toolParser) {
    args.push('--tool-call-parser', toolParser)
  }
  if (enableAutoToolChoice === true) {
    args.push('--enable-auto-tool-choice')
  }

  return args
}

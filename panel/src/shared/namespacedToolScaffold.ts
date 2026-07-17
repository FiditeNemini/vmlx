type ParsedToolCall = {
  function?: {
    name?: string
  }
}

// Some Qwen3.6 checkpoints emit a human-readable preview immediately before
// their real parser-native XML call:
//
//   [request-tag]
//   call:default_api:file_info{path: "panel/package.json"}
//   >
//
// The XML call is the executable source of truth. Replaying this preview as a
// Responses output_text item after the tool result teaches the next generation
// that it should repeat the call and can terminate the Electron agent loop.
// Match only a complete line-oriented preview; removal is additionally gated
// below on a parsed function call with the same tool name.
const NAMESPACED_TOOL_PREVIEW =
  /(?:^\[[^\]\r\n]{1,160}\][ \t]*\r?\n)?^call:[A-Za-z_][A-Za-z0-9_.-]*:([A-Za-z_][A-Za-z0-9_]*)\{[^\r\n]*\}[ \t]*(?:\r?\n^>[ \t]*)?(?:\r?\n|$)/gim

export function stripRedundantNamespacedToolPreview(
  content: string,
  parsedToolCalls: ParsedToolCall[],
): string {
  if (!content || parsedToolCalls.length === 0 || !content.includes('call:')) {
    return content
  }

  const parsedNames = new Set(
    parsedToolCalls
      .map((call) => call.function?.name?.trim())
      .filter((name): name is string => Boolean(name)),
  )
  if (parsedNames.size === 0) return content

  return content
    .replace(NAMESPACED_TOOL_PREVIEW, (preview, toolName: string) =>
      parsedNames.has(toolName) ? '' : preview,
    )
    .trim()
}

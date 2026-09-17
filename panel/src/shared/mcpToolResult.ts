export interface McpToolResultPayload {
  content?: unknown;
  is_error?: boolean;
  error_message?: string | null;
}

/** Bound tool-owned text identically for the display and model continuation. */
export function boundToolResultText(text: string, maxChars: number): string {
  if (text.length > maxChars) {
    return text.slice(0, maxChars) +
      `\n\n[Truncated — showing first ${maxChars} of ${text.length} characters]`;
  }
  return text;
}

/** Preserve tool execution feedback in both Chat and Responses continuations. */
export function formatMcpToolResult(
  result: McpToolResultPayload,
  maxChars: number,
): string {
  const content = typeof result.content === "string"
    ? result.content
    : JSON.stringify(result.content, null, 2) ?? "";
  let text = content;
  if (result.is_error) {
    // Older engines can return is_error + content without error_message.
    // Test emptiness without trimming the actual tool-owned string.
    const explicit = result.error_message;
    const detail = typeof explicit === "string" && explicit.trim()
      ? explicit
      : result.content != null && content.trim() ? content : "Unknown error";
    text = `Error: ${detail}`;
  }
  // Apply the existing successful-tool bound to errors too. A rejected call
  // must not bypass the session's tool-result context limit.
  return boundToolResultText(text, maxChars);
}

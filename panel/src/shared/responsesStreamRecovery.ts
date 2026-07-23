export interface ResponsesToolBufferReconciliation {
  clearSpeculativeBuffering: boolean;
  authoritativeText: string | null;
  rejectedControlMarkup?: boolean;
}

/**
 * Line-start tool-call markers that activate client-side speculative
 * buffering. Includes real native dialects and common hallucinated tags.
 *
 * Contract: line-start tool control markup is never visible assistant prose.
 * A Responses terminal can carry parser-rejected or incomplete tool markup in
 * output_text even though no concrete function_call item was emitted. That
 * control text must be rejected, not fenced and shown to the user. Ordinary
 * authoritative prose is still restored after a false-positive heartbeat.
 */
export const TOOL_CALL_MARKER_LINE_START =
  /(?:^|\n)\s*(?:<zyphra_tool_call\b|<function(?:=|\b)|<minimax:tool_call|<tool_call\b|\[Calling tool:|<invoke name=|<read_file\b|<write_file\b|<run_command\b|<search_files\b|<edit_file\b|<list_directory\b|<execute_command\b|<bash\b)/;

/**
 * Reconcile a Responses stream that advertised speculative tool generation but
 * completed without a concrete function_call item.
 *
 * The server's output_text.done value is authoritative after native tool
 * parsing. A heartbeat or partial XML prefix may have caused the Electron
 * client to suppress earlier text deltas, so the final text must replace the
 * current-iteration buffer instead of being skipped merely because a delta was
 * observed on the wire.
 */
export function reconcileResponsesToolBufferAtStreamEnd(args: {
  useResponsesApi: boolean;
  clientToolCallBuffering: boolean;
  receivedToolCallCount: number;
  finalText: string;
}): ResponsesToolBufferReconciliation {
  if (
    !args.useResponsesApi ||
    !args.clientToolCallBuffering ||
    args.receivedToolCallCount > 0
  ) {
    return {
      clearSpeculativeBuffering: false,
      authoritativeText: null,
    };
  }

  if (
    args.finalText.length > 0 &&
    TOOL_CALL_MARKER_LINE_START.test(args.finalText)
  ) {
    return {
      clearSpeculativeBuffering: true,
      authoritativeText: null,
      rejectedControlMarkup: true,
    };
  }

  return {
    clearSpeculativeBuffering: true,
    authoritativeText: args.finalText.length > 0 ? args.finalText : null,
  };
}

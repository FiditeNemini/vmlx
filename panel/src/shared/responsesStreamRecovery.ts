export interface ResponsesToolBufferReconciliation {
  clearSpeculativeBuffering: boolean;
  authoritativeText: string | null;
}

/**
 * Line-start tool-call markers that activate client-side speculative
 * buffering. Includes real native dialects and common hallucinated tags.
 *
 * Contract: text restored by reconcileResponsesToolBufferAtStreamEnd is
 * authoritative zero-tool output and MUST NOT be re-scanned with this
 * pattern — the restored text often still contains the very marker that
 * activated buffering, and re-scanning re-suppresses it forever (the
 * stream is already over, so no second reconciliation can rescue it).
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

  return {
    clearSpeculativeBuffering: true,
    authoritativeText: args.finalText.length > 0 ? args.finalText : null,
  };
}

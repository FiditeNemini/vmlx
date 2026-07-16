export interface ResponsesToolBufferReconciliation {
  clearSpeculativeBuffering: boolean;
  authoritativeText: string | null;
}

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

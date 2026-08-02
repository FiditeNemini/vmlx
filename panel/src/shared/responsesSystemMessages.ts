export interface ResponsesMessageLike {
  role?: string;
}

/**
 * Split Responses messages into top-level instructions and ordered input.
 *
 * Generic chat templates require every system message to be folded into the
 * leading instructions slot. DeepSeek V4's native encoder is the exception:
 * later system messages are positional conversation input, so moving them to
 * the front changes the prompt prefix. For DSV4, extract only the contiguous
 * leading system run and preserve every later message in its original order.
 */
export function splitResponsesSystemMessages<T extends ResponsesMessageLike>(
  messages: readonly T[],
  preserveNativeOrder: boolean,
): { systemMessages: T[]; inputMessages: T[] } {
  if (!preserveNativeOrder) {
    return {
      systemMessages: messages.filter((message) => message.role === "system"),
      inputMessages: messages.filter((message) => message.role !== "system"),
    };
  }

  let firstNonSystem = 0;
  while (
    firstNonSystem < messages.length &&
    messages[firstNonSystem]?.role === "system"
  ) {
    firstNonSystem += 1;
  }

  return {
    systemMessages: messages.slice(0, firstNonSystem),
    inputMessages: messages.slice(firstNonSystem),
  };
}

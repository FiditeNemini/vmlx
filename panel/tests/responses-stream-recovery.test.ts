import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { reconcileResponsesToolBufferAtStreamEnd } from "../src/shared/responsesStreamRecovery";

describe("Responses speculative tool-buffer reconciliation", () => {
  it("restores authoritative final text when a heartbeat produced no function call", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 0,
        finalText: "B1-REASON-SOAK14-PASS",
      }),
    ).toEqual({
      clearSpeculativeBuffering: true,
      authoritativeText: "B1-REASON-SOAK14-PASS",
    });
  });

  it("clears a zero-tool speculative buffer even when the server final is empty", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 0,
        finalText: "",
      }),
    ).toEqual({ clearSpeculativeBuffering: true, authoritativeText: null });
  });

  it("does not expose buffered markup when a concrete function call arrived", () => {
    expect(
      reconcileResponsesToolBufferAtStreamEnd({
        useResponsesApi: true,
        clientToolCallBuffering: true,
        receivedToolCallCount: 1,
        finalText: "pre-tool prose",
      }),
    ).toEqual({ clearSpeculativeBuffering: false, authoritativeText: null });
  });

  it("does nothing for non-Responses or non-buffered streams", () => {
    for (const [useResponsesApi, clientToolCallBuffering] of [
      [false, true],
      [true, false],
    ] as const) {
      expect(
        reconcileResponsesToolBufferAtStreamEnd({
          useResponsesApi,
          clientToolCallBuffering,
          receivedToolCallCount: 0,
          finalText: "visible",
        }),
      ).toEqual({ clearSpeculativeBuffering: false, authoritativeText: null });
    }
  });

  it("is wired after both initial and follow-up SSE streams", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    expect(source).toContain("responsesFinalText = parsed.text");
    expect(source).toContain("const reconcileResponsesToolBuffer = () =>");
    expect(source.match(/reconcileResponsesToolBuffer\(\);/g)).toHaveLength(2);
    expect(source).toContain("_sawResponsesTextDelta = false;");
    expect(source).toContain('responsesFinalText = "";');
  });
});

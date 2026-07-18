import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  reconcileResponsesToolBufferAtStreamEnd,
  TOOL_CALL_MARKER_LINE_START,
} from "../src/shared/responsesStreamRecovery";

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

  it("marker pattern catches hallucinated line-start tool dialects that get buffered", () => {
    // The exact Q36MTP-TOOL-CALL-NOT-EMITTED shape: a textual <run_command>
    // dialect the server's native parser did not convert to a function_call.
    expect(
      TOOL_CALL_MARKER_LINE_START.test('<run_command>\n{"command": "pwd"}'),
    ).toBe(true);
    expect(
      TOOL_CALL_MARKER_LINE_START.test("prose then\n  <tool_call>{}"),
    ).toBe(true);
    // Prose mentions mid-line must NOT activate buffering.
    expect(
      TOOL_CALL_MARKER_LINE_START.test("I'll use the <run_command> tool now"),
    ).toBe(false);
  });

  it("restore path bypasses marker re-detection so restored text cannot be re-swallowed", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    // emitDelta must accept the bypass flag and gate detection on it: the
    // restored authoritative text often still contains the marker that
    // activated buffering; re-scanning it would re-suppress it forever.
    expect(source).toContain("bypassToolMarkerDetection = false,");
    expect(source).toContain(
      "if (!isReasoningDelta && !bypassToolMarkerDetection) {",
    );
    // The zero-tool restore call must pass bypass=true.
    expect(source).toContain("emitDelta(restoredText, false, false, true);");
    // Detection uses the shared exported pattern (kept in sync with tests).
    expect(source).toContain("TOOL_CALL_MARKER_LINE_START");
  });

  it("restored tool-dialect text is fenced and the sanitizer cannot empty it", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );

    // Restore path fences tool-dialect text so markdown renders it verbatim
    // instead of swallowing the tags as HTML.
    expect(source).toContain('"```text\\n" + reconciliation.authoritativeText.trim() + "\\n```"');
    // Final persistence guard: sanitizing a non-empty answer down to nothing
    // (with no executed tool) must preserve the original in a fence rather
    // than persisting a blank assistant turn.
    expect(source).toContain("const preSanitizeContent = fullContent.trim();");
    expect(source).toContain("const visibleAfterSanitize = fullContent");
    expect(source).toContain(
      "Sanitizer emptied a ${preSanitizeContent.length}-char answer",
    );
    expect(source).toContain(
      "receivedToolCalls.filter(Boolean).length === 0",
    );
    // Speculative "generating" statuses must NOT mask the guard.
    expect(source).not.toContain(
      "collectedToolStatuses.length === 0 &&\n          preSanitizeContent",
    );
  });
});

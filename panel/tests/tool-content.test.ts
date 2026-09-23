import { describe, expect, it } from "vitest";
import { appendVisibleToolContent } from "../src/shared/toolContent";

describe("visible content across tool boundaries", () => {
  it("does not retract the initial newline already streamed before a tool call", () => {
    const streamed = "\nCompletion confirmed: REAL_";
    expect(appendVisibleToolContent("", streamed)).toBe(streamed);
  });
  it("keeps every earlier byte and tool offset through multiple rounds", () => {
    let retained = "";
    for (const segment of ["\nFirst  ", "\nSecond\t", "Third\n"]) {
      const visible = retained ? retained + "\n\n" + segment : segment;
      const boundary = appendVisibleToolContent(retained, segment);
      expect(boundary).toBe(visible);
      expect(boundary.startsWith(retained)).toBe(true);
      retained = boundary;
    }
    expect(retained).toBe("\nFirst  \n\n\nSecond\t\n\nThird\n");
  });
  it("does not add empty tool-only passes to the final answer", () => {
    for (const empty of ["", "\n", " \t"]) {
      expect(appendVisibleToolContent("previous", empty)).toBe("previous");
      expect(appendVisibleToolContent("", empty)).toBe("");
    }
  });
});

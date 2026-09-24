import { describe, expect, it } from "vitest";
import { appendVisibleToolContent, visibleToolStreamContent } from "../src/shared/toolContent";

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


describe("tool-only stream prefixes", () => {
  it("holds a blank tool-call preamble and publishes its follow-up without a reset", () => {
    expect(visibleToolStreamContent("", "\n\n")).toBeNull();
    const boundary = appendVisibleToolContent("", "\n\n");
    expect(visibleToolStreamContent(boundary, "Confirmed")).toBe("Confirmed");
  });
  it("does not append then retract separators after earlier visible tool rounds", () => {
    const previous = "Earlier answer  ";
    expect(visibleToolStreamContent(previous, " \n\t")).toBeNull();
    expect(appendVisibleToolContent(previous, " \n\t")).toBe(previous);
    expect(visibleToolStreamContent(previous, "Next")).toBe(previous + "\n\nNext");
  });
  it("retains meaningful indentation and trailing whitespace without trimming", () => {
    expect(visibleToolStreamContent("", "    ")).toBeNull();
    const code = "    return 42;  \n";
    expect(visibleToolStreamContent("", code)).toBe(code);
    expect(visibleToolStreamContent("Prior", code)).toBe("Prior\n\n" + code);
  });
});

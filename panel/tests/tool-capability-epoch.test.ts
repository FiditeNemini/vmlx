import { describe, expect, it } from "vitest";
import {
  toolCapabilityFingerprint,
  toolCapabilityNames,
  toolCapabilityTransitionInstruction,
} from "../src/main/tool-capability-epoch";

const fileInfo = {
  type: "function",
  function: {
    name: "file_info",
    description: "Inspect a file",
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
};

const readFile = {
  type: "function",
  function: {
    name: "read_file",
    description: "Read a file",
    parameters: {
      required: ["path"],
      properties: { path: { type: "string" } },
      type: "object",
    },
  },
};

describe("tool capability epochs", () => {
  it("fingerprints schema content canonically instead of object key order", () => {
    const reordered = {
      type: "function",
      function: {
        parameters: {
          required: ["path"],
          properties: { path: { type: "string" } },
          type: "object",
        },
        description: "Read a file",
        name: "read_file",
      },
    };
    expect(toolCapabilityFingerprint([readFile])).toBe(
      toolCapabilityFingerprint([reordered]),
    );
    expect(toolCapabilityFingerprint([])).toBe("tools:none");
  });

  it("sorts current function names for a stable transition boundary", () => {
    expect(toolCapabilityNames([readFile, fileInfo])).toEqual([
      "file_info",
      "read_file",
    ]);
  });

  it("marks tools OFF to ON and names the current authoritative catalog", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityTransitionInstruction(
        "tools:none",
        current,
        ["file_info"],
      ),
    ).toContain("Current callable functions: file_info.");
  });

  it("repairs a legacy unknown epoch only when tools are now attached", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityTransitionInstruction(undefined, current, ["file_info"]),
    ).toContain("authoritative tool catalog");
    expect(
      toolCapabilityTransitionInstruction(undefined, "tools:none", []),
    ).toBeUndefined();
  });

  it("does not inject a boundary when the latest assistant used this catalog", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityTransitionInstruction(current, current, ["file_info"]),
    ).toBeUndefined();
  });

  it("marks tools ON to OFF without granting stale capabilities", () => {
    const previous = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityTransitionInstruction(previous, "tools:none", []),
    ).toContain("No callable function schemas");
  });
});

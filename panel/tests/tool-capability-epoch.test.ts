import { describe, expect, it } from "vitest";
import {
  toolCapabilityEpochInstruction,
  toolCapabilityFingerprint,
  toolCapabilityNames,
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
      toolCapabilityEpochInstruction(["tools:none"], current, ["file_info"]),
    ).toContain("Current callable functions: file_info.");
  });

  it("repairs a legacy unknown epoch only when tools are now attached", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction([undefined], current, ["file_info"]),
    ).toContain("authoritative tool catalog");
    expect(
      toolCapabilityEpochInstruction([undefined], "tools:none", []),
    ).toBeUndefined();
  });

  it("does not inject a boundary when every known assistant used this catalog", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction([current, current], current, [
        "file_info",
      ]),
    ).toBeUndefined();
  });

  it("marks tools ON to OFF without granting stale capabilities", () => {
    const previous = toolCapabilityFingerprint([fileInfo]);
    expect(
      toolCapabilityEpochInstruction([previous], "tools:none", []),
    ).toContain("No callable function schemas");
  });

  it("keeps the tools-OFF instruction stable after the transition turn", () => {
    const priorTools = toolCapabilityFingerprint([fileInfo]);
    const transition = toolCapabilityEpochInstruction(
      [priorTools],
      "tools:none",
      [],
    );
    const laterSameEpoch = toolCapabilityEpochInstruction(
      [priorTools, "tools:none", "tools:none"],
      "tools:none",
      [],
    );
    expect(laterSameEpoch).toBe(transition);
  });

  it("keeps the tools-ON instruction stable after an older OFF epoch", () => {
    const current = toolCapabilityFingerprint([fileInfo]);
    const transition = toolCapabilityEpochInstruction(["tools:none"], current, [
      "file_info",
    ]);
    const laterSameEpoch = toolCapabilityEpochInstruction(
      ["tools:none", current, current],
      current,
      ["file_info"],
    );
    expect(laterSameEpoch).toBe(transition);
  });

  it("does not add instructions to a fresh or uniformly no-tools chat", () => {
    expect(
      toolCapabilityEpochInstruction([], "tools:none", []),
    ).toBeUndefined();
    expect(
      toolCapabilityEpochInstruction(
        ["tools:none", "tools:none"],
        "tools:none",
        [],
      ),
    ).toBeUndefined();
  });
});

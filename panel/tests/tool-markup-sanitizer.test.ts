import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  stripLeakedToolMarkup,
  TOOL_MARKUP_TAG_NAMES,
} from "../src/shared/toolMarkupSanitizer";

// The release proof's own visible-leak pattern, kept in sync deliberately: a
// sanitizer that leaves behind anything this matches ships a leaking answer.
const RELEASE_PROOF_LEAK_REGEX =
  /<think>|<\/think>|<tool_calls?>|<\/tool_calls?>|<function\b|<\/function>|<invoke\b|<\/invoke>|<parameter\b|<\/parameter>|<arg_key>|<\/arg_key>|<arg_value>|<\/arg_value>|<minimax:tool_call>|<\/minimax:tool_call>|<zyphra_tool_call>|<\/zyphra_tool_call>/i;

describe("leaked tool-markup sanitizer", () => {
  it("removes the orphan closing tags a Zaya turn rendered on screen", () => {
    // Verbatim from the live Electron app (zaya_text, Responses + tools): the
    // model's answer ended with three closing tags and the UI showed them.
    const rendered = [
      "The task is completed; the final response is a three-line receipt.",
      "Copy every character literally, including the dollar sign and both backslashes.",
      "Return this exact three-line rendering receipt and nothing else.",
      "</parameter>",
      "</function>",
      "</zyphra_tool_call>",
    ].join("\n");

    const cleaned = stripLeakedToolMarkup(rendered);

    expect(cleaned).toContain("three-line receipt");
    expect(cleaned).not.toContain("</parameter>");
    expect(cleaned).not.toContain("</function>");
    expect(cleaned).not.toContain("</zyphra_tool_call>");
    expect(RELEASE_PROOF_LEAK_REGEX.test(cleaned)).toBe(false);
  });

  it("still removes complete paired blocks", () => {
    for (const [input, mustKeep] of [
      ["before<tool_call>{\"a\":1}</tool_call>after", "beforeafter"],
      [
        "pre<zyphra_tool_call>call</zyphra_tool_call>post",
        "prepost",
      ],
      ["a<minimax:tool_call>x</minimax:tool_call>b", "ab"],
      ["x<invoke name=\"t\">body</invoke>y", "xy"],
      ["p<parameter name=\"q\">v</parameter>s", "ps"],
      ["m[Calling tool: run({\"a\":1})]n", "mn"],
    ] as const) {
      expect(stripLeakedToolMarkup(input)).toBe(mustKeep);
    }
  });

  it("removes hallucinated Claude-style tool calls, paired and self-closing", () => {
    expect(
      stripLeakedToolMarkup("go <run_command>pwd</run_command> on"),
    ).toBe("go  on");
  });

  it("keeps prose that follows a self-closing hallucinated tool call", () => {
    // Regression: the paired rule ends with `|$`, so it matched the
    // self-closing form and deleted everything after it. The answer text was
    // silently truncated at the tag.
    expect(
      stripLeakedToolMarkup('go <read_file path="a.txt" /> on'),
    ).toBe("go  on");
    expect(
      stripLeakedToolMarkup('<write_file path="x"/>the real answer survives'),
    ).toBe("the real answer survives");
  });

  it("still drops the tail of an unterminated tool call", () => {
    // An opening tag with no close is a broken tool call, not prose: the tail
    // belongs to the call and must not be shown.
    expect(stripLeakedToolMarkup("before <run_command>pwd and then")).toBe(
      "before ",
    );
  });

  it("leaves every orphan tag name in the shared list unrendered", () => {
    for (const name of TOOL_MARKUP_TAG_NAMES) {
      const cleaned = stripLeakedToolMarkup(`answer</${name}>`);
      expect(cleaned, `orphan </${name}> survived`).toBe("answer");
    }
  });

  it("does not touch ordinary prose or unrelated markup", () => {
    for (const text of [
      "The function returns 4 and the parameter is optional.",
      "Use <div> and <span> in HTML.",
      "A generic <thing> tag stays.",
      "2 < 3 and 5 > 4",
    ]) {
      expect(stripLeakedToolMarkup(text)).toBe(text);
    }
  });

  it("is used by BOTH the final-save and abort/partial-save paths", () => {
    const source = readFileSync(
      new URL("../src/main/ipc/chat.ts", import.meta.url),
      "utf8",
    );
    // These two were byte-identical copies; a fix to one was inert in the other.
    expect(source).toContain("fullContent = stripLeakedToolMarkup(fullContent);");
    expect(source).toContain(
      "partialContent = stripLeakedToolMarkup(partialContent);",
    );
    // And the duplicated inline regexes are gone.
    expect(
      source.match(/<zyphra_tool_call>\[\\s\\S\]\*\?<\\\/zyphra_tool_call>/g),
    ).toBeNull();
  });
});

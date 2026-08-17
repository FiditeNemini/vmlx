import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  stripLeakedToolMarkup,
  stripStreamingToolTags,
  toolMarkupHoldbackLength,
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

  it("removes a bare think marker that survived into visible content", () => {
    // Verbatim from an omni AUDIO turn (nemotron-omni-nano): the server had
    // already emitted reasoning separately and a stray close tag still reached
    // the answer, which the release proof correctly flagged as a parser leak.
    const rendered = [
      "A synthetic electronic tone is heard.",
      "</think>",
      "A synthetic electronic tone is heard.",
    ].join("\n");
    const cleaned = stripLeakedToolMarkup(rendered);
    expect(cleaned).toContain("synthetic electronic tone");
    expect(/<\/?think>/.test(cleaned)).toBe(false);
  });

  it("leaves a paired think block to the reasoning extraction path", () => {
    // Only BARE markers are residue; a complete block is the reasoning
    // parser's business, not this function's.
    const paired = "<think>deliberating</think>the answer";
    expect(stripLeakedToolMarkup(paired)).toBe(paired);
  });

  it("does not eat the word think from prose", () => {
    const prose = "I think the answer is four.";
    expect(stripLeakedToolMarkup(prose)).toBe(prose);
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

  it("withholds only fragments that could still become a tool tag", () => {
    expect(toolMarkupHoldbackLength("answer</parameter")).toBe(
      "</parameter".length,
    );
    expect(toolMarkupHoldbackLength("done<")).toBe(1);
    expect(toolMarkupHoldbackLength("done</")).toBe(2);
    // A complete tag needs no holdback — it is strippable right now.
    expect(toolMarkupHoldbackLength("answer</parameter>")).toBe(0);
    // Ordinary prose is untouched.
    expect(toolMarkupHoldbackLength("a normal sentence.")).toBe(0);
    expect(toolMarkupHoldbackLength("2 < 3 and more")).toBe(0);
  });

  it("never shows an orphan tag to the renderer even when split across deltas", () => {
    // Models tokenize "</parameter>" as several tokens; this is the common
    // case, so per-delta stripping alone is not enough.
    const deltas = ["The answer.", "</", "parameter", ">", " done"];
    let holdback = "";
    let shown = "";
    for (const raw of deltas) {
      const merged = holdback + raw;
      const hold = toolMarkupHoldbackLength(merged);
      holdback = hold > 0 ? merged.slice(merged.length - hold) : "";
      const emittable = hold > 0 ? merged.slice(0, merged.length - hold) : merged;
      shown += stripStreamingToolTags(emittable);
    }
    shown += holdback; // end-of-stream flush
    expect(shown).toBe("The answer. done");
    expect(RELEASE_PROOF_LEAK_REGEX.test(shown)).toBe(false);
  });

  it("flushes a held fragment that turned out to be prose", () => {
    // An answer ending in "<" holds back one character; dropping it would
    // silently truncate the answer.
    const merged = "compare 2 <";
    const hold = toolMarkupHoldbackLength(merged);
    expect(hold).toBe(1);
    const shown = merged.slice(0, merged.length - hold) + merged.slice(-hold);
    expect(shown).toBe("compare 2 <");
  });

  it("streaming strip leaves paired openings for the tool buffer to handle", () => {
    // Mid-stream the closing half has not arrived; suppressing the opening is
    // speculative tool buffering's job, not this function's.
    expect(stripStreamingToolTags("<tool_call>{")).toBe("<tool_call>{");
    expect(stripStreamingToolTags("x</tool_call>")).toBe("x");
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

/**
 * Tool-dialect tag names that models emit as textual tool calls when the
 * server's native parser did not convert them into function_call items.
 *
 * Kept as one list so the paired-block strip, the orphan-tag strip, and the
 * release proof's leak detector cannot drift apart.
 */
export const TOOL_MARKUP_TAG_NAMES = [
  "zyphra_tool_call",
  "minimax:tool_call",
  "tool_calls",
  "tool_call",
  "invoke",
  "function",
  "parameter",
  "arg_key",
  "arg_value",
  "read_file",
  "write_file",
  "run_command",
  "search_files",
  "edit_file",
  "list_directory",
  "execute_command",
  "bash",
] as const;

const HALLUCINATED_TOOL_TAGS =
  "read_file|write_file|run_command|search_files|edit_file|list_directory|execute_command|bash";

const tagAlternation = TOOL_MARKUP_TAG_NAMES.map((name) =>
  name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
).join("|");

/**
 * Closing (or bare container) tool tags with no matching opening tag.
 *
 * Every paired strip below removes a complete block. When a model emits only
 * the tail of a dialect — which happens whenever the opening portion was
 * consumed as a real tool call, or the model simply hallucinates a closer —
 * nothing removed it and it reached the user's screen. Observed live in the
 * Electron app on a Zaya-8B turn that rendered:
 *
 *     </parameter>
 *     </function>
 *     </zyphra_tool_call>
 *
 * as visible assistant prose.
 */
const ORPHAN_TOOL_TAG_REGEX = new RegExp(
  `<\\/(?:${tagAlternation})>|<(?:tool_calls|arg_key|arg_value)>`,
  "g",
);

/**
 * Every complete orphan/container tag this module strips while streaming.
 * Openings of paired dialects are deliberately NOT here: those activate
 * speculative tool buffering upstream, which already suppresses them.
 */
const STREAM_STRIPPABLE_TAGS = [
  ...TOOL_MARKUP_TAG_NAMES.map((name) => `</${name}>`),
  "<tool_calls>",
  "<arg_key>",
  "<arg_value>",
];

/**
 * How many trailing characters of a streamed chunk must be withheld because
 * they could still grow into a tool tag.
 *
 * Models tokenize `</parameter>` as several tokens ("</", "parameter", ">"), so
 * stripping per-delta without a holdback misses the split case - which is the
 * common case, not the rare one. Same technique as the <think> marker holdback.
 */
export function toolMarkupHoldbackLength(content: string): number {
  let hold = 0;
  for (const tag of STREAM_STRIPPABLE_TAGS) {
    const max = Math.min(tag.length - 1, content.length);
    for (let len = 1; len <= max; len++) {
      if (content.endsWith(tag.slice(0, len))) hold = Math.max(hold, len);
    }
  }
  return hold;
}

/**
 * Remove complete orphan/container tool tags from streamed content.
 *
 * Paired blocks are NOT touched here: mid-stream the closing half has not
 * arrived yet, and speculative tool buffering already withholds an opening
 * marker. This only drops tags that are complete and cannot be prose.
 */
export function stripStreamingToolTags(content: string): string {
  return content.replace(ORPHAN_TOOL_TAG_REGEX, "");
}

/**
 * Strip textual tool-call markup the server did not parse.
 *
 * Shared by the final-save path and the abort/partial-save path: those two were
 * byte-identical copies, and a fix applied to one of them was inert in the
 * other.
 *
 * Callers own their own template-token and Harmony-protocol handling; this
 * function is only about tool dialects.
 */
export function stripLeakedToolMarkup(content: string): string {
  let text = content;
  text = text.replace(/<zyphra_tool_call>[\s\S]*?<\/zyphra_tool_call>/g, "");
  text = text.replace(/<minimax:tool_call>[\s\S]*?<\/minimax:tool_call>/g, "");
  text = text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "");
  text = text.replace(/\[Calling tool:\s*\w+\(\{[\s\S]*?\}\)\]/g, "");
  text = text.replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/g, "");
  text = text.replace(/<function(?:=|\b)[\s\S]*?(?:<\/function>|$)/g, "");
  text = text.replace(/<parameter\b[^>]*>[\s\S]*?<\/parameter>/g, "");
  // Self-closing BEFORE the paired form. The paired rule ends with `|$`, so it
  // matches a self-closing tag too and then runs to end-of-string, deleting
  // every character of real prose that followed a complete
  // `<read_file path="a.txt" />`. Ordered this way the tag alone is removed and
  // the answer survives; previously this rule sat after the paired one and
  // could never match anything.
  text = text.replace(
    new RegExp(`<(?:${HALLUCINATED_TOOL_TAGS})\\b[^>]*\\/>`, "g"),
    "",
  );
  text = text.replace(
    new RegExp(
      `<(?:${HALLUCINATED_TOOL_TAGS})\\b[^>]*>[\\s\\S]*?(?:<\\/(?:${HALLUCINATED_TOOL_TAGS})>|$)`,
      "g",
    ),
    "",
  );
  // Orphans last: the paired rules above have already taken whole blocks, so
  // whatever tail remains had no opening tag to pair with.
  text = text.replace(ORPHAN_TOOL_TAG_REGEX, "");
  // Reasoning markers are owned by the reasoning parser, not by this function —
  // but by FINALIZE any that survive into visible content are residue, and a
  // lone marker is never legitimate prose. Observed live on an omni AUDIO turn
  // (nemotron-omni-nano), which rendered:
  //   A synthetic electronic tone is heard.
  //   </think>
  //   A synthetic electronic tone is heard.
  // i.e. the server had already emitted reasoning separately and a stray close
  // tag still reached the answer. Paired <think>…</think> blocks are left to the
  // reasoning extraction path; only bare markers are removed here.
  text = text.replace(/(?:^|\n)\s*<\/?think>\s*(?=\n|$)/g, "\n");
  return text;
}

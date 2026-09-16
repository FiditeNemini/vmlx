import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { formatMcpToolResult } from "../src/shared/mcpToolResult";

describe("MCP model-facing tool result", () => {
  it.each(["missing field: tax", "  indented\nline\\n\n", "null", "拒否: field"])(
    "keeps original error detail and whitespace: %s", (detail) => {
      expect(formatMcpToolResult({ is_error: true, content: detail, error_message: null }, 50000))
        .toBe(`Error: ${detail}`);
    },
  );

  it("prefers an explicit message without stripping it", () => {
    expect(formatMcpToolResult({ is_error: true, content: "detail", error_message: "  denied\n" }, 50000))
      .toBe("Error:   denied\n");
  });

  it.each([undefined, null, "", " \n\t"])("names genuinely absent error detail: %s", (content) => {
    expect(formatMcpToolResult({ is_error: true, content }, 50000)).toBe("Error: Unknown error");
  });

  it("uses content when the explicit message is blank", () => {
    expect(formatMcpToolResult({ is_error: true, content: "missing", error_message: " \n" }, 50000))
      .toBe("Error: missing");
  });

  it.each([{ field: null, literal: "null" }, ["a", 2], false, 0])(
    "preserves structured error types: %s", (content) => {
      const text = formatMcpToolResult({ is_error: true, content }, 50000);
      expect(JSON.parse(text.slice("Error: ".length))).toEqual(content);
    },
  );

  it.each([null, "", "  a\n", { n: null }, [1, "null"], false, 0])(
    "keeps existing success rendering: %s", (content) => {
      expect(formatMcpToolResult({ content }, 50000))
        .toBe(typeof content === "string" ? content : JSON.stringify(content, null, 2));
    },
  );

  it.each([false, true])("bounds both success and rejection: error=%s", (is_error) => {
    const full = `${is_error ? "Error: " : ""}${"a".repeat(100)}`;
    expect(formatMcpToolResult({ is_error, content: "a".repeat(100) }, 20)).toBe(
      `${full.slice(0, 20)}\n\n[Truncated — showing first 20 of ${full.length} characters]`,
    );
    expect(formatMcpToolResult({ is_error, content: "a".repeat(100) }, full.length)).toBe(full);
  });

  it("sends the same bounded result to the tool display and both wire APIs", () => {
    const source = readFileSync("src/main/ipc/chat.ts", "utf8");
    const mcp = source.slice(source.indexOf("const result = await execRes.json();"),
      source.indexOf("// Inject media from read_image/read_video tool results"));
    expect(mcp).toContain("resultText = formatMcpToolResult(");
    expect(mcp).toContain("overrides?.toolResultMaxChars || 50000");
    expect(mcp).toContain('"error",\n                      tc.function.name,\n                      resultText,');
    expect(mcp).toContain("call_id: tc.id,\n                    output: resultText");
    expect(mcp).toContain('role: "tool", tool_call_id: tc.id, content: resultText');
    expect(mcp).not.toContain('result.error_message || "Unknown error"');
  });
});

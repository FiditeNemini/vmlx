import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { boundToolResultText, formatMcpToolResult } from "../src/shared/mcpToolResult";

describe("MCP shared text bound", () => {
  it.each(["", "finite detail", "  indented\nline\t  ", "拒否 😀 café", String.raw`literal\n\t\"\\`])(
    "preserves untruncated text exactly: %s", (text) => {
      expect(boundToolResultText(text, text.length)).toBe(text);
      expect(boundToolResultText(text, text.length + 1)).toBe(text);
    },
  );

  it("keeps the existing leading UTF-16 slice plus suffix, not a total-length cap", () => {
    const text = "A😀B";
    expect(text.length).toBe(4);
    expect(boundToolResultText(text, 2)).toBe(
      "A\uD83D\n\n[Truncated — showing first 2 of 4 characters]",
    );
    expect(boundToolResultText(text, 3)).toBe(
      "A😀\n\n[Truncated — showing first 3 of 4 characters]",
    );
    expect(boundToolResultText(text, 2).length).toBeGreaterThan(2);
  });

  it.each(["Error (503): ", "Tool execution error: "])(
    "bounds the whole exceptional result including its %s prefix", (prefix) => {
      const text = prefix + String.raw`  rejected\n"literal" 拒否 😀 `.repeat(12);
      const maxChars = 40;
      const expected = text.slice(0, maxChars) +
        `\n\n[Truncated — showing first ${maxChars} of ${text.length} characters]`;
      expect(boundToolResultText(text, maxChars)).toBe(expected);
      expect(boundToolResultText(text, maxChars).startsWith(prefix)).toBe(true);
      expect(boundToolResultText(text, text.length)).toBe(text);
    },
  );

  it("retains the same bound for formatted success and tool rejection", () => {
    const detail = "  rejected\\n 拒否 😀 ".repeat(10);
    expect(formatMcpToolResult({ content: detail }, 31))
      .toBe(boundToolResultText(detail, 31));
    expect(formatMcpToolResult({ is_error: true, content: detail }, 31))
      .toBe(boundToolResultText(`Error: ${detail}`, 31));
    const source = readFileSync("src/shared/mcpToolResult.ts", "utf8");
    expect(source).toMatch(/return boundToolResultText\(\s*text,\s*maxChars\s*\)/);
  });
});

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

  it("bounds non-2xx and thrown errors before displaying or recording either wire result", () => {
    const source = readFileSync("src/main/ipc/chat.ts", "utf8");
    const start = source.indexOf('const execRes = await fetch(`${baseUrl}/v1/mcp/execute`');
    const end = source.indexOf("// Inject media from read_image/read_video tool results", start);
    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    const mcp = source.slice(start, end);
    const httpStart = mcp.indexOf("if (!execRes.ok)");
    const successStart = mcp.indexOf("const result = await execRes.json();");
    const catchStart = mcp.indexOf("} catch (err: any)");
    const flushStart = mcp.indexOf("await flushToolStatusToRenderer();");
    expect(httpStart).toBeGreaterThan(-1);
    expect(successStart).toBeGreaterThan(httpStart);
    expect(catchStart).toBeGreaterThan(successStart);
    expect(flushStart).toBeGreaterThan(catchStart);
    const http = mcp.slice(httpStart, successStart);
    const caught = mcp.slice(catchStart, flushStart);
    expect(http).toMatch(/resultText = boundToolResultText\(\s*`Error \(\$\{execRes.status\}\): \$\{errText\}`,\s*overrides\?\.toolResultMaxChars \|\| 50000,?\s*\)/);
    expect(caught).toMatch(/resultText = boundToolResultText\(\s*`Tool execution error: \$\{err.message\}`,\s*overrides\?\.toolResultMaxChars \|\| 50000,?\s*\)/);
    const boundedDisplay = /emitToolStatus\(\s*"error",\s*tc.function.name,\s*resultText,\s*toolIteration,\s*tc.id,?\s*\)/;
    expect(http).toMatch(boundedDisplay);
    expect(caught).toMatch(boundedDisplay);
    expect(caught).not.toMatch(/emitToolStatus\(\s*"error",\s*tc.function.name,\s*err.message,/);
    const abortGuard = 'if (err?.name === "AbortError") throw err;';
    expect(caught).toContain(abortGuard);
    expect(caught.indexOf(abortGuard)).toBeLessThan(caught.indexOf("resultText = boundToolResultText("));
    const continuation = mcp.slice(flushStart);
    expect(continuation).toMatch(/type: "function_call_output",\s*call_id: tc.id,\s*output: resultText/);
    expect(continuation).toMatch(/role: "tool",\s*tool_call_id: tc.id,\s*content: resultText/);
  });
});

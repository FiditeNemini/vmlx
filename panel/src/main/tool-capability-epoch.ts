import { createHash } from "node:crypto";

const NO_TOOLS_FINGERPRINT = "tools:none";

export interface ToolCapabilityDefinition {
  type?: unknown;
  function?: {
    name?: unknown;
    description?: unknown;
    parameters?: unknown;
  };
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => [key, canonicalize(nested)]),
  );
}

export function toolCapabilityFingerprint(
  tools: ToolCapabilityDefinition[],
): string {
  if (tools.length === 0) return NO_TOOLS_FINGERPRINT;
  const canonicalTools = tools
    .map((tool) => ({
      type: tool.type,
      function: canonicalize(tool.function),
    }))
    .sort((left, right) =>
      String(left.function && (left.function as any).name).localeCompare(
        String(right.function && (right.function as any).name),
      ),
    );
  return `tools:${createHash("sha256")
    .update(JSON.stringify(canonicalTools))
    .digest("hex")
    .slice(0, 20)}`;
}

export function toolCapabilityNames(
  tools: ToolCapabilityDefinition[],
): string[] {
  return tools
    .map((tool) =>
      typeof tool.function?.name === "string" ? tool.function.name : "",
    )
    .filter(Boolean)
    .sort();
}

export function toolCapabilityTransitionInstruction(
  previousFingerprint: string | undefined,
  currentFingerprint: string,
  currentToolNames: string[],
): string | undefined {
  // Legacy assistant rows have no fingerprint. Avoid injecting anything into
  // ordinary no-tools chats, while still repairing a legacy tools-OFF -> ON
  // transition the first time the current catalog is attached.
  const changed =
    previousFingerprint !== undefined
      ? previousFingerprint !== currentFingerprint
      : currentToolNames.length > 0;
  if (!changed) return undefined;

  if (currentToolNames.length === 0) {
    return [
      "Tool capability epoch changed for this request.",
      "No callable function schemas are attached to the current turn.",
      "Earlier assistant statements or tool calls belong to an older capability epoch and do not grant tools now.",
    ].join(" ");
  }

  return [
    "Tool capability epoch changed for this request.",
    "The function schemas attached to the current request are the authoritative tool catalog for this turn.",
    `Current callable functions: ${currentToolNames.join(", ")}.`,
    "Ignore earlier assistant statements about tool availability when they conflict with this current catalog.",
  ].join(" ");
}


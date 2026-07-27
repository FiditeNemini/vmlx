import crypto from "node:crypto";
import {
  chmodSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

// @ts-expect-error The production proof harness is deliberately plain Node ESM.
import {
  applyAssertionFailureStatus,
  applyTopLevelCorrelationStatus,
  assertCdpExpressionSyntax,
  captureBundleGenerationContract,
  deriveProvenSurfaces,
  correlateTerminalResponseToCacheExecution,
  isCacheRequestCorrelationVerified,
  isServerRequestCorrelationVerified,
  localRendererModuleEvidence,
  ownedUiProducerPid,
  parseResolvedSamplingKwargs,
  privateCacheAttestationSessionArgs,
  readPrivateExternalJson,
  resolveIndependentBundleGenerationDefaults,
  upsertBoundedDomSample,
  uniqueProofBasename,
  validateExactToolLoopEvidence,
  validateAttachOnlyLifecycle,
  validateGenerationDefaultsEvidence,
  validateModelBundleBinding,
  validateOwnedRunIntent,
  validatePairedApiEvidence,
  validateOwnedReuseSessionAttestation,
  validateOwnedUiReleaseSentinel,
  validateReasoningEvidence,
  validateRenderedDomEvidence,
  validateRequestCorrelatedCacheEvidence,
  validateServerCacheEvidence,
  validateUiRuntimeProvenance,
  viteRawRendererModulePath,
  waitForOwnedUiReleaseSentinel,
  writePrivateArtifactFile,
} from "../scripts/live-real-ui-model-proof.mjs";

const sha = "a".repeat(64);
const otherSha = "b".repeat(64);
const testExecutablePath = realpathSync(process.execPath);
const testExecutableBytes = readFileSync(testExecutablePath);
const testExecutableSha = crypto
  .createHash("sha256")
  .update(testExecutableBytes)
  .digest("hex");
const testExecutablePathSha = crypto
  .createHash("sha256")
  .update(testExecutablePath)
  .digest("hex");
const testExecutablePrefixPath = path.dirname(testExecutablePath);
const testExecutablePrefixPathSha = crypto
  .createHash("sha256")
  .update(testExecutablePrefixPath)
  .digest("hex");
const assistantIds = ["assistant-1", "assistant-2", "assistant-3"];
const prompts = [
  "First bound prompt",
  "Second bound prompt",
  "Preserve $43 and render \\(47 \\times 19 = 893 < 920 = 46 \\times 20\\).",
];
const persistedContents = [
  "REAL_UI_LIVE_TOOL_ONE Answer complete",
  "REAL_UI_LIVE_TOOL_TWO Answer complete",
  "$43 and \\(47 \\times 19 = 893 < 920 = 46 \\times 20\\)",
];
const renderedContents = [
  persistedContents[0],
  persistedContents[1],
  "$43 and 47 × 19 = 893 < 920 = 46 × 20",
];

describe("owned UI producer identity", () => {
  it("binds orchestrated attestations to the directly observed parent worker", () => {
    expect(ownedUiProducerPid({
      orchestrated: true,
      harnessPid: 65680,
      parentPid: 65667,
    })).toBe(65667);
    expect(ownedUiProducerPid({
      orchestrated: false,
      harnessPid: 65680,
      parentPid: 65667,
    })).toBe(65680);
  });

  it("fails closed when the orchestrated parent PID is unavailable", () => {
    expect(() => ownedUiProducerPid({
      orchestrated: true,
      harnessPid: 65680,
      parentPid: 1,
    })).toThrow(/producer PID is invalid/);
  });
});
const defaults = {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  minP: 0.05,
  repeatPenalty: 1.1,
  maxNewTokens: 2048,
  source: "jang_config",
};
const modelBundleAttestation = {
  schema: "vmlx-bundle-config-v1",
  directory_state: "available",
  files: {
    "config.json": { state: "present", size_bytes: 2, sha256: sha },
    "generation_config.json": {
      state: "present",
      size_bytes: 2,
      sha256: sha,
    },
    "jang_config.json": { state: "present", size_bytes: 2, sha256: sha },
    "tokenizer_config.json": {
      state: "present",
      size_bytes: 2,
      sha256: sha,
    },
    "chat_template.jinja": {
      state: "present",
      size_bytes: 2,
      sha256: sha,
    },
  },
  aggregate_sha256: sha,
  fingerprint_sha256: sha,
};

describe("generated CDP expression syntax", () => {
  it("accepts the renderer navigation expression used by the live UI worker", () => {
    const expression = `
      (async () => {
        const created = { session: { id: "session-1" } };
        window.dispatchEvent(new CustomEvent("vmlx:navigate", {
          detail: {
            mode: "server",
            panel: "session",
            sessionId: created.session.id,
          },
        }));
      })()
    `;

    expect(assertCdpExpressionSyntax(expression, "renderer-real-ui-chat"))
      .toBe(expression);
  });

  it("rejects the missing dispatchEvent close that broke the release gate", () => {
    const malformed = `
      window.dispatchEvent(new CustomEvent("vmlx:navigate", {
        detail: { mode: "server" },
      });
    `;

    expect(() => assertCdpExpressionSyntax(malformed, "renderer-real-ui-chat"))
      .toThrow(/generated an invalid CDP expression/);
  });

  it("keeps private cache attestation cleanup state in finally scope", async () => {
    const cleanupValues: string[] = [];
    const expression = `
      (async () => {
        let proofSessionId = "session-1";
        let privateConfigRestoreAdditionalArgs = "--existing";
        const privateCacheAttestationArgs = "--private-proof";
        try {
          throw new Error("body failed");
        } finally {
          if (
            proofSessionId
            && privateConfigRestoreAdditionalArgs !== null
            && privateCacheAttestationArgs
          ) {
            cleanupValues.push(privateCacheAttestationArgs);
          }
        }
      })()
    `;

    assertCdpExpressionSyntax(expression, "renderer-real-ui-chat");
    await expect(eval(expression)).rejects.toThrow("body failed");
    expect(cleanupValues).toEqual(["--private-proof"]);
  });

  it("declares the production worker attestation value before its try/finally", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const workerScopeStart = harnessSource.indexOf(
      "        let proofSessionId = null;",
    );
    const declarationIndex = harnessSource.indexOf(
      "        const privateCacheAttestationArgs = ${JSON.stringify(privateCacheAttestationArgs)};",
      workerScopeStart,
    );
    const tryIndex = harnessSource.indexOf("        try {", workerScopeStart);
    const finallyIndex = harnessSource.indexOf(
      "        } finally {",
      workerScopeStart,
    );
    const cleanupReadIndex = harnessSource.indexOf(
      "            && privateCacheAttestationArgs",
      finallyIndex,
    );

    expect(workerScopeStart).toBeGreaterThan(-1);
    expect(declarationIndex).toBeGreaterThan(workerScopeStart);
    expect(declarationIndex).toBeLessThan(tryIndex);
    expect(tryIndex).toBeLessThan(finallyIndex);
    expect(cleanupReadIndex).toBeGreaterThan(finallyIndex);
  });

  it("double-escapes every backslash inside the generated worker template", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const workerStart = harnessSource.indexOf(
      "      rendererResult = await evaluate(cdp, `",
    );
    const workerEnd = harnessSource.indexOf(
      "\n    `, 1_200_000)",
      workerStart,
    );
    const workerTemplate = harnessSource.slice(workerStart, workerEnd);
    const hazards: string[] = [];
    for (let index = 0; index < workerTemplate.length; index += 1) {
      if (workerTemplate[index] !== "\\") continue;
      let runLength = 0;
      while (workerTemplate[index + runLength] === "\\") {
        runLength += 1;
      }
      if (runLength % 2 === 1) {
        hazards.push(workerTemplate.slice(index, index + runLength + 1));
      }
      index += runLength - 1;
    }

    expect(workerStart).toBeGreaterThan(-1);
    expect(workerEnd).toBeGreaterThan(workerStart);
    expect(hazards).toEqual([]);
  });

  it("binds the live proof to the product-owned Chat Settings controls", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const toolbarSource = readFileSync(
      path.resolve("src/renderer/src/components/layout/ChatModeToolbar.tsx"),
      "utf8",
    );
    const sessionViewSource = readFileSync(
      path.resolve("src/renderer/src/components/sessions/SessionView.tsx"),
      "utf8",
    );
    const chatSettingsSource = readFileSync(
      path.resolve("src/renderer/src/components/chat/ChatSettings.tsx"),
      "utf8",
    );

    expect(toolbarSource).toContain(
      'data-vmlx-control="chat-settings"',
    );
    expect(sessionViewSource).toContain(
      'data-vmlx-control="chat-settings"',
    );
    expect(chatSettingsSource).toContain(
      'data-vmlx-surface="chat-settings"',
    );
    expect(chatSettingsSource).toContain(
      'data-vmlx-control="chat-settings-save"',
    );
    expect(chatSettingsSource).toContain(
      "data-vmlx-state={saving ? 'saving' : dirty ? 'dirty' : 'saved'}",
    );
    expect(harnessSource).toContain(
      'document.querySelectorAll(\'[data-vmlx-control="chat-settings"]\')',
    );
    expect(harnessSource).toContain(
      'document.querySelector(\'[data-vmlx-surface="chat-settings"]\')',
    );
    expect(harnessSource).toContain(
      '\'[data-vmlx-control="chat-settings-save"]\'',
    );
    expect(harnessSource).toContain(
      "const button = buttons.length === 1 ? buttons[0] : null;",
    );
    expect(harnessSource).toContain(
      "button.getAttribute('data-vmlx-state') === 'dirty'",
    );
    expect(harnessSource).toContain(
      "current.getAttribute('data-vmlx-state') === 'saved'",
    );
    expect(harnessSource).not.toContain(
      "(button.textContent || '').replace(/\\\\s+/g, ' ').trim() === 'Save'",
    );
    expect(harnessSource).not.toContain(
      ".trim() === 'Chat'\n            ) || null,",
    );
    expect(harnessSource).toContain("builtinInput.click();");
    expect(harnessSource).toContain("input.click();");
    expect(harnessSource).toContain("'visible Working Directory input'");
    expect(harnessSource).not.toContain("checkedSetter.call(");
  });

  it("keeps streaming proof capture linear and bounds full DOM samples", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );

    expect(harnessSource).toContain("const maxDomSamplesPerMessage = 24;");
    expect(harnessSource).toContain("const streamTraceState = new Map();");
    expect(harnessSource).toContain(
      "const upsertBoundedDomSample = ${upsertBoundedDomSample.toString()}",
    );
    expect(harnessSource).toContain(
      "fullContentLength: data.fullContent.length",
    );
    expect(harnessSource).toContain(
      "scheduleDomSample(messageId, event + ':' + channel, event !== 'stream')",
    );
    expect(harnessSource).not.toContain("payload: data,\n          });");
  });

  it("preserves the terminal DOM state without ever exceeding the hard sample cap", () => {
    const samples: Array<Record<string, unknown>> = [];
    const state = {
      count: 0,
      lastSignature: "",
      lastStoredIndex: -1,
    };
    for (let index = 1; index <= 30; index += 1) {
      upsertBoundedDomSample(
        samples,
        state,
        {
          messageId: "assistant-1",
          answerText: "x".repeat(index),
          reasoningText: "r",
          katexCount: 0,
          toolCards: [],
        },
        24,
      );
    }
    expect(samples).toHaveLength(24);
    expect(state.count).toBe(24);

    upsertBoundedDomSample(
      samples,
      state,
      {
        messageId: "assistant-1",
        answerText: "terminal answer",
        reasoningText: "terminal reasoning",
        katexCount: 1,
        toolCards: [{ phase: "result" }],
      },
      24,
      true,
    );
    expect(samples).toHaveLength(24);
    expect(samples.at(-1)?.answerText).toBe("terminal answer");
  });

  it("stages and attests each V5 cache phase before the real Start click", () => {
    const harnessSource = readFileSync(
      path.resolve("scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const preflightSource = readFileSync(
      path.resolve("scripts/scoped-release-preflight-18.py"),
      "utf8",
    );

    expect(harnessSource).toContain(
      "usePagedCache: Boolean(activeReleasePhase.paged_ram)",
    );
    expect(harnessSource).toContain(
      "kvCacheQuantization: String(activeReleasePhase.kv_cache_quantization || 'none')",
    );
    expect(harnessSource).toContain(
      "const releasePagedCacheMemoryPercent = activeReleasePhase?.paged_ram",
    );
    expect(harnessSource).toMatch(
      /const releasePagedCacheMemoryPercent = activeReleasePhase\?\.paged_ram\s+\? 10\s+: null/,
    );
    expect(harnessSource).toContain(
      "{ cacheMemoryPercent: releasePagedCacheMemoryPercent }",
    );
    expect(
      harnessSource.match(
        /cacheMemoryPercent: releasePagedCacheMemoryPercent/g,
      ),
    ).toHaveLength(2);
    expect(harnessSource).not.toContain(
      "const releasePagedCacheMemoryPercent =\n          ${",
    );
    expect(preflightSource).toContain(
      '"VMLINUX_REAL_UI_EXPECT_PAGED_CACHE": (',
    );
    expect(preflightSource).toContain(
      '"VMLINUX_REAL_UI_MAX_TOKENS": "2048"',
    );
  });
});

function canonicalJson(value: unknown): string {
  const normalize = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(normalize);
    if (!node || typeof node !== "object") return node;
    return Object.fromEntries(
      Object.keys(node as Record<string, unknown>)
        .sort()
        .map((key) => [
          key,
          normalize((node as Record<string, unknown>)[key]),
        ]),
    );
  };
  return JSON.stringify(normalize(value)).replace(
    /[^\x20-\x7e]/g,
    (character) =>
      `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

function canonicalHash(value: unknown): string {
  return crypto.createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function ownedRunIntent(
  primaryBundlePath: string,
  nativeBundlePath: string,
  harnessRoot: string,
): Record<string, unknown> {
  const harnessPaths = {
    ui: "panel/scripts/live-real-ui-model-proof.mjs",
    api: "tests/cross_matrix/run_agentic_protocol_matrix.py",
    cache: "tests/cross_matrix/run_cache_hierarchy_live_gate.py",
    semantic: "panel/scripts/scoped-release-preflight-18.py",
  };
  const harnesses = Object.fromEntries(
    Object.entries(harnessPaths).map(([name, relativePath]) => {
      const absolutePath = path.join(harnessRoot, relativePath);
      mkdirSync(path.dirname(absolutePath), { recursive: true });
      writeFileSync(absolutePath, `${name}-harness`);
      return [
        name,
        {
          relative_path: relativePath,
          sha256: crypto
            .createHash("sha256")
            .update(readFileSync(absolutePath))
            .digest("hex"),
        },
      ];
    }),
  );
  const primary = {
    representative_id: "primary_tq_supported",
    bundle_role: "primary",
    session_policy: "primary_stable_session",
    model: "primary-model",
    model_bundle_path: primaryBundlePath,
    bundle_fingerprint_sha256: "1".repeat(64),
    native_cache_policy: "standard_kv",
  };
  const phasePlan = [
    {
      phase_index: 0,
      phase_name: "primary_ssd_only_store",
      cache_policy: "q4",
      paged_ram: false,
      operation: "store",
      restart_required: false,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-reasoning-render-store",
      ui_turn_count: 1,
      api_action_profile: "full-agentic-plus-cache-store",
    },
    {
      phase_index: 1,
      phase_name: "primary_ssd_only_restart_probe",
      cache_policy: "q4",
      paged_ram: false,
      operation: "probe",
      restart_required: true,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-tool-restart-probe",
      ui_turn_count: 1,
      api_action_profile: "cache-probe",
    },
    {
      phase_index: 2,
      phase_name: "primary_paged_on_store",
      cache_policy: "q4",
      paged_ram: true,
      operation: "store-evict-refault",
      restart_required: true,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-history-paged-evict-refault",
      ui_turn_count: 1,
      api_action_profile: "cache-evict-refault",
    },
    {
      phase_index: 3,
      phase_name: "primary_paged_on_restart_probe",
      cache_policy: "q4",
      paged_ram: true,
      operation: "probe",
      restart_required: true,
      kv_cache_quantization: "auto",
      tq_policy: "auto-model-safe-required",
      ui_action_profile: "primary-restart-followup",
      ui_turn_count: 1,
      api_action_profile: "cache-restart-probe",
    },
    {
      phase_index: 4,
      phase_name: "primary_tq_off",
      cache_policy: "ssd-only",
      paged_ram: false,
      operation: "store-probe",
      restart_required: true,
      kv_cache_quantization: "none",
      tq_policy: "explicit-off",
      ui_action_profile: "primary-tq-off-probe",
      ui_turn_count: 1,
      api_action_profile: "cache-tq-off-store-probe",
    },
  ].map((phase) => ({ ...primary, ...phase }));
  phasePlan.push({
    phase_index: 5,
    phase_name: "native_exception",
    representative_id: "secondary_native_exception",
    bundle_role: "native",
    cache_policy: "native",
    paged_ram: false,
    operation: "switch-validate",
    restart_required: true,
    session_policy: "distinct_native_session",
    kv_cache_quantization: "none",
    tq_policy: "native-suppressed",
    ui_action_profile: "native-three-turn-switch",
    ui_turn_count: 3,
    api_action_profile: "full-agentic-native-cache",
    native_cache_policy: "typed-composite",
    model: "native-model",
    model_bundle_path: nativeBundlePath,
    bundle_fingerprint_sha256: "2".repeat(64),
  });
  const intent: Record<string, unknown> = {
    schema: "vmlx-r18-owned-run-intent-v5",
    run_id: "run",
    nonce: "nonce",
    source_commit: "3".repeat(40),
    source_tree: "4".repeat(40),
    harnesses,
    direct_base_url: "http://127.0.0.1:8001",
    native_direct_base_url: "http://127.0.0.1:8002",
    gateway_base_url: "http://127.0.0.1:8088",
    direct_health_url: "http://127.0.0.1:8001/health",
    native_direct_health_url: "http://127.0.0.1:8002/health",
    gateway_health_url: "http://127.0.0.1:8088/health",
    l2_size_eviction_requirements: {
      disk_bytes_within_saved_limit: true,
      older_unused_prefix_eviction_required: true,
      recent_target_survival_required: true,
      restart_restore_required: true,
      counter_only_evidence_allowed: false,
    },
    created_at: new Date().toISOString(),
    phase_plan: phasePlan,
  };
  intent.canonical_sha256 = canonicalHash(intent);
  return intent;
}

function streamEvent(
  sequence: number,
  channel: "reasoning" | "content",
  fullContent: string,
  delta: string,
) {
  return {
    sequence,
    event: "stream",
    channel,
    delta,
    cumulativeReset: false,
    payload: {
      fullContent,
      isReasoning: channel === "reasoning",
    },
  };
}

function primaryTrace(messageId: string, finalContent: string, toolIndex?: number) {
  const events: Array<Record<string, unknown>> = [
    streamEvent(1, "reasoning", "Reason", "Reason"),
    streamEvent(2, "reasoning", "Reason carefully", " carefully"),
  ];
  let sequence = 3;
  if (toolIndex != null) {
    const callId = `call-${toolIndex + 1}`;
    events.push(
      {
        sequence: sequence++,
        event: "tool",
        channel: "tool",
        payload: { toolCallId: callId, phase: "calling" },
      },
      {
        sequence: sequence++,
        event: "tool",
        channel: "tool",
        payload: { toolCallId: callId, phase: "result" },
      },
    );
  }
  const prefix = finalContent.slice(0, Math.max(1, Math.floor(finalContent.length / 2)));
  events.push(
    streamEvent(sequence++, "content", prefix, prefix),
    streamEvent(
      sequence++,
      "content",
      finalContent,
      finalContent.slice(prefix.length),
    ),
    {
      sequence,
      event: "terminal",
      channel: "terminal",
      payload: { messageId },
    },
  );
  return { messageId, events };
}

function cacheStats({
  processed,
  hitRequests,
  hitTokens,
  partialTokens,
  skippedTokens,
}: {
  processed: number;
  hitRequests: number;
  hitTokens: number;
  partialTokens: number;
  skippedTokens: number;
}) {
  return {
    scheduler_stats: {
      num_requests_processed: processed,
      cache_hit_requests: hitRequests,
      cache_hit_tokens: hitTokens,
      cache_reuse_partial_tokens: partialTokens,
      cache_reuse_skip_tokens: skippedTokens,
    },
  };
}

function goodResult(): Record<string, any> {
  const sourceCommit = "c".repeat(40);
  const modules = localRendererModuleEvidence();
  const records = assistantIds.map((id, index) => ({
    id,
    content: persistedContents[index],
  }));
  const messages = records.map((record, index) => ({
    messageId: record.id,
    role: "assistant",
    visible: true,
    answerText: renderedContents[index],
    reasoningText: "Reason carefully",
    html:
      index === 2
        ? '$43 and <span class="katex">47 × 19 = 893 &lt; 920 = 46 × 20</span>'
        : record.content,
    katexCount: index === 2 ? 1 : 0,
    katexErrorCount: 0,
    katexAnnotations:
      index === 2
        ? ["47 \\times 19 = 893 < 920 = 46 \\times 20"]
        : [],
    currencyOccurrences:
      index === 2 ? [{ text: "$43", insideKatex: false }] : [],
    toolCards:
      index < 2
        ? [
            {
              callId: `call-${index + 1}`,
              name: "run_command",
              phase: "result",
              visible: true,
            },
          ]
        : [],
  }));
  const samples = assistantIds.flatMap((messageId, index) => [
    {
      messageId,
      answerText: renderedContents[index].slice(
        0,
        Math.max(1, Math.floor(renderedContents[index].length / 2)),
      ),
      reasoningText: "Reason",
    },
    {
      messageId,
      answerText: renderedContents[index],
      reasoningText: "Reason carefully",
    },
  ]);
  const healthRaw = {
    status: "healthy",
    model_loaded: true,
    model_name: "/private/models/test-model",
    model_bundle_provenance: modelBundleAttestation,
    runtime_provenance: {
      model_bundle_provenance: modelBundleAttestation,
    },
  };
  const runtimeBinding = {
    backend_pid: 4321,
    pid: 4321,
    fingerprint_sha256: sha,
    runtime_source_hashes: {
      server_module_sha256: sha,
      package_init_sha256: sha,
      python_source_tree_sha256: sha,
      python_executable_fingerprint_sha256: testExecutablePathSha,
    },
    python_source_file_count: 10,
    python_source_read_error_count: 0,
    model_name: "/private/models/test-model",
    model_bundle_fingerprint_sha256: sha,
    cache_topology_fingerprint_sha256: sha,
  };
  const result: Record<string, any> = {
    format: "vmlx-electron-ui-proof-v2",
    run_id: "ui-run-1",
    status: "pass",
    surfaceStatus: "partial_ui_only",
    modelPath: "/private/models/test-model",
    servedModel: "test-model",
    baseUrl: "http://127.0.0.1:8000",
    requestedWireApi: "chat",
    requestedEnableThinking: undefined,
    requestedBuiltinTools: true,
    requestedServerCacheControls: true,
    assistantMessageIds: assistantIds,
    assistantRecords: records,
    persistedReasoningByMessage: assistantIds.map(() => [
      "Reason carefully",
    ]),
    messageEventTrace: assistantIds.map((messageId, index) =>
      primaryTrace(
        messageId,
        persistedContents[index],
        index < 2 ? index : undefined,
      ),
    ),
    renderedDom: {
      messages,
      samples,
      rawI18nKeys: [],
      visibleErrors: [],
      transientAlerts: [],
    },
    requestContract: {
      promptOne: prompts[0],
      promptTwo: prompts[1],
      promptThree: prompts[2],
      requestMaxTokens: undefined,
      maxToolIterations: 4,
      toolResultMaxChars: 12500,
      samplingOverrides: {},
    },
    persistedToolsByMessage: [
      [
        {
          phase: "calling",
          toolCallId: "call-1",
          toolName: "run_command",
        },
        {
          phase: "result",
          toolCallId: "call-1",
          toolName: "run_command",
          detail: "REAL_UI_LIVE_TOOL_ONE",
        },
      ],
      [
        {
          phase: "calling",
          toolCallId: "call-2",
          toolName: "run_command",
        },
        {
          phase: "result",
          toolCallId: "call-2",
          toolName: "run_command",
          detail: "REAL_UI_LIVE_TOOL_TWO",
        },
      ],
      [],
    ],
    persistedOaiCallsByMessage: [
      [
        {
          id: "call-1",
          function: {
            name: "run_command",
            arguments: JSON.stringify({
              command:
                "printf REAL_UI_LIVE_TOOL_ONE > real_ui_tool_probe_1.txt",
            }),
          },
        },
      ],
      [
        {
          id: "call-2",
          function: {
            name: "run_command",
            arguments: JSON.stringify({
              command:
                "test -f real_ui_tool_probe_1.txt && printf REAL_UI_LIVE_TOOL_TWO > real_ui_tool_probe_2.txt",
            }),
          },
        },
      ],
      [],
    ],
    persistedOaiResultsByMessage: [
      [{ tool_call_id: "call-1", output: "REAL_UI_LIVE_TOOL_ONE" }],
      [{ tool_call_id: "call-2", output: "REAL_UI_LIVE_TOOL_TWO" }],
      [],
    ],
    toolProbeFiles: {
      "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
      "real_ui_tool_probe_2.txt": "REAL_UI_LIVE_TOOL_TWO",
    },
    toolProbeCleanup: {
      removed: true,
      paths: [
        "/private/work/real_ui_tool_probe_1.txt",
        "/private/work/real_ui_tool_probe_2.txt",
      ],
    },
    bundleGenerationContract: {
      bundle_path: "/private/models/test-model",
      defaults,
      health_attestation: modelBundleAttestation,
    },
    rendererGenerationDefaults: defaults,
    chatSettingsDom: {
      values: {
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        minP: 0.05,
        repeatPenalty: 1.1,
      },
      maxTokens: { value: "", placeholder: "2048 (model default)" },
      wireApi: "completions",
      reasoningMode: "Auto",
      builtinToolsEnabled: true,
      workingDirectory: "/private/work",
      maxToolIterations: "4",
      toolResultMaxChars: "12500",
      toolCategories: {
        file: true,
        search: false,
        shell: true,
        webSearch: false,
        urlFetch: false,
        git: false,
        utilities: true,
      },
    },
    chatSettingsInteraction: {
      openedVisibly: true,
      controlsChanged: ["API Wire Format", "Tool Result Limit"],
      savedViaVisibleControl: true,
      reopenedAfterSave: true,
      persistedAfterReopen: true,
    },
    chatOverrides: {
      wireApi: "completions",
      builtinToolsEnabled: true,
      workingDirectory: "/private/work",
      maxToolIterations: 4,
      toolResultMaxChars: 12500,
    },
    workingDirectory: "/private/work",
    uiTurnEvidence: assistantIds.map((assistantMessageId, index) => ({
      turn: index + 1,
      prompt: prompts[index],
      proofRequestId: `user-${index + 1}`,
      terminalProofRequestId: `user-${index + 1}`,
      requestIds: [`wire-request-${index + 1}`],
      userMessageId: `user-${index + 1}`,
      assistantMessageId,
      terminalMessageId: assistantMessageId,
      terminalResponseId: `server-request-${index + 1}`,
      logMatchMode: "exact_identity_ring_safe",
    })),
    resolvedSamplingKwargs: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      min_p: 0.05,
      repetition_penalty: 1.1,
      max_tokens: 2048,
    },
    resolvedSamplingRecords: assistantIds.map((assistantMessageId, index) => ({
      route_model:
        "Resolved sampling kwargs route=/v1/chat/completions model=/private/models/test-model",
      route: "/v1/chat/completions",
      model: "/private/models/test-model",
      turn: index + 1,
      proof_request_id: `user-${index + 1}`,
      request_id: `wire-request-${index + 1}`,
      message_id: assistantMessageId,
      correlation_source: "server_emitted",
      user_message_id: `user-${index + 1}`,
      assistant_message_id: assistantMessageId,
      values: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.05,
        repetition_penalty: 1.1,
        max_tokens: 2048,
      },
    })),
    requestCorrelation: {
      status: "verified",
      turns: assistantIds.map((assistantMessageId, index) => ({
        turn: index + 1,
        proofRequestId: `user-${index + 1}`,
        userMessageId: `user-${index + 1}`,
        assistantMessageId,
        serverProofRequestId: `user-${index + 1}`,
        serverRequestIds: [`wire-request-${index + 1}`],
        serverMessageId: assistantMessageId,
        resolvedLogCorrelated: true,
        cacheObservationCorrelated: true,
      })),
    },
    serverCacheControls: {
      runningSessionDrawer: true,
      visibleBlockDiskChecked: true,
      initialCacheControls: {
        enablePrefixCache: true,
        usePagedCache: true,
        enableBlockDiskCache: true,
      },
      argv: ["--use-paged-cache", "--enable-block-disk-cache"],
    },
    session: {
      effective_config: {
        enablePrefixCache: true,
        usePagedCache: true,
        enableBlockDiskCache: true,
      },
    },
    cacheRequestEvidence: [
      {
        turn: 1,
        proofRequestId: "user-1",
        terminalResponseId: "server-request-1",
        serverRequestId: "server-request-1",
        executionRequestId: "server-request-1",
        correlationStatus: "verified",
        userMessageId: "user-1",
        assistantMessageId: assistantIds[0],
        serverObservation: {
          proof_request_id: "user-1",
          request_id: "server-request-1",
          terminal_response_id: "server-request-1",
          message_id: assistantIds[0],
          correlation_source:
            "chat_complete_response_id_to_scheduler_last_cache_execution",
          prompt_tokens: 128,
          cached_tokens: 0,
          prefill_tokens: 128,
          cache_hit_request: 0,
          cache_hit_tokens: 0,
          cache_reuse_partial_tokens: 0,
          cache_reuse_skip_tokens: 0,
        },
        before: cacheStats({
          processed: 0,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
        after: cacheStats({
          processed: 1,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
      },
      {
        turn: 2,
        proofRequestId: "user-2",
        terminalResponseId: "server-request-2",
        serverRequestId: "server-request-2",
        executionRequestId: "server-request-2",
        correlationStatus: "verified",
        userMessageId: "user-2",
        assistantMessageId: assistantIds[1],
        serverObservation: {
          proof_request_id: "user-2",
          request_id: "server-request-2",
          terminal_response_id: "server-request-2",
          message_id: assistantIds[1],
          correlation_source:
            "chat_complete_response_id_to_scheduler_last_cache_execution",
          cache_reuse_applied: true,
          prompt_tokens: 192,
          cached_tokens: 128,
          prefill_tokens: 64,
          cache_hit_request: 1,
          cache_hit_tokens: 128,
          cache_reuse_partial_tokens: 64,
          cache_reuse_skip_tokens: 64,
        },
        before: cacheStats({
          processed: 1,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
        after: cacheStats({
          processed: 2,
          hitRequests: 1,
          hitTokens: 128,
          partialTokens: 64,
          skippedTokens: 64,
        }),
      },
      {
        turn: 3,
        proofRequestId: "user-3",
        terminalResponseId: "server-request-3",
        serverRequestId: "server-request-3",
        executionRequestId: "server-request-3",
        correlationStatus: "verified",
        userMessageId: "user-3",
        assistantMessageId: assistantIds[2],
        serverObservation: {
          proof_request_id: "user-3",
          request_id: "server-request-3",
          terminal_response_id: "server-request-3",
          message_id: assistantIds[2],
          correlation_source:
            "chat_complete_response_id_to_scheduler_last_cache_execution",
          cache_reuse_applied: true,
          prompt_tokens: 192,
          cached_tokens: 128,
          prefill_tokens: 64,
          cache_hit_request: 1,
          cache_hit_tokens: 128,
          cache_reuse_partial_tokens: 64,
          cache_reuse_skip_tokens: 64,
        },
        before: cacheStats({
          processed: 2,
          hitRequests: 1,
          hitTokens: 128,
          partialTokens: 64,
          skippedTokens: 64,
        }),
        after: cacheStats({
          processed: 3,
          hitRequests: 2,
          hitTokens: 256,
          partialTokens: 128,
          skippedTokens: 128,
        }),
      },
    ],
    server: {
      models: { data: [{ id: "test-model" }] },
      health: {
        ...healthRaw,
        effective_defaults: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          min_p: 0.05,
          repetition_penalty: 1.1,
          max_output_tokens: 2048,
        },
        native_cache: {
          prefix: true,
          paged: true,
          block_disk_only: false,
          block_disk_l2: true,
        },
      },
    },
    uiRuntimeProvenance: {
      mode: "electron-dev",
      renderer_source_tree_sha256: sha,
      renderer_build_source_commit: sourceCommit,
      electron_executable: testExecutablePath,
      electron_executable_sha256: testExecutableSha,
      source_commit: sourceCommit,
      source_tree: "tree",
      vite_renderer_source_seen: true,
      vite_client_seen: true,
      cdp_process_binding: {
        launched_root_pid: 1200,
        process_tree_pids: [1200, 1234],
        listener_pid: 1234,
        belongs_to_launched_process_tree: true,
        executable_path: testExecutablePath,
        executable_sha256: testExecutableSha,
        executable_path_fingerprint_sha256: testExecutablePathSha,
      },
      backend_python_process_binding: {
        listener_pid: 4321,
        health_pid: 4321,
        executable_path: testExecutablePath,
        executable_sha256: testExecutableSha,
        executable_path_fingerprint_sha256: testExecutablePathSha,
      },
      served_renderer_modules: modules,
      served_renderer_source_sha256: canonicalHash(modules),
    },
    gitProvenance: {
      before: {
        renderer_source_tree_sha256: sha,
        commit: sourceCommit,
        tree: "tree",
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
      after: {
        renderer_source_tree_sha256: sha,
        commit: sourceCommit,
        tree: "tree",
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
    },
    healthProvenance: {
      before: { raw: structuredClone(healthRaw), binding: runtimeBinding },
      after: { raw: structuredClone(healthRaw), binding: runtimeBinding },
    },
  };
  return result;
}

function sse(...objects: Array<Record<string, unknown> | "[DONE]">) {
  return objects
    .map((object) =>
      object === "[DONE]"
        ? "data: [DONE]\n\n"
        : `data: ${JSON.stringify(object)}\n\n`,
    )
    .join("");
}

const pairedToolParameters: Record<string, Record<string, unknown>> = {
  file_info: {
    type: "object",
    properties: { path: { type: "string" } },
    required: ["path"],
    additionalProperties: false,
  },
  run_command: {
    type: "object",
    properties: { command: { type: "string" } },
    required: ["command"],
    additionalProperties: false,
  },
};
const pairedCaptureSemantics = [
  "Exact decompressed response-body bytes delivered to protocol parsers: ",
  "streaming bytes before requests.iter_lines line splitting or Unicode ",
  "decoding, and Responses nonstream bytes before JSON decoding; excludes ",
  "HTTP transfer framing and compressed transport octets.",
].join("");

function expectedTerminals(protocol: string, mode: string, round: number) {
  const toolRound = round < 3;
  if (protocol === "chat") {
    return [toolRound ? "tool_calls" : "stop", ...(mode === "stream" ? ["DONE"] : [])];
  }
  if (protocol === "responses") return ["response.completed"];
  if (protocol === "anthropic") {
    return [toolRound ? "tool_use" : "end_turn", ...(mode === "stream" ? ["message_stop"] : [])];
  }
  return [toolRound ? "tool_calls" : "stop"];
}

function toolContracts(protocol: string, stage: number) {
  const names = protocol === "ollama"
    ? stage === 1
      ? ["file_info"]
      : stage === 2
        ? ["run_command"]
        : []
    : ["file_info", "run_command"];
  return names.map((name) => ({
    name,
    parameters: structuredClone(pairedToolParameters[name]),
  }));
}

function fixtureToolChoice(protocol: string, mode: string, stage: number) {
  if (protocol === "ollama") return null;
  if (stage === 3) {
    return protocol === "anthropic" ? { type: "none" } : "none";
  }
  const name = stage === 1 ? "file_info" : "run_command";
  if (stage === 1 && mode !== "stream") {
    return protocol === "anthropic" ? { type: "any" } : "required";
  }
  if (protocol === "chat") {
    return { type: "function", function: { name } };
  }
  if (protocol === "responses") {
    return { type: "function", name };
  }
  return { type: "tool", name };
}

function rawRound(
  protocol: string,
  round: number,
  callId: string,
  expectedFinal: string,
) {
  const reasoning = round < 3 ? `Reason ${protocol} ${round} A.B.` : "";
  const content = round === 3 ? expectedFinal : "";
  const toolName = round === 1 ? "file_info" : "run_command";
  const args = round === 1
    ? { path: "panel/package.json" }
    : { command: "pwd" };
  if (protocol === "chat") {
    return sse(
      ...(round < 3
        ? [
            { choices: [{ delta: { reasoning_content: `Reason ${protocol} ${round} A.` } }] },
            { choices: [{ delta: { reasoning_content: "B." } }] },
            {
              choices: [{
                delta: {
                  tool_calls: [{
                    index: 0,
                    id: callId,
                    function: { name: toolName, arguments: JSON.stringify(args) },
                  }],
                },
              }],
            },
          ]
        : [
            { choices: [{ delta: { content: content.slice(0, 10) } }] },
            { choices: [{ delta: { content: content.slice(10) } }] },
          ]),
      { choices: [{ delta: {}, finish_reason: round < 3 ? "tool_calls" : "stop" }] },
      "[DONE]",
    );
  }
  if (protocol === "responses") {
    return sse(
      ...(round < 3
        ? [
            { type: "response.reasoning_summary_text.delta", delta: `Reason ${protocol} ${round} A.` },
            { type: "response.reasoning_summary_text.delta", delta: "B." },
            {
              type: "response.output_item.added",
              output_index: 0,
              item: {
                type: "function_call",
                id: `item-${callId}`,
                call_id: callId,
                name: toolName,
                arguments: "",
              },
            },
            {
              type: "response.function_call_arguments.delta",
              item_id: `item-${callId}`,
              call_id: callId,
              delta: JSON.stringify(args),
            },
            {
              type: "response.function_call_arguments.done",
              item_id: `item-${callId}`,
              call_id: callId,
              name: toolName,
              arguments: JSON.stringify(args),
            },
          ]
        : [
            { type: "response.output_text.delta", delta: content.slice(0, 10) },
            { type: "response.output_text.delta", delta: content.slice(10) },
          ]),
      { type: "response.completed", response: { status: "completed" } },
    );
  }
  if (protocol === "anthropic") {
    return sse(
      ...(round < 3
        ? [
            { type: "content_block_delta", delta: { type: "thinking_delta", thinking: `Reason ${protocol} ${round} A.` } },
            { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "B." } },
            {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "tool_use",
                id: callId,
                name: toolName,
                input: args,
              },
            },
          ]
        : [
            { type: "content_block_delta", delta: { type: "text_delta", text: content.slice(0, 10) } },
            { type: "content_block_delta", delta: { type: "text_delta", text: content.slice(10) } },
          ]),
      { type: "message_delta", delta: { stop_reason: round < 3 ? "tool_use" : "end_turn" } },
      { type: "message_stop" },
    );
  }
  const rows = round < 3
    ? [
        { message: { thinking: `Reason ${protocol} ${round} A.` } },
        { message: { thinking: "B." } },
        {
          message: {
            tool_calls: [{
              id: callId,
              function: { name: toolName, arguments: args },
            }],
          },
        },
      ]
    : [
        { message: { content: content.slice(0, 10) } },
        { message: { content: content.slice(10) } },
      ];
  rows.push({ done: true, done_reason: round < 3 ? "tool_calls" : "stop" } as any);
  return `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`;
}

function rawResponsesNonstream(
  round: number,
  callId: string,
  expectedFinal: string,
) {
  const output = round < 3
    ? [
        {
          type: "reasoning",
          summary: [
            { type: "summary_text", text: `Reason responses ${round} A.` },
            { type: "summary_text", text: "B." },
          ],
        },
        {
          type: "function_call",
          id: `item-${callId}`,
          call_id: callId,
          name: round === 1 ? "file_info" : "run_command",
          arguments: JSON.stringify(
            round === 1
              ? { path: "panel/package.json" }
              : { command: "pwd" },
          ),
        },
      ]
    : [
        {
          type: "message",
          content: [{ type: "output_text", text: expectedFinal }],
        },
      ];
  return JSON.stringify({
    id: `response-nonstream-${round}`,
    status: "completed",
    output,
  });
}

function createValidPairedArtifact(result: Record<string, any>) {
  const binding = result.healthProvenance.after.binding;
  const directory = mkdtempSync(path.join(tmpdir(), "vmlx-paired-api-v2-"));
  const repoRoot = realpathSync(path.resolve(process.cwd(), ".."));
  const runDirectory = path.join(directory, result.run_id);
  mkdirSync(runDirectory, { mode: 0o700 });
  const protocolNames = ["chat", "responses", "anthropic", "ollama"];
  const modeNames = ["stream", "nonstream"];
  const flows: Record<string, any> = { direct: {}, gateway: {} };
  const manifestRoutes: Array<Record<string, any>> = [];
  let sequence = 0;
  for (const baseLabel of ["direct", "gateway"]) {
    for (const protocol of protocolNames) {
      flows[baseLabel][protocol] = {};
      for (const mode of modeNames) {
        const executionOutputs = [
          JSON.stringify({
            path: "panel/package.json",
            size_human: "5.2 KB",
          }),
          JSON.stringify({ stdout: repoRoot }),
        ];
        const executions = [
          {
            name: "file_info",
            call_id: `${baseLabel}-${protocol}-${mode}-file`,
            arguments: { path: "panel/package.json" },
            result: { path: "panel/package.json", size_human: "5.2 KB" },
            output_chars: executionOutputs[0].length,
            output_sha256: crypto.createHash("sha256").update(executionOutputs[0]).digest("hex"),
          },
          {
            name: "run_command",
            call_id: `${baseLabel}-${protocol}-${mode}-pwd`,
            arguments: { command: "pwd" },
            result: { stdout: repoRoot },
            output_chars: executionOutputs[1].length,
            output_sha256: crypto.createHash("sha256").update(executionOutputs[1]).digest("hex"),
          },
        ];
        const expectedFinal =
          `AGENTIC-${protocol.toUpperCase()}-${mode.toUpperCase()}-DONE SIZE=5.2 KB PWD=${repoRoot}`;
        const requests = [1, 2, 3].map((stage) => {
          const bodyIdentity = { baseLabel, protocol, mode, stage };
          const canonicalBodyIdentity = { protocol, mode, stage };
          const request: Record<string, any> = {
            stage,
            body_chars: canonicalJson(bodyIdentity).length,
            body_sha256: canonicalHash(bodyIdentity),
            canonical_body_sha256: canonicalHash(canonicalBodyIdentity),
            tool_choice: fixtureToolChoice(protocol, mode, stage),
            stream: mode === "stream",
            enable_thinking: true,
            previous_response_id: (
              protocol === "responses" && stage > 1
                ? `${baseLabel}-${protocol}-${mode}-${stage - 1}`
                : null
            ),
            max_output_tokens: 512,
            tool_contracts: toolContracts(protocol, stage),
            tool_history_linkage: [],
          };
          if (stage >= 2 && protocol === "responses") {
            const prior = executions[stage - 2];
            request.tool_history_linkage.push({
              kind: "tool_result",
              role: "function_call_output",
              call_id: prior.call_id,
              output_chars: prior.output_chars,
              output_sha256: prior.output_sha256,
            });
          } else if (stage >= 2) {
            for (const prior of executions.slice(0, stage - 1)) {
              request.tool_history_linkage.push(
                {
                  kind: "assistant_tool_call",
                  role: "assistant",
                  call_id: protocol === "ollama" ? "" : prior.call_id,
                  name: prior.name,
                },
                {
                  kind: "tool_result",
                  role: protocol === "anthropic" ? "user" : "tool",
                  call_id: protocol === "ollama" ? "" : prior.call_id,
                  ...(protocol === "ollama" ? { name: prior.name } : {}),
                  output_chars: prior.output_chars,
                  output_sha256: prior.output_sha256,
                },
              );
            }
          }
          return request;
        });
        const rounds = [1, 2, 3].map((roundNumber) => {
          const reasoning = roundNumber < 3 ? `Reason ${protocol} ${roundNumber} A.B.` : "";
          const content = roundNumber === 3 ? expectedFinal : "";
          const terminals = expectedTerminals(protocol, mode, roundNumber);
          const events: Array<Record<string, any>> = [];
          let at = 1;
          if (roundNumber < 3) {
            for (const part of [`Reason ${protocol} ${roundNumber} A.`, "B."]) {
              events.push({
                at_ms: at++,
                channel: "reasoning",
                kind: `${protocol}.reasoning.delta`,
                chars: part.length,
                sha256: crypto.createHash("sha256").update(part).digest("hex"),
              });
            }
            events.push({
              at_ms: at++,
              channel: "tool",
              kind: `${protocol}.tool`,
              call_id: executions[roundNumber - 1].call_id,
            });
          } else {
            for (const part of [content.slice(0, 10), content.slice(10)]) {
              events.push({
                at_ms: at++,
                channel: "content",
                kind: `${protocol}.content.delta`,
                chars: part.length,
                sha256: crypto.createHash("sha256").update(part).digest("hex"),
              });
            }
          }
          for (const terminal of terminals) {
            events.push({ at_ms: at++, channel: "terminal", kind: terminal });
          }
          return {
            status_code: 200,
            elapsed_ms: 10,
            response_id: `${baseLabel}-${protocol}-${mode}-${roundNumber}`,
            reasoning_chars: reasoning.length,
            reasoning_sha256: crypto.createHash("sha256").update(reasoning).digest("hex"),
            reasoning_delta_count: roundNumber < 3 ? (mode === "stream" ? 2 : 1) : 0,
            content,
            content_chars: content.length,
            content_sha256: crypto.createHash("sha256").update(content).digest("hex"),
            content_delta_count: roundNumber === 3 ? (mode === "stream" ? 2 : 1) : 0,
            tool_calls: roundNumber < 3
              ? [{
                  index: 0,
                  id: executions[roundNumber - 1].call_id,
                  name: executions[roundNumber - 1].name,
                  arguments: executions[roundNumber - 1].arguments,
                }]
              : [],
            terminals,
            errors: [],
            events,
          };
        });
        flows[baseLabel][protocol][mode] = {
          pass: true,
          checks: { exact: true },
          expected_final: expectedFinal,
          requests,
          rounds,
          executions,
          terminal_classification: rounds.map((round) => ({
            pass: true,
            values: round.terminals,
            semantic: round.terminals.filter((value: string) =>
              !["DONE", "message_stop"].includes(value)),
          })),
        };
        if (mode !== "stream" && !(mode === "nonstream" && protocol === "responses")) {
          continue;
        }
        for (let roundNumber = 1; roundNumber <= 3; roundNumber += 1) {
          sequence += 1;
          const captureLabel = `${mode}-flow-round${roundNumber}`;
          const body = mode === "stream"
            ? rawRound(
                protocol,
                roundNumber,
                executions[roundNumber - 1]?.call_id || "",
                expectedFinal,
              )
            : rawResponsesNonstream(
                roundNumber,
                executions[roundNumber - 1]?.call_id || "",
                expectedFinal,
              );
          const stem = `${String(sequence).padStart(4, "0")}-${baseLabel}-${protocol}-${captureLabel}`;
          const bodyFile = `${stem}.decompressed-parser-input.bin`;
          const metadataFile = `${stem}.metadata.json`;
          const bodyPath = path.join(runDirectory, bodyFile);
          const metadataPath = path.join(runDirectory, metadataFile);
          writePrivateArtifactFile(bodyPath, body);
          const bodySha = crypto.createHash("sha256").update(body).digest("hex");
          const requestPayload = { ...requests[roundNumber - 1] };
          delete requestPayload.stage;
          const requestBodySha = canonicalHash({
            baseLabel,
            protocol,
            roundNumber,
          });
          const metadata = {
            schema_version: 1,
            capture_layer: "requests.decompressed_response_parser_input",
            capture_semantics: pairedCaptureSemantics,
            base_label: baseLabel,
            protocol,
            capture_label: captureLabel,
            request: {
              method: "POST",
              url: `${baseLabel === "direct" ? "http://127.0.0.1:8000" : "http://127.0.0.1:8080"}${{
                chat: "/v1/chat/completions",
                responses: "/v1/responses",
                anthropic: "/v1/messages",
                ollama: "/api/chat",
              }[protocol]}`,
              body_bytes: 1,
              body_sha256: requestBodySha,
              prepared_payload_body_sha256: requestPayload.body_sha256,
              prepared_payload_canonical_body_sha256:
                requestPayload.canonical_body_sha256,
              headers: [],
              payload: {
                ...requestPayload,
                model: result.servedModel,
                top_level_fields: ["model"],
              },
            },
            response: {
              status_code: 200,
              headers: [],
              body_file: bodyFile,
              body_bytes: Buffer.byteLength(body),
              body_sha256: bodySha,
            },
          };
          writePrivateArtifactFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
          const metadataBytes = readFileSync(metadataPath);
          manifestRoutes.push({
            base_label: baseLabel,
            protocol,
            capture_label: captureLabel,
            expected: 1,
            started: 1,
            finished: 1,
            errors: [],
            artifacts: [{
              sequence,
              body_file: bodyFile,
              metadata_file: metadataFile,
              verified: true,
              body_bytes: Buffer.byteLength(body),
              body_sha256: bodySha,
              metadata_sha256: crypto.createHash("sha256").update(metadataBytes).digest("hex"),
              request_body_sha256: requestBodySha,
              prepared_payload_body_sha256: requestPayload.body_sha256,
              prepared_payload_canonical_body_sha256:
                requestPayload.canonical_body_sha256,
            }],
          });
        }
      }
    }
  }
  const manifest = {
    schema_version: 1,
    enabled: true,
    capture_layer: "requests.decompressed_response_parser_input",
    capture_semantics: pairedCaptureSemantics,
    run_id: result.run_id,
    expected: manifestRoutes.length,
    started: manifestRoutes.length,
    finished: manifestRoutes.length,
    errors: 0,
    complete: true,
    routes: manifestRoutes,
  };
  const manifestPath = path.join(runDirectory, "manifest.json");
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writePrivateArtifactFile(manifestPath, manifestText);
  const producerSource = realpathSync(
    path.join(repoRoot, "tests/cross_matrix/run_agentic_protocol_matrix.py"),
  );
  const producerExecutable = realpathSync(process.execPath);
  const producerExecutableBytes = readFileSync(producerExecutable);
  const producerHarnessBytes = readFileSync(producerSource);
  const source = {
    git_root: repoRoot,
    head: result.gitProvenance.after.commit,
    tree: result.gitProvenance.after.tree,
    clean: true,
    status_sha256: sha,
    server_module_sha256: result.gitProvenance.after.server_module_sha256,
    package_init_sha256: result.gitProvenance.after.package_init_sha256,
    python_source_tree_sha256: result.gitProvenance.after.python_source_tree_sha256,
    python_source_file_count: result.gitProvenance.after.python_source_file_count,
    python_source_read_error_count: result.gitProvenance.after.python_source_read_error_count,
  };
  const runner = {
    repo_venv: true,
    repo_python: true,
    python_executable_path: producerExecutable,
    python_executable_fingerprint_sha256:
      binding.runtime_source_hashes.python_executable_fingerprint_sha256,
    python_prefix_path: testExecutablePrefixPath,
    python_prefix_fingerprint_sha256: testExecutablePrefixPathSha,
    producer_pid: 987654,
    producer_executable_path: producerExecutable,
    producer_executable_sha256: crypto.createHash("sha256").update(producerExecutableBytes).digest("hex"),
    producer_executable_size_bytes: producerExecutableBytes.length,
    producer_harness_relative_path: "tests/cross_matrix/run_agentic_protocol_matrix.py",
    producer_harness_path: producerSource,
    producer_harness_sha256: crypto.createHash("sha256").update(producerHarnessBytes).digest("hex"),
    producer_harness_size_bytes: producerHarnessBytes.length,
  };
  const bundle = {
    ...result.bundleGenerationContract.health_attestation,
    model_name: result.servedModel,
  };
  const healthFull = result.healthProvenance.after.raw;
  const healthRow = {
    url: "http://127.0.0.1:8000/health",
    full: healthFull,
    full_sha256: canonicalHash(healthFull),
    identity: binding,
  };
  const rawCapture = {
    ...manifest,
    manifest_file: "manifest.json",
    manifest_path: manifestPath,
    run_directory: runDirectory,
    manifest_sha256: crypto.createHash("sha256").update(manifestText).digest("hex"),
  };
  const value = {
    schema: "vmlx-agentic-protocol-matrix-v2",
    schema_version: 2,
    run_id: result.run_id,
    requested_model: result.servedModel,
    repo_root: repoRoot,
    bases: {
      direct: "http://127.0.0.1:8000",
      gateway: "http://127.0.0.1:8080",
    },
    protocols: protocolNames,
    modes: modeNames,
    second_tool_choice: "explicit",
    backend_identity_fingerprint_sha256: binding.fingerprint_sha256,
    identity: {
      source: {
        declared_head: source.head,
        before: source,
        after: structuredClone(source),
      },
      runner: {
        before: runner,
        after: structuredClone(runner),
      },
      bundle: {
        before: bundle,
        after: structuredClone(bundle),
      },
      health: {
        direct: {
          before: healthRow,
          after: structuredClone(healthRow),
        },
        gateway: {
          before: {
            ...structuredClone(healthRow),
            url: "http://127.0.0.1:8080/health",
          },
          after: {
            ...structuredClone(healthRow),
            url: "http://127.0.0.1:8080/health",
          },
        },
      },
      failures: [],
    },
    raw_capture: rawCapture,
    flows,
    abort_recovery: {},
    checks: {
      identity_provenance_pass: true,
      all_requested_flows_present: true,
      all_flows_pass: true,
      abort_recovery_skipped: true,
      all_abort_recovery_pass: true,
      raw_capture_complete: true,
    },
    pass: true,
  };
  const artifactPath = path.join(directory, "paired-api-proof.json");
  writePrivateArtifactFile(artifactPath, JSON.stringify(value));
  return {
    directory,
    artifact: readPrivateExternalJson(
      artifactPath,
      "Paired raw API proof artifact fixture",
    ),
  };
}

function writePairedArtifactValue(
  directory: string,
  name: string,
  value: Record<string, unknown>,
) {
  const artifactPath = path.join(directory, name);
  writePrivateArtifactFile(artifactPath, JSON.stringify(value));
  return readPrivateExternalJson(artifactPath, `Paired fixture ${name}`);
}

function rewriteRawCaptureBody(
  value: Record<string, any>,
  {
    baseLabel,
    protocol,
    captureLabel,
    body,
  }: {
    baseLabel: string;
    protocol: string;
    captureLabel: string;
    body: string;
  },
) {
  const rawCapture = value.raw_capture;
  const route = rawCapture.routes.find(
    (row: Record<string, any>) =>
      row.base_label === baseLabel
      && row.protocol === protocol
      && row.capture_label === captureLabel,
  );
  if (!route) throw new Error("raw capture fixture route not found");
  const artifact = route.artifacts[0];
  const bodyPath = path.join(rawCapture.run_directory, artifact.body_file);
  const metadataPath = path.join(
    rawCapture.run_directory,
    artifact.metadata_file,
  );
  writeFileSync(bodyPath, body);
  artifact.body_bytes = Buffer.byteLength(body);
  artifact.body_sha256 = crypto
    .createHash("sha256")
    .update(body)
    .digest("hex");
  const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
  metadata.response.body_bytes = artifact.body_bytes;
  metadata.response.body_sha256 = artifact.body_sha256;
  const metadataText = `${JSON.stringify(metadata, null, 2)}\n`;
  writeFileSync(metadataPath, metadataText);
  artifact.metadata_sha256 = crypto
    .createHash("sha256")
    .update(metadataText)
    .digest("hex");

  const manifest = structuredClone(rawCapture);
  for (const field of [
    "manifest_file",
    "manifest_path",
    "manifest_sha256",
    "run_directory",
  ]) delete manifest[field];
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(rawCapture.manifest_path, manifestText);
  rawCapture.manifest_sha256 = crypto
    .createHash("sha256")
    .update(manifestText)
    .digest("hex");
}

function rewriteRawCaptureMetadata(
  value: Record<string, any>,
  {
    baseLabel,
    protocol,
    captureLabel,
    mutate,
  }: {
    baseLabel: string;
    protocol: string;
    captureLabel: string;
    mutate: (metadata: Record<string, any>, artifact: Record<string, any>) => void;
  },
) {
  const rawCapture = value.raw_capture;
  const route = rawCapture.routes.find(
    (row: Record<string, any>) =>
      row.base_label === baseLabel
      && row.protocol === protocol
      && row.capture_label === captureLabel,
  );
  if (!route) throw new Error("raw capture fixture route not found");
  const artifact = route.artifacts[0];
  const metadataPath = path.join(
    rawCapture.run_directory,
    artifact.metadata_file,
  );
  const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
  mutate(metadata, artifact);
  const metadataText = `${JSON.stringify(metadata, null, 2)}\n`;
  writeFileSync(metadataPath, metadataText);
  artifact.metadata_sha256 = crypto
    .createHash("sha256")
    .update(metadataText)
    .digest("hex");

  const manifest = structuredClone(rawCapture);
  for (const field of [
    "manifest_file",
    "manifest_path",
    "manifest_sha256",
    "run_directory",
  ]) delete manifest[field];
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(rawCapture.manifest_path, manifestText);
  rawCapture.manifest_sha256 = crypto
    .createHash("sha256")
    .update(manifestText)
    .digest("hex");
}

describe("real UI model proof harness", () => {
  it("maps renderer paths to distinct raw Vite filesystem module identities", () => {
    const panelRoot = path.resolve(new URL("..", import.meta.url).pathname);
    expect(
      viteRawRendererModulePath(
        "src/renderer/src/main.tsx",
        "r18 proof/one",
      ),
    ).toBe(
      `/@fs${panelRoot}/src/renderer/src/main.tsx?raw&vmlx_proof=r18%20proof%2Fone`,
    );
    expect(
      viteRawRendererModulePath(
        "src/renderer/src/components/chat/ReasoningBox.tsx",
        "r18-two",
      ),
    ).toBe(
      `/@fs${panelRoot}/src/renderer/src/components/chat/ReasoningBox.tsx?raw&vmlx_proof=r18-two`,
    );
  });

  it("rejects renderer proof paths outside the served renderer root", () => {
    expect(() => viteRawRendererModulePath("src/main.tsx", "r18")).toThrow(
      /outside the Vite renderer root/,
    );
    expect(() =>
      viteRawRendererModulePath("src/renderer/../main.tsx", "r18"),
    ).toThrow(/unsafe served path/);
  });

  it("uses only a validated private token-file path in session launch args", () => {
    const root = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    try {
      const token = `private_${"q".repeat(48)}`;
      const tokenPath = path.join(root, "cache-attestation.token");
      writeFileSync(tokenPath, token, { mode: 0o600 });
      chmodSync(tokenPath, 0o600);
      const args = privateCacheAttestationSessionArgs(tokenPath);
      expect(args).toBe(
        "--enable-private-cache-attestation "
          + `--private-cache-attestation-token-file=${realpathSync(tokenPath)}`,
      );
      expect(args).not.toContain(token);

      chmodSync(tokenPath, 0o640);
      expect(() => privateCacheAttestationSessionArgs(tokenPath)).toThrow(
        /0600|group\/world/,
      );
      chmodSync(tokenPath, 0o600);
      const linkPath = path.join(root, "cache-attestation-link.token");
      symlinkSync(tokenPath, linkPath);
      expect(() => privateCacheAttestationSessionArgs(linkPath)).toThrow(
        /regular, non-symlink/,
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("uses model/run-specific exclusive private artifacts and bundle-owned defaults", () => {
    expect(
      uniqueProofBasename({
        requested: "ui-proof",
        model: "Gemma 4/MoE",
        run: "run 42",
      }),
    ).toBe(
      `ui-proof-Gemma-4-MoE-run-42-${canonicalHash({
        requested: "ui-proof",
        model: "Gemma 4/MoE",
        run: "run 42",
      }).slice(0, 20)}`,
    );
    expect(
      resolveIndependentBundleGenerationDefaults(
        {
          temperature: 0.2,
          top_p: 0.8,
          top_k: 20,
          repetition_penalty: 1.02,
          max_new_tokens: 1024,
        },
        {
          chat: {
            reasoning: { default_mode: "thinking" },
            sampling_defaults: {
              temperature: 0.7,
              top_p: 0.9,
              top_k: 40,
              min_p: 0.05,
              repetition_penalty_thinking: 1.1,
              max_new_tokens: 2048,
            },
          },
        },
        { model_type: "laguna" },
      ),
    ).toEqual(defaults);

    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    try {
      const artifact = path.join(directory, "proof.json");
      writePrivateArtifactFile(artifact, "{}");
      expect(statSync(artifact).mode & 0o777).toBe(0o600);
      expect(readFileSync(artifact, "utf8")).toBe("{}");
      expect(() => writePrivateArtifactFile(artifact, "forged")).toThrow();
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("retains bundle config hashes and reports a generic missing-template warning", () => {
    const bundle = mkdtempSync(path.join(tmpdir(), "vmlx-ui-proof-bundle-"));
    try {
      writeFileSync(
        path.join(bundle, "config.json"),
        JSON.stringify({ model_type: "laguna" }),
      );
      writeFileSync(
        path.join(bundle, "generation_config.json"),
        JSON.stringify({ temperature: 0.7, max_new_tokens: 2048 }),
      );
      writeFileSync(
        path.join(bundle, "tokenizer_config.json"),
        JSON.stringify({
          chat_template: "{% include 'chat_template.jinja' %}",
        }),
      );
      const missing = captureBundleGenerationContract(bundle);
      expect(missing.template.usable).toBe(false);
      expect(missing.template.warning).toMatch(/no usable chat template/i);
      expect(missing.files["config.json"].sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(missing.health_attestation.schema).toBe("vmlx-bundle-config-v1");

      writeFileSync(path.join(bundle, "chat_template.jinja"), "{{ messages }}");
      const usable = captureBundleGenerationContract(bundle);
      expect(usable.template.usable).toBe(true);
      expect(usable.template.warning).toBeNull();
    } finally {
      rmSync(bundle, { recursive: true, force: true });
    }
  });

  it("accepts fully linked UI evidence across all nine hardened contracts", () => {
    const result = goodResult();
    expect(validateRenderedDomEvidence(result)).toEqual([]);
    expect(validateReasoningEvidence(result, "required")).toEqual([]);
    expect(validateExactToolLoopEvidence(result)).toEqual([]);
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
    expect(validateRequestCorrelatedCacheEvidence(result)).toEqual([]);
    expect(validateServerCacheEvidence(result)).toEqual([]);
    expect(validateUiRuntimeProvenance(result)).toEqual([]);
    expect(validateModelBundleBinding(result)).toEqual([]);
    expect(validatePairedApiEvidence(result)).toEqual([]);
  });

  it("binds the terminal response ID to the exact health cache execution", () => {
    const correlated = correlateTerminalResponseToCacheExecution({
      terminal: {
        messageId: assistantIds[0],
        responseId: "chatcmpl-current",
      },
      cacheSnapshot: {
        scheduler: {
          last_cache_execution: {
            request_id: "chatcmpl-current",
            cache_reuse_applied: true,
            prompt_tokens: 192,
            cached_tokens: 128,
            prefill_tokens: 64,
          },
        },
      },
      turn: 1,
      proofRequestId: "ui-run-1:ui:1",
      userMessageId: "user-1",
      assistantMessageId: assistantIds[0],
    });
    expect(correlated.correlationStatus).toBe("verified");
    expect(correlated.serverRequestId).toBe("chatcmpl-current");
    expect(correlated.executionRequestId).toBe("chatcmpl-current");

    const overwritten = correlateTerminalResponseToCacheExecution({
      terminal: {
        messageId: assistantIds[0],
        responseId: "chatcmpl-current",
      },
      cacheSnapshot: {
        scheduler: {
          last_cache_execution: {
            request_id: "chatcmpl-unrelated-later-request",
          },
        },
      },
      turn: 1,
      proofRequestId: "ui-run-1:ui:1",
      userMessageId: "user-1",
      assistantMessageId: assistantIds[0],
    });
    expect(overwritten.correlationStatus).toBe(
      "partial_request_identity_mismatch",
    );

    const missing = correlateTerminalResponseToCacheExecution({
      terminal: { messageId: assistantIds[0] },
      cacheSnapshot: { scheduler: {} },
      turn: 1,
      proofRequestId: "ui-run-1:ui:1",
      userMessageId: "user-1",
      assistantMessageId: assistantIds[0],
    });
    expect(missing.correlationStatus).toBe(
      "partial_product_support_missing",
    );
  });

  it("rejects missing or overwritten per-turn cache request identity", () => {
    const result = goodResult();
    expect(isCacheRequestCorrelationVerified(result)).toBe(true);

    result.cacheRequestEvidence[1].executionRequestId =
      "chatcmpl-unrelated-later-request";
    result.cacheRequestEvidence[1].serverObservation.request_id =
      "chatcmpl-unrelated-later-request";
    expect(isCacheRequestCorrelationVerified(result)).toBe(false);
    expect(validateRequestCorrelatedCacheEvidence(result).join("\n")).toMatch(
      /do not exactly match|lacks exact server-emitted request correlation/,
    );

    result.cacheRequestEvidence[1].terminalResponseId = null;
    result.cacheRequestEvidence[1].serverRequestId = null;
    result.cacheRequestEvidence[1].correlationStatus =
      "partial_product_support_missing";
    expect(isCacheRequestCorrelationVerified(result)).toBe(false);
  });

  it("retains the newest HTTP response ID for continuation cancel and complete", () => {
    const source = readFileSync("src/main/ipc/chat.ts", "utf8");
    const responseCreated = source.slice(
      source.indexOf('responsesEventType === "response.created"'),
      source.indexOf("response.reasoning_summary_text.delta"),
    );
    const chatResponse = source.slice(
      source.indexOf("// Track response ID for server-side cancel"),
      source.indexOf("// Update usage BEFORE emitting delta"),
    );
    expect(responseCreated).toContain("entry.responseId = respId");
    expect(chatResponse).toContain("entry.responseId = parsed.id");
    expect(responseCreated).not.toContain("!entry.responseId");
    expect(chatResponse).not.toContain("!entry.responseId");
    expect(source.match(/responseId: activeRequests\.get\(chatId\)\?\.responseId/g))
      .toHaveLength(2);

    const active = { responseId: undefined as string | undefined };
    for (const responseId of ["chatcmpl-first", "chatcmpl-second"]) {
      active.responseId = responseId;
    }
    expect(active.responseId).toBe("chatcmpl-second");
  });

  it("accepts a one-turn tool restart profile without borrowing later-turn evidence", () => {
    const result = structuredClone(goodResult());
    result.requestContract = {
      ...result.requestContract,
      promptTwo: undefined,
      promptThree: undefined,
      uiActionProfile: "primary-tool-restart-probe",
      uiTurnCount: 1,
      apiActionProfile: "cache-probe",
    };
    result.assistantMessageIds = result.assistantMessageIds.slice(0, 1);
    result.assistantRecords = result.assistantRecords.slice(0, 1);
    result.persistedReasoningByMessage =
      result.persistedReasoningByMessage.slice(0, 1);
    result.persistedToolsByMessage = result.persistedToolsByMessage.slice(0, 1);
    result.persistedOaiCallsByMessage =
      result.persistedOaiCallsByMessage.slice(0, 1);
    result.persistedOaiResultsByMessage =
      result.persistedOaiResultsByMessage.slice(0, 1);
    result.messageEventTrace = result.messageEventTrace.slice(0, 1);
    result.renderedDom.messages = result.renderedDom.messages.slice(0, 1);
    result.renderedDom.samples = result.renderedDom.samples.filter(
      (sample: Record<string, unknown>) =>
        sample.messageId === result.assistantMessageIds[0],
    );
    result.uiTurnEvidence = result.uiTurnEvidence.slice(0, 1);
    result.resolvedSamplingRecords = result.resolvedSamplingRecords.slice(0, 1);
    result.requestCorrelation.turns = result.requestCorrelation.turns.slice(0, 1);
    result.cacheRequestEvidence = result.cacheRequestEvidence.slice(0, 1);
    result.cacheRequestEvidence[0].serverObservation = {
      ...result.cacheRequestEvidence[0].serverObservation,
      cache_reuse_applied: true,
      prompt_tokens: 192,
      cached_tokens: 128,
      prefill_tokens: 64,
      cache_hit_request: 1,
      cache_hit_tokens: 128,
      cache_reuse_partial_tokens: 64,
      cache_reuse_skip_tokens: 64,
    };
    result.cacheRequestEvidence[0].after = cacheStats({
      processed: 1,
      hitRequests: 1,
      hitTokens: 128,
      partialTokens: 64,
      skippedTokens: 64,
    });
    result.toolProbeFiles = {
      "real_ui_tool_probe_1.txt": "REAL_UI_LIVE_TOOL_ONE",
    };
    result.serverCacheControls.initialCacheControls.usePagedCache = false;
    result.serverCacheControls.argv = [
      "--no-paged-cache",
      "--enable-block-disk-cache",
    ];
    result.session.effective_config.usePagedCache = false;
    result.server.health.native_cache.paged = false;
    result.server.health.native_cache.block_disk_only = true;

    expect(validateRenderedDomEvidence(result)).toEqual([]);
    expect(validateReasoningEvidence(result, "required")).toEqual([]);
    expect(validateExactToolLoopEvidence(result)).toEqual([]);
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);
    expect(validateRequestCorrelatedCacheEvidence(result)).toEqual([]);
    expect(validateServerCacheEvidence(result)).toEqual([]);
    expect(isServerRequestCorrelationVerified(result)).toBe(true);
  });

  it("rejects forged CDP process bytes and served renderer source", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance.cdp_process_binding.listener_pid = 0;
    result.uiRuntimeProvenance.served_renderer_modules[0].sha256 = otherSha;
    expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
      /CDP listener PID|served through CDP/,
    );
  });

  it("rejects a renderer build commit that does not match the frozen proof HEAD", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance.renderer_build_source_commit = "d".repeat(40);
    expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
      /build-injected renderer source commit/,
    );
  });

  it("derives renderer provenance from Git and wires local request identity through every wire request", () => {
    const viteConfig = readFileSync(
      path.join(process.cwd(), "electron.vite.config.ts"),
      "utf8",
    );
    const rendererMain = readFileSync(
      path.join(process.cwd(), "src/renderer/src/main.tsx"),
      "utf8",
    );
    const chatSource = readFileSync(
      path.join(process.cwd(), "src/main/ipc/chat.ts"),
      "utf8",
    );
    const serverSource = readFileSync(
      path.join(process.cwd(), "../vmlx_engine/server.py"),
      "utf8",
    );
    expect(viteConfig).toContain("['-C', __dirname, 'rev-parse', 'HEAD']");
    expect(viteConfig).toContain("__VMLINUX_BUILD_SOURCE_COMMIT__");
    expect(rendererMain).toContain(
      "globalThis.__VMLINUX_SOURCE_COMMIT__ = __VMLINUX_BUILD_SOURCE_COMMIT__",
    );
    expect(rendererMain).toContain(
      "document.documentElement.dataset.sourceCommit = __VMLINUX_BUILD_SOURCE_COMMIT__",
    );
    expect(chatSource).toContain("const proofRequestId = userMessage.id;");
    expect(chatSource).toContain('"X-vMLX-Proof-Request-ID": proofRequestId');
    expect(chatSource).toContain('"X-vMLX-Request-ID": requestId');
    expect(chatSource).toContain('"X-vMLX-Message-ID": assistantMessageId');
    expect(chatSource.match(/nextLocalRequestCorrelationHeaders\(\)/g)).toHaveLength(3);
    expect(chatSource.match(/requestIds: \[\.\.\.wireRequestIds\]/g)).toHaveLength(2);
    expect(serverSource).toContain("def _request_header_value(request: Any, name: str)");
    expect(
      serverSource.match(/proof_request_id=_request_header_value\(/g),
    ).toHaveLength(2);
  });

  it("binds installed ASAR, release manifest, bundled Python, runtime, and proof HEAD", () => {
    const result = structuredClone(goodResult());
    result.uiRuntimeProvenance = {
      ...result.uiRuntimeProvenance,
      mode: "installed-app",
      app_asar_sha256: sha,
      external_release_manifest_sha256: sha,
      external_release_manifest: {
        schema: "vmlx-installed-release-manifest-v1",
        source_commit: result.gitProvenance.after.commit,
        source_tree: "tree",
        app_asar_sha256: sha,
        electron_executable_sha256: testExecutableSha,
        bundled_provenance_sha256: sha,
        bundled_python_executable_fingerprint_sha256: testExecutablePathSha,
      },
      bundled_provenance_sha256: sha,
      bundled_provenance: {
        vmlx: { commit: result.gitProvenance.after.commit, version: "1.6.18" },
      },
      bundled_source: {
        server_module_sha256: sha,
        package_init_sha256: sha,
        python_source_tree_sha256: sha,
        python_source_file_count: 10,
        python_source_read_error_count: 0,
      },
    };
    expect(validateUiRuntimeProvenance(result)).toEqual([]);

    result.uiRuntimeProvenance.external_release_manifest.source_commit = "stale";
    result.uiRuntimeProvenance.bundled_source.server_module_sha256 = otherSha;
    expect(validateUiRuntimeProvenance(result).join("\n")).toMatch(
      /release manifest|bundled\/runtime\/source identity/,
    );
  });

  it("rejects a different served model or bundle attestation", () => {
    const result = structuredClone(goodResult());
    result.server.models.data = [{ id: "different-model" }];
    result.healthProvenance.after.raw.model_bundle_provenance = {
      ...modelBundleAttestation,
      aggregate_sha256: otherSha,
    };
    expect(validateModelBundleBinding(result).join("\n")).toMatch(
      /exactly bound|bundle attestation/,
    );
  });

  it("rejects settings not visibly persisted and stale route/model log records", () => {
    const result = structuredClone(goodResult());
    result.chatSettingsInteraction.persistedAfterReopen = false;
    result.resolvedSamplingRecords[1].model = "stale-model";
    result.resolvedSamplingRecords[2].message_id = "wrong-message";
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /visibly changed, saved, reopened, and persisted|route\/model|server-emitted request\/message correlation/,
    );
  });

  it("distinguishes inherited defaults from explicit visible per-chat overrides", () => {
    const result = structuredClone(goodResult());
    result.requestContract.samplingOverrides = { temperature: 0.25 };
    result.requestContract.requestMaxTokens = 512;
    result.chatOverrides.temperature = 0.25;
    result.chatOverrides.maxTokens = 512;
    result.chatSettingsDom.values.temperature = 0.25;
    result.chatSettingsDom.maxTokens.value = "512";
    result.resolvedSamplingKwargs.temperature = 0.25;
    result.resolvedSamplingKwargs.max_tokens = 512;
    for (const record of result.resolvedSamplingRecords) {
      record.values.temperature = 0.25;
      record.values.max_tokens = 512;
    }
    expect(validateGenerationDefaultsEvidence(result)).toEqual([]);

    result.chatOverrides.temperature = 0.7;
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /explicit temperature override=0.25 was not persisted exactly/,
    );
  });

  it("rejects missing visible cache state, contradictory argv, and capacity-only telemetry", () => {
    const result = structuredClone(goodResult());
    delete result.serverCacheControls.initialCacheControls.enablePrefixCache;
    result.serverCacheControls.argv = [
      "--disable-prefix-cache",
      "--no-paged-cache",
      "--enable-block-disk-cache",
    ];
    result.cacheRequestEvidence = result.cacheRequestEvidence.map(
      (row: Record<string, any>, index: number) => ({
        ...row,
        before: cacheStats({
          processed: index,
          hitRequests: 0,
          hitTokens: 0,
          partialTokens: 0,
          skippedTokens: 0,
        }),
        after: {
          ...cacheStats({
            processed: index + 1,
            hitRequests: 0,
            hitTokens: 0,
            partialTokens: 0,
            skippedTokens: 0,
          }),
          max_cache_blocks: 8192,
        },
      }),
    );
    expect(validateServerCacheEvidence(result).join("\n")).toMatch(
      /visible cache control|contradicted enabled prefix cache|contradictory --no-paged-cache|no UI turn had request-correlated/,
    );
  });

  it("rejects cumulative resets, transient parser markers, and final/persisted mismatch", () => {
    const result = structuredClone(goodResult());
    result.messageEventTrace[0].events[1].cumulativeReset = true;
    result.messageEventTrace[1].events[0].delta = "<thi";
    result.messageEventTrace[2].events.at(-2).delta += " different final";
    result.messageEventTrace[2].events.at(-2).payload.fullContentLength =
      result.messageEventTrace[2].events.at(-2).payload.fullContent.length
      + " different final".length;
    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /stream reset or shrank|transient reasoning stream|does not equal persisted final/,
    );
  });

  it.each([
    ["<t", "hink>"],
    ["[T", "HINK]"],
    ["<m", "m:think>"],
  ])("rejects parser markers split across stream boundaries", (first, second) => {
    const result = structuredClone(goodResult());
    const reasoningEvents = result.messageEventTrace[0].events.filter(
      (event: Record<string, unknown>) => event.event === "stream" && event.channel === "reasoning",
    );
    reasoningEvents[0].delta = first;
    reasoningEvents[0].payload.fullContentLength = first.length;
    delete reasoningEvents[0].payload.fullContent;
    reasoningEvents[1].delta = second;
    reasoningEvents[1].payload.fullContentLength = first.length + second.length;
    delete reasoningEvents[1].payload.fullContent;
    result.persistedReasoningByMessage[0] = [first + second];

    expect(validateReasoningEvidence(result, "required").join("\n")).toMatch(
      /parser markers across reasoning stream boundaries/,
    );
  });

  it("accepts compact stream events without retaining cumulative payload text", () => {
    const result = structuredClone(goodResult());
    for (const trace of result.messageEventTrace) {
      for (const event of trace.events) {
        if (event.event !== "stream") continue;
        event.payload.fullContentLength = event.payload.fullContent.length;
        delete event.payload.fullContent;
      }
    }

    expect(validateRenderedDomEvidence(result)).toEqual([]);
    expect(validateReasoningEvidence(result, "required")).toEqual([]);
  });

  it("rejects duplicate/error tool records and missing probe cleanup", () => {
    const result = structuredClone(goodResult());
    result.persistedOaiCallsByMessage[1][0].id = "call-1";
    result.persistedToolsByMessage[1].push({
      phase: "error",
      toolCallId: "call-1",
      toolName: "run_command",
    });
    result.toolProbeCleanup.removed = false;
    expect(validateExactToolLoopEvidence(result).join("\n")).toMatch(
      /IDs are not unique|error status|not cleaned/,
    );
  });

  it("rejects unlinked DOM content, malformed KaTeX/currency, and raw i18n chrome", () => {
    const linkResult = structuredClone(goodResult());
    linkResult.renderedDom.messages[0].answerText = "unrelated";
    expect(validateRenderedDomEvidence(linkResult).join("\n")).toMatch(
      /normalized visible answer is not linked/,
    );

    const renderResult = structuredClone(goodResult());
    renderResult.renderedDom.messages[2].currencyOccurrences[0].insideKatex =
      true;
    renderResult.renderedDom.messages[2].katexAnnotations = [
      "47 \\times 19 = wrong",
    ];
    renderResult.renderedDom.rawI18nKeys = ["layout.chatHistory.groupToday"];
    expect(validateRenderedDomEvidence(renderResult).join("\n")).toMatch(
      /outside KaTeX|does not exactly match|raw i18n keys/,
    );
  });

  it("requires a separate, exactly bound raw API artifact before dual-surface claims", () => {
    const missing = structuredClone(goodResult());
    missing.surfaceStatus = "dual_surface_attested";
    expect(validatePairedApiEvidence(missing).join("\n")).toMatch(
      /without a separate raw API artifact/,
    );

    const valid = structuredClone(goodResult());
    valid.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(valid);
    try {
      valid.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(valid)).toEqual([]);

      valid.pairedApiArtifact.value.backend_pid = 9999;
      expect(validatePairedApiEvidence(valid).join("\n")).toMatch(
        /metadata does not match|backend identity/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects substituted producer, direct/gateway, source, and bundle identities", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      value.bases.gateway = value.bases.direct;
      value.identity.source.before.head = "stale";
      value.identity.source.after.head = "stale";
      value.identity.source.declared_head = "stale";
      value.identity.bundle.before.fingerprint_sha256 = otherSha;
      value.identity.bundle.after.fingerprint_sha256 = otherSha;
      for (const phase of ["before", "after"]) {
        value.identity.runner[phase].producer_pid =
          result.healthProvenance.after.binding.backend_pid;
        value.identity.runner[phase].producer_harness_sha256 = otherSha;
      }
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "substituted-identities.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /origins are not bound|source identity mismatch|bundle identity|producer executable\/harness bytes|aliases Electron\/backend/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects duplicate, out-of-order, and post-terminal protocol events", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const round = value.flows.direct.chat.stream.rounds[0];
      round.terminals.push("DONE");
      round.events.push({
        at_ms: 0,
        channel: "content",
        kind: "chat.content.delta",
      });
      value.flows.direct.chat.stream.terminal_classification[0].values =
        round.terminals;
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-terminals.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /terminal status\/count\/order|non-terminal output after terminalization|timestamps are not monotonic/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects reversed Chat and Anthropic native terminal arrays", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      for (const [baseLabel, protocol] of [
        ["direct", "chat"],
        ["gateway", "anthropic"],
      ]) {
        const flow = value.flows[baseLabel][protocol].stream;
        flow.rounds[2].terminals.reverse();
        const terminalEvents = flow.rounds[2].events.filter(
          (event: Record<string, any>) => event.channel === "terminal",
        );
        terminalEvents.reverse().forEach(
          (event: Record<string, any>, index: number) => {
            const target = flow.rounds[2].events.findIndex(
              (candidate: Record<string, any>) =>
                candidate.channel === "terminal"
                && candidate.kind === event.kind,
            );
            flow.rounds[2].events[target] = {
              ...event,
              at_ms: 100 + index,
            };
          },
        );
        flow.terminal_classification[2].values =
          structuredClone(flow.rounds[2].terminals);
      }
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "reversed-native-terminals.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /terminal status\/count\/order|terminal classifications are not exact/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects malformed tool schemas, arguments, result linkage, and missing reasoning chain", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const flow = value.flows.gateway.responses.stream;
      flow.requests[0].tool_contracts[0].parameters.required = [];
      flow.rounds[0].tool_calls[0].arguments = { path: "wrong.json" };
      flow.executions[0].output_sha256 = otherSha;
      flow.rounds[1].reasoning_chars = 0;
      flow.rounds[1].reasoning_sha256 = crypto
        .createHash("sha256")
        .update("")
        .digest("hex");
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-tools.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /tool schema is not exact|exact tool name\/arguments|not exactly linked|lacks a separate reasoning payload|does not prove reasoning/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects tool-choice, continuation, history-order, and direct/gateway request drift", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      value.second_tool_choice = "auto";
      value.flows.gateway.chat.nonstream.requests[1].tool_choice = "auto";
      value.flows.direct.responses.stream.requests[1].previous_response_id =
        "wrong-response";
      value.flows.direct.anthropic.nonstream.requests[1].tool_history_linkage.reverse();
      value.flows.gateway.ollama.stream.requests[0].canonical_body_sha256 =
        otherSha;
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-request-chain.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /exact passing vmlx-agentic|tool_choice is not exact|continuation ID is not exact|tool history order is not exact|canonical request bodies are not byte-parity equivalent/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects a forged raw manifest and artifact path traversal", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const forgedManifest = structuredClone(value.raw_capture);
      for (const field of [
        "manifest_file",
        "manifest_path",
        "manifest_sha256",
        "run_directory",
      ]) delete forgedManifest[field];
      forgedManifest.routes[0].artifacts[0].body_file = "../escape.bin";
      const forgedManifestPath = path.join(
        value.raw_capture.run_directory,
        "forged-manifest.json",
      );
      const forgedManifestText = `${JSON.stringify(forgedManifest, null, 2)}\n`;
      writePrivateArtifactFile(forgedManifestPath, forgedManifestText);
      value.raw_capture = {
        ...forgedManifest,
        manifest_file: "forged-manifest.json",
        manifest_path: forgedManifestPath,
        run_directory: value.raw_capture.run_directory,
        manifest_sha256: crypto
          .createHash("sha256")
          .update(forgedManifestText)
          .digest("hex"),
      };
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "forged-manifest-proof.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /filename is unsafe|raw capture/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects reordered raw reasoning/tool bytes and incomplete Responses terminal status", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const chatCall =
        value.flows.direct.chat.stream.rounds[0].tool_calls[0];
      rewriteRawCaptureBody(value, {
        baseLabel: "direct",
        protocol: "chat",
        captureLabel: "stream-flow-round1",
        body: sse(
          {
            choices: [{
              delta: {
                tool_calls: [{
                  index: 0,
                  id: chatCall.id,
                  function: {
                    name: chatCall.name,
                    arguments: JSON.stringify(chatCall.arguments),
                  },
                }],
              },
            }],
          },
          { choices: [{ delta: { reasoning_content: "Reason chat 1 A." } }] },
          { choices: [{ delta: { reasoning_content: "B." } }] },
          { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
          "[DONE]",
        ),
      });

      const final =
        value.flows.gateway.responses.stream.expected_final;
      rewriteRawCaptureBody(value, {
        baseLabel: "gateway",
        protocol: "responses",
        captureLabel: "stream-flow-round3",
        body: sse(
          { type: "response.output_text.delta", delta: final.slice(0, 10) },
          { type: "response.output_text.delta", delta: final.slice(10) },
          {
            type: "response.completed",
            response: { status: "incomplete" },
          },
        ),
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "bad-raw-order-terminal.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw stream does not preserve progressive reasoning before tool output/,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw bytes do not reproduce the public flow\/request evidence/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects Responses nonstream raw bytes without explicit completed status", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      const expectedFinal =
        value.flows.gateway.responses.nonstream.expected_final;
      rewriteRawCaptureBody(value, {
        baseLabel: "gateway",
        protocol: "responses",
        captureLabel: "nonstream-flow-round3",
        body: JSON.stringify({
          id: "missing-status",
          output: [{
            type: "message",
            content: [{ type: "output_text", text: expectedFinal }],
          }],
        }),
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "missing-nonstream-status.json",
        value,
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw bytes do not reproduce the public flow\/request evidence/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("rejects cross-origin health and forged prepared-body parity", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const value = structuredClone(fixture.artifact.value);
      value.identity.health.gateway.before.url =
        "http://127.0.0.1:8000/health";
      rewriteRawCaptureMetadata(value, {
        baseLabel: "direct",
        protocol: "chat",
        captureLabel: "stream-flow-round1",
        mutate: (metadata, artifact) => {
          metadata.request.prepared_payload_canonical_body_sha256 = otherSha;
          artifact.prepared_payload_canonical_body_sha256 = otherSha;
        },
      });
      result.pairedApiArtifact = writePairedArtifactValue(
        fixture.directory,
        "unbound-health-prepared-body.json",
        value,
      );
      const failures = validatePairedApiEvidence(result).join("\n");
      expect(failures).toMatch(/health URL origin is not bound/);
      expect(failures).toMatch(/raw body\/metadata\/request\/response binding is invalid/);
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("retains only stable DOM proof identifiers, not answer, reasoning, argument, or result payloads", () => {
    for (const relative of [
      "src/renderer/src/components/chat/MessageBubble.tsx",
      "src/renderer/src/components/chat/ReasoningBox.tsx",
      "src/renderer/src/components/chat/InlineToolCall.tsx",
      "src/renderer/src/components/chat/ToolCallStatus.tsx",
    ]) {
      const source = readFileSync(path.join(process.cwd(), relative), "utf8");
      expect(source).not.toMatch(
        /data-vmlx-proof-(?:answer-source|reasoning-source|tool-arguments|tool-result)/,
      );
    }
  });

  it("keeps a valid raw API matrix separate but not dual when UI request correlation is partial", () => {
    const result = goodResult();
    result.requestCorrelation = {
      status: "partial_product_support_missing",
      turns: [],
    };
    result.surfaceStatus = "partial_dual_surface_uncorrelated";
    const fixture = createValidPairedArtifact(result);
    try {
      result.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(result)).toEqual([]);
      const surfaces = deriveProvenSurfaces(result);
      expect(surfaces).toContain("separate_raw_api");
      expect(surfaces).not.toContain("dual_surface");
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("keeps top-level status PARTIAL until exact UI request correlation is verified", () => {
    const verified = goodResult();
    const fixture = createValidPairedArtifact(verified);
    try {
      verified.pairedApiArtifact = fixture.artifact;
      expect(validatePairedApiEvidence(verified)).toEqual([]);
      expect(applyTopLevelCorrelationStatus(verified)).toMatchObject({
        status: "pass",
        pass: true,
        surfaceStatus: "dual_surface_attested",
      });

      const uiOnly = goodResult();
      expect(applyTopLevelCorrelationStatus(uiOnly)).toMatchObject({
        status: "partial",
        pass: false,
        surfaceStatus: "partial_ui_only",
      });

      const partial = structuredClone(verified);
      partial.requestCorrelation = {
        status: "partial_product_support_missing",
        turns: [],
      };
      expect(applyTopLevelCorrelationStatus(partial)).toMatchObject({
        status: "partial",
        pass: false,
        surfaceStatus: "partial_dual_surface_uncorrelated",
      });

      const rendererFailed = structuredClone(verified);
      expect(
        applyTopLevelCorrelationStatus(rendererFailed, {
          rendererFailed: true,
        }),
      ).toMatchObject({
        status: "fail",
        pass: false,
        surfaceStatus: "partial_ui_only",
      });

      expect(
        applyAssertionFailureStatus(
          structuredClone(verified),
          Object.assign(new Error("bound assertion failed"), {
            failures: ["bound assertion failed"],
          }),
        ),
      ).toMatchObject({
        status: "fail",
        pass: false,
        failureStage: "release_assertions",
        assertionFailures: ["bound assertion failed"],
      });
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("requires an exact parent-owned attach-only Electron lifecycle", () => {
    expect(validateAttachOnlyLifecycle({
      cdpUrl: "http://127.0.0.1:9335",
      electronPid: 4242,
      owner: "parent",
      teardownAllowed: false,
    })).toEqual([]);
    expect(validateAttachOnlyLifecycle({
      cdpUrl: "http://example.com:9335",
      electronPid: 0,
      owner: "child",
      teardownAllowed: true,
    }).join("\n")).toMatch(
      /not an exact loopback origin|PID is invalid|not parent-owned|allowed to tear down/,
    );
  });

  it("requires one canonical parent-owned six-phase run intent", () => {
    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-owned-intent-"));
    try {
      const primaryBundle = path.join(directory, "primary");
      const nativeBundle = path.join(directory, "native");
      mkdirSync(primaryBundle);
      mkdirSync(nativeBundle);
      const value = ownedRunIntent(primaryBundle, nativeBundle, directory);
      const raw = JSON.stringify(value);
      const rawSha = crypto.createHash("sha256").update(raw).digest("hex");
      const opened = {
        opened_nofollow: true,
        nlink: 1,
        mode: 0o600,
        sha256: rawSha,
        value,
      };
      const options = {
        runId: "run",
        nonce: "nonce",
        expectedSha256: rawSha,
        expectedSourceCommit: "3".repeat(40),
        expectedSourceTree: "4".repeat(40),
        expectedUiHarnessSha256: (
          (value.harnesses as Record<string, Record<string, string>>).ui.sha256
        ),
        harnessRoot: directory,
        activePhaseIndex: 0,
        activeModel: "primary-model",
        activeModelBundlePath: primaryBundle,
        expectedDirectBaseUrl: "http://127.0.0.1:8001",
        expectedGatewayBaseUrl: "http://127.0.0.1:8088",
      };
      expect(validateOwnedRunIntent(opened, options)).toEqual([]);
      expect(validateOwnedRunIntent(opened, {
        ...options,
        activePhaseIndex: 5,
        activeModel: "native-model",
        activeModelBundlePath: nativeBundle,
        expectedDirectBaseUrl: "http://127.0.0.1:8002",
      })).toEqual([]);

      const reordered = structuredClone(value);
      [
        (reordered.phase_plan as Array<Record<string, unknown>>)[0],
        (reordered.phase_plan as Array<Record<string, unknown>>)[1],
      ] = [
        (reordered.phase_plan as Array<Record<string, unknown>>)[1],
        (reordered.phase_plan as Array<Record<string, unknown>>)[0],
      ];
      reordered.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(reordered).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: reordered },
        options,
      ).join("\n")).toMatch(/phase 0 policy\/order/);

      const forged = structuredClone(value);
      (forged.harnesses as Record<string, unknown>).extra = {
        relative_path: "unexpected",
        sha256: sha,
      };
      (forged.phase_plan as Array<Record<string, unknown>>)[0].unexpected = true;
      (forged as Record<string, unknown>).unknown = true;
      expect(validateOwnedRunIntent(
        { ...opened, value: forged },
        options,
      ).join("\n")).toMatch(/top-level fields are missing or unexpected/);

      const crossOrigin = structuredClone(value);
      crossOrigin.direct_health_url = "http://127.0.0.1:9999/health";
      crossOrigin.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(crossOrigin).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: crossOrigin },
        options,
      ).join("\n")).toMatch(/primary\/native\/gateway endpoint binding is invalid/);

      const sharedDirect = structuredClone(value);
      sharedDirect.native_direct_base_url = sharedDirect.direct_base_url;
      sharedDirect.native_direct_health_url = sharedDirect.direct_health_url;
      sharedDirect.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(sharedDirect).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: sharedDirect },
        options,
      ).join("\n")).toMatch(/primary\/native\/gateway endpoint binding is invalid/);

      const nativePath = structuredClone(value);
      nativePath.native_direct_base_url = "http://127.0.0.1:8002/extra";
      nativePath.canonical_sha256 = canonicalHash(
        Object.fromEntries(
          Object.entries(nativePath).filter(([key]) => key !== "canonical_sha256"),
        ),
      );
      expect(validateOwnedRunIntent(
        { ...opened, value: nativePath },
        options,
      ).join("\n")).toMatch(/primary\/native\/gateway endpoint binding is invalid/);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("accepts only the exact parent-owned prior phase session attestation", () => {
    const directory = mkdtempSync(
      path.join(tmpdir(), "vmlx-reuse-session-attestation-"),
    );
    try {
      const activePhase = {
        phase_index: 2,
        representative_id: "primary_tq_supported",
        bundle_fingerprint_sha256: "1".repeat(64),
      };
      const options = {
        runId: "run",
        nonce: "nonce",
        runIntentSha256: "2".repeat(64),
        sessionId: "session-primary",
        activePhase,
        model: "primary-model",
        modelBundlePath: directory,
        electronPid: 4103,
        cdpOrigin: "http://127.0.0.1:9335",
        gatewayPid: 4102,
        gatewayBaseUrl: "http://127.0.0.1:8080",
        sourceCommit: "3".repeat(40),
        sourceTree: "4".repeat(40),
      };
      const opened = {
        opened_nofollow: true,
        nlink: 1,
        mode: 0o600,
        value: {
          schema: "vmlx-r18-owned-ui-session-attestation-v5",
          run_id: options.runId,
          nonce: options.nonce,
          run_intent_sha256: options.runIntentSha256,
          phase_index: 1,
          representative_id: activePhase.representative_id,
          session_id: options.sessionId,
          model: options.model,
          model_bundle_path: directory,
          bundle_fingerprint_sha256:
            activePhase.bundle_fingerprint_sha256,
          electron_pid: options.electronPid,
          cdp_origin: options.cdpOrigin,
          gateway_pid: options.gatewayPid,
          gateway_base_url: options.gatewayBaseUrl,
          lifecycle_owner: "parent",
          source_commit: options.sourceCommit,
          source_tree: options.sourceTree,
          created_at: new Date().toISOString(),
        },
      };
      expect(validateOwnedReuseSessionAttestation(opened, options)).toEqual([]);

      for (const [field, value] of [
        ["phase_index", 0],
        ["model", "wrong-model"],
        ["lifecycle_owner", "ui-proof-child"],
        ["electron_pid", 9999],
      ] as const) {
        const tampered = structuredClone(opened);
        (tampered.value as Record<string, unknown>)[field] = value;
        expect(
          validateOwnedReuseSessionAttestation(tampered, options).join("\n"),
        ).toMatch(/stale, wrong-phase, wrong-model|not owned/);
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects stale, mismatched, preexisting, and symlink release sentinels", async () => {
    const now = Date.now();
    const runIntentSha = "d".repeat(64);
    const sessionAttestationSha = "e".repeat(64);
    const activePhase = {
      phase_index: 0,
      phase_name: "primary_ssd_only_store",
      representative_id: "primary_tq_supported",
      bundle_role: "primary",
      cache_policy: "q4",
      paged_ram: false,
      ui_action_profile: "primary-reasoning-render-store",
      ui_turn_count: 1,
      api_action_profile: "full-agentic-plus-cache-store",
      model: "primary-model",
      bundle_fingerprint_sha256: "1".repeat(64),
    };
    const validOpened = {
      opened_nofollow: true,
      nlink: 1,
      mode: 0o600,
      value: {
        schema: "vmlx-r18-owned-ui-release-v5",
        run_id: "run",
        nonce: "nonce",
        session_id: "session",
        ...activePhase,
        run_intent_sha256: runIntentSha,
        ui_session_attestation_sha256: sessionAttestationSha,
        api_capture_sha256: otherSha,
        cache_capture_sha256: "c".repeat(64),
        released_at: new Date(now + 1000).toISOString(),
      },
    };
    expect(validateOwnedUiReleaseSentinel(validOpened, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    })).toEqual([]);

    const mismatched = structuredClone(validOpened);
    mismatched.value.nonce = "wrong";
    mismatched.value.released_at = new Date(now - 1000).toISOString();
    expect(validateOwnedUiReleaseSentinel(mismatched, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    }).join("\n")).toMatch(/does not match|stale/);

    const missingRunIntent = structuredClone(validOpened);
    delete missingRunIntent.value.run_intent_sha256;
    expect(validateOwnedUiReleaseSentinel(missingRunIntent, {
      runId: "run",
      nonce: "nonce",
      sessionId: "session",
      orchestrated: true,
      runIntentSha256: runIntentSha,
      uiSessionAttestationSha256: sessionAttestationSha,
      activePhase,
      notBeforeMs: now,
    }).join("\n")).toMatch(/run_intent_sha256 is missing or unbound/);

    const directory = mkdtempSync(path.join(tmpdir(), "vmlx-owned-release-"));
    try {
      const runIntentArtifact = path.join(directory, "run-intent.json");
      const sessionAttestationArtifact = path.join(directory, "ui-session.json");
      const apiArtifact = path.join(directory, "api.json");
      const cacheArtifact = path.join(directory, "cache.json");
      writePrivateArtifactFile(runIntentArtifact, '{"intent":"exact"}');
      writePrivateArtifactFile(
        sessionAttestationArtifact,
        '{"ui_session":"exact"}',
      );
      writePrivateArtifactFile(apiArtifact, '{"api":"exact"}');
      writePrivateArtifactFile(cacheArtifact, '{"cache":"exact"}');
      const exactRunIntentSha = crypto
        .createHash("sha256")
        .update(readFileSync(runIntentArtifact))
        .digest("hex");
      const exactSessionAttestationSha = crypto
        .createHash("sha256")
        .update(readFileSync(sessionAttestationArtifact))
        .digest("hex");
      const apiSha = crypto
        .createHash("sha256")
        .update(readFileSync(apiArtifact))
        .digest("hex");
      const cacheSha = crypto
        .createHash("sha256")
        .update(readFileSync(cacheArtifact))
        .digest("hex");
      const boundValue = {
        ...structuredClone(validOpened.value),
        run_intent_sha256: exactRunIntentSha,
        ui_session_attestation_sha256: exactSessionAttestationSha,
        api_capture_sha256: apiSha,
        cache_capture_sha256: cacheSha,
      };
      const artifactBindings = {
        runIntentPath: runIntentArtifact,
        runIntentSha256: exactRunIntentSha,
        uiSessionAttestationPath: sessionAttestationArtifact,
        uiSessionAttestationSha256: exactSessionAttestationSha,
        activePhase,
      };
      const preexisting = path.join(directory, "preexisting.json");
      writePrivateArtifactFile(preexisting, JSON.stringify(boundValue));
      await expect(waitForOwnedUiReleaseSentinel({
        filePath: preexisting,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 50,
        pollMs: 1,
      })).rejects.toThrow(/existed before/);

      const target = path.join(directory, "target.json");
      const link = path.join(directory, "release-link.json");
      writePrivateArtifactFile(target, JSON.stringify(validOpened.value));
      const waiter = waitForOwnedUiReleaseSentinel({
        filePath: link,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 250,
        pollMs: 2,
      });
      setTimeout(() => symlinkSync(target, link), 10);
      await expect(waiter).rejects.toThrow(/regular, non-symlink/);

      const released = path.join(directory, "released.json");
      const releasedWaiter = waitForOwnedUiReleaseSentinel({
        filePath: released,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 250,
        pollMs: 2,
      });
      setTimeout(
        () => writePrivateArtifactFile(released, JSON.stringify(boundValue)),
        10,
      );
      await expect(releasedWaiter).resolves.toMatchObject({
        run_intent_sha256: exactRunIntentSha,
        ui_session_attestation_sha256: exactSessionAttestationSha,
        api_capture_sha256: apiSha,
        cache_capture_sha256: cacheSha,
        run_intent_path: realpathSync(runIntentArtifact),
        ui_session_attestation_path:
          realpathSync(sessionAttestationArtifact),
        api_capture_path: realpathSync(apiArtifact),
        cache_capture_path: realpathSync(cacheArtifact),
      });

      const unbound = path.join(directory, "unbound.json");
      const unboundWaiter = waitForOwnedUiReleaseSentinel({
        filePath: unbound,
        runId: "run",
        nonce: "nonce",
        sessionId: "session",
        ...artifactBindings,
        apiArtifactPath: apiArtifact,
        cacheArtifactPath: cacheArtifact,
        notBeforeMs: now,
        timeoutMs: 250,
        pollMs: 2,
      });
      setTimeout(
        () => writePrivateArtifactFile(
          unbound,
          JSON.stringify({
            ...boundValue,
            cache_capture_sha256: otherSha,
          }),
        ),
        10,
      );
      await expect(unboundWaiter).rejects.toThrow(
        /hashes do not match safely reopened exact artifacts/,
      );
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("parses route, model, and only named resolved sampling kwargs", () => {
    expect(
      parseResolvedSamplingKwargs([
        "INFO Resolved sampling kwargs route=/v1/chat/completions model=/models/laguna kwargs={'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'enable_thinking': True}",
      ]),
    ).toEqual([
      {
        route_model:
          "Resolved sampling kwargs route=/v1/chat/completions model=/models/laguna",
        route: "/v1/chat/completions",
        model: "/models/laguna",
        raw: "{'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'enable_thinking': True}",
        values: {
          temperature: 0.7,
          top_p: 0.9,
          top_k: 40,
          enable_thinking: true,
        },
      },
    ]);
  });

  it("parses the complete server-emitted proof, request, and message identity tuple", () => {
    expect(
      parseResolvedSamplingKwargs([
        "INFO Resolved sampling kwargs route=/v1/responses model=/models/gemma proof_request_id=user-1 request_id=wire-1 message_id=assistant-1 kwargs={'temperature': 0.8}",
      ])[0],
    ).toMatchObject({
      route: "/v1/responses",
      model: "/models/gemma",
      proof_request_id: "user-1",
      request_id: "wire-1",
      message_id: "assistant-1",
      correlation_source: "server_emitted",
      values: { temperature: 0.8 },
    });
  });

  it("accepts multiple unique tool-continuation request IDs under one UI turn and rejects reuse", () => {
    const result = structuredClone(goodResult());
    const followupId = "wire-request-1-followup";
    result.uiTurnEvidence[0].requestIds.push(followupId);
    result.requestCorrelation.turns[0].serverRequestIds.push(followupId);
    result.resolvedSamplingRecords.splice(1, 0, {
      ...structuredClone(result.resolvedSamplingRecords[0]),
      request_id: followupId,
    });
    expect(isServerRequestCorrelationVerified(result)).toBe(true);

    result.uiTurnEvidence[1].requestIds = [followupId];
    result.requestCorrelation.turns[1].serverRequestIds = [followupId];
    result.resolvedSamplingRecords.find(
      (record: Record<string, unknown>) => record.proof_request_id === "user-2",
    ).request_id = followupId;
    expect(isServerRequestCorrelationVerified(result)).toBe(false);
  });

  it("keeps a fixed full-tuple hash suffix when long sanitized prefixes collide", () => {
    const common = "x".repeat(400);
    const first = uniqueProofBasename({
      requested: common,
      model: `${common}-model-a`,
      run: `${common}-run-a`,
    });
    const second = uniqueProofBasename({
      requested: common,
      model: `${common}-model-b`,
      run: `${common}-run-b`,
    });
    expect(first).not.toBe(second);
    expect(first).toMatch(/-[0-9a-f]{20}$/);
    expect(second).toMatch(/-[0-9a-f]{20}$/);
    expect(first.length).toBeLessThanOrEqual(240);
    expect(second.length).toBeLessThanOrEqual(240);
  });

  it("rejects metadata-only or self-referential API evidence instead of treating JSON existence as dual proof", () => {
    const result = goodResult();
    result.surfaceStatus = "dual_surface_attested";
    const fixture = createValidPairedArtifact(result);
    try {
      const metadataOnly = structuredClone(fixture.artifact);
      metadataOnly.value.protocols = {};
      result.pairedApiArtifact = metadataOnly;
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /metadata does not match|exact Chat\/Responses\/Anthropic\/Ollama/,
      );

      const selfPath = path.join(fixture.directory, "self-referential.json");
      const selfValue = structuredClone(fixture.artifact.value);
      selfValue.raw_capture.manifest_path = selfPath;
      selfValue.raw_capture.manifest_sha256 = "0".repeat(64);
      writePrivateArtifactFile(selfPath, JSON.stringify(selfValue));
      result.pairedApiArtifact = readPrivateExternalJson(
        selfPath,
        "Self-referential API fixture",
      );
      expect(validatePairedApiEvidence(result).join("\n")).toMatch(
        /raw capture manifest|manifest bytes\/path\/summary/,
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("keeps time-window logs and global cache deltas PARTIAL without server-emitted request IDs", () => {
    const result = goodResult();
    result.requestCorrelation = {
      status: "partial_product_support_missing",
      reason: "server does not emit proof request IDs",
      turns: result.uiTurnEvidence.map((turn: Record<string, any>) => ({
        turn: turn.turn,
        proofRequestId: turn.proofRequestId,
        userMessageId: turn.userMessageId,
        assistantMessageId: turn.assistantMessageId,
        serverProofRequestId: null,
        serverRequestId: null,
        serverMessageId: null,
      })),
    };
    result.resolvedSamplingRecords = result.resolvedSamplingRecords.map(
      (record: Record<string, any>, index: number) => {
        const copy = { ...record, observed_window_turn: index + 1 };
        delete copy.proof_request_id;
        delete copy.request_id;
        delete copy.message_id;
        delete copy.correlation_source;
        return copy;
      },
    );
    result.cacheRequestEvidence = result.cacheRequestEvidence.map(
      (row: Record<string, any>) => ({
        ...row,
        correlationStatus: "partial_product_support_missing",
        serverRequestId: null,
        serverObservation: null,
      }),
    );

    expect(isServerRequestCorrelationVerified(result)).toBe(false);
    expect(validateGenerationDefaultsEvidence(result).join("\n")).toMatch(
      /lack exact proof\/request\/message correlation/,
    );
    expect(validateRequestCorrelatedCacheEvidence(result).join("\n")).toMatch(
      /do not exactly match|lacks exact server-emitted request correlation/,
    );
    const surfaces = deriveProvenSurfaces(result);
    expect(surfaces).not.toContain("generation_defaults_visible_ui");
    expect(surfaces).not.toContain("generation_defaults_applied");
    expect(surfaces).not.toContain("cache_hit_telemetry");
  });
});

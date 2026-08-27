import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { isExpectedProcessStdioDisconnectError } from "../src/main/childProcessStreamGuards";

const source = readFileSync(
  resolve(process.cwd(), "src/main/api-gateway.ts"),
  "utf8",
);
const bodySource = readFileSync(
  resolve(process.cwd(), "src/main/gateway-body.ts"),
  "utf8",
);
const codeSnippetsSource = readFileSync(
  resolve(process.cwd(), "src/renderer/src/components/api/CodeSnippets.tsx"),
  "utf8",
);

describe("Ollama gateway parity contracts", () => {
  it("translates Ollama image messages into OpenAI content parts", () => {
    expect(source).toContain("private translateOllamaMessages");
    expect(source).toContain("data:image/png;base64,");
    expect(source).toContain('type: "image_url"');
    expect(source).toContain("messages: this.translateOllamaMessages");
  });

  it("translates vMLX Ollama video and audio message extensions", () => {
    expect(source).toContain('data:video/mp4;base64,${video}');
    expect(source).toContain('type: "video_url"');
    expect(source).toContain('data:audio/wav;base64,${audioItem}');
    expect(source).toContain('type: "audio_url"');
  });

  it("maps Ollama json and schema formats to OpenAI response_format", () => {
    expect(source).toContain("private ollamaResponseFormat");
    expect(source).toContain('format === "json"');
    expect(source).toContain('type: "json_object"');
    expect(source).toContain('type: "json_schema"');
    expect(source).toContain('name: "ollama_schema"');
  });

  it("preserves reasoning deltas as Ollama thinking output", () => {
    expect(source).toContain("delta?.reasoning_content || delta?.reasoning");
    expect(source).toContain(
      "choice?.message?.reasoning_content || choice?.message?.reasoning",
    );
    expect(source).toContain("message.thinking");
  });

  it("forwards thinking kwargs through app gateway Ollama routes", () => {
    expect(source).toContain("parsed?.enable_thinking");
    expect(source).toContain("private applyOllamaThinking");
    expect(source).toContain("private normalizeOllamaBoolean");
    expect(source).toContain("const think = this.normalizeOllamaBoolean(parsed?.think)");
    expect(source).toContain("const enableThinking = this.normalizeOllamaBoolean(parsed?.enable_thinking)");
    expect(source).toContain("openaiBody.enable_thinking = think");
    expect(source).toContain("private shouldForwardOllamaReasoningEffort");
    expect(source).toContain("openaiBody?.enable_thinking === false");
    expect(source).toContain("openaiBody.reasoning_effort = parsed.reasoning_effort");
    expect(source).toContain("openaiBody.chat_template_kwargs = parsed.chat_template_kwargs");
  });

  it("does not force enable_thinking when Ollama thinking controls are omitted", () => {
    expect(source).toContain("private applyOllamaThinking");
    expect(source).not.toContain("Omitted thinking controls default on");
    expect(source).not.toContain("openaiBody.enable_thinking = true;");
  });

  it("routes Ollama generate through chat templates unless raw=true", () => {
    expect(source).toContain("const useRawCompletion = parsed.raw === true");
    expect(source).toContain("prompt: parsed.prompt || \"\"");
    expect(source).toContain("messages: [");
    expect(source).toContain('path: backendPath');
    expect(source).toContain('choice?.message?.content || ""');
  });

  it("forwards cache bypass controls through chat and generate routes", () => {
    const cacheSaltForwards =
      source.match(/openaiBody\.cache_salt = parsed\.cache_salt/g) || [];
    const skipForwards =
      source.match(
        /openaiBody\.skip_prefix_cache = parsed\.skip_prefix_cache/g,
      ) || [];
    expect(cacheSaltForwards.length).toBeGreaterThanOrEqual(2);
    expect(skipForwards.length).toBeGreaterThanOrEqual(2);
  });

  it("forwards Ollama min_p sampling through chat and generate routes", () => {
    const forwards = source.match(/openaiBody\.min_p = opts\.min_p/g) || [];
    expect(forwards.length).toBeGreaterThanOrEqual(2);
  });

  it("forwards Ollama num_ctx/max context controls as max_prompt_tokens", () => {
    expect(source).toContain("private applyOllamaPromptContextLimit");
    expect(source).toContain("opts?.num_ctx");
    expect(source).toContain("opts?.max_prompt_tokens");
    expect(source).toContain("parsed?.max_context_tokens");
    const forwards = source.match(/openaiBody\.max_prompt_tokens = Math\.floor\(parsedValue\)/g) || [];
    expect(forwards.length).toBeGreaterThanOrEqual(1);
    const calls = source.match(/this\.applyOllamaPromptContextLimit\(parsed, opts, openaiBody\)/g) || [];
    expect(calls.length).toBeGreaterThanOrEqual(2);
  });

  it("preserves backend prompt_too_long status for Ollama gateway routes", () => {
    expect(source).toContain("private sendOllamaBackendError");
    const statusChecks = source.match(/proxyRes\.statusCode \|\| 200\) >= 400/g) || [];
    expect(statusChecks.length).toBeGreaterThanOrEqual(4);
    const errorForwards = source.match(/this\.sendOllamaBackendError\(res, proxyRes\.statusCode \|\| 502/g) || [];
    expect(errorForwards.length).toBeGreaterThanOrEqual(4);
    expect(source).toContain("error?.message");
    expect(source).toContain("error?.code");
  });

  it("applies gateway timeout handling to Ollama embeddings proxy requests", () => {
    const start = source.indexOf("private async handleOllamaEmbed");
    const end = source.indexOf("// ═══════════════════════════════════════════════════════════════", start);
    const embedSource = source.slice(start, end);

    expect(embedSource).toContain('path: "/v1/embeddings"');
    expect(embedSource).toContain("timeout: this.effectiveGatewayProxyTimeoutMs(routedSession, parsed)");
    expect(embedSource).toContain('proxyReq.on("timeout"');
    expect(embedSource).toContain('this.sendJson(res, 504, { error: "Timed out" })');
  });

  it("implements Ollama HEAD/root and version probes for strict clients", () => {
    expect(source).toContain('this.endResponse(res, "Ollama is running\\n")');
    expect(source).toContain('url === "/api/version"');
    expect(source).toContain('version: "0.12.6"');
    expect(source).toContain('method === "HEAD"');
    expect(source).toContain('url === "/" || url === "/api/version"');
  });

  it("shows API key auth in copied Ollama gateway snippets when configured", () => {
    expect(codeSnippetsSource).toContain("function buildOllamaAuthHeader");
    expect(codeSnippetsSource).toContain("Authorization: Bearer ${apiKey}");
    expect(codeSnippetsSource).toContain("headers=headers");
    expect(codeSnippetsSource).not.toContain("function buildOllamaCurl(baseUrl: string, _apiKey");
    expect(codeSnippetsSource).not.toContain("function buildOllamaPython(baseUrl: string, _apiKey");
  });

  it("enforces gateway-side API keys for local model lists and routed sessions", () => {
    expect(source).toContain("private requireAnyGatewayAuth");
    expect(source).toContain("private requireSessionAuth");
    expect(source).toContain("private gatewayApiKeys");
    expect(source).toContain('code: "invalid_api_key"');
    expect(source).toContain("if (!this.requireAnyGatewayAuth(req, res)) return;");
    expect(source).toContain("if (!this.requireSessionAuth(req, res, session)) return;");
    expect(source).toContain("headers: this.jsonProxyHeadersWithAuth(req)");
  });

  it("returns the actual active gateway port after restart", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    expect(mainSource).toContain("function gatewayStatusPayload()");
    expect(mainSource).toContain("activeRequests: apiGateway.activeRequestCount");
    expect(mainSource).toContain(
      "return gatewayStatusPayload()",
    );
    expect(mainSource).not.toContain(
      "return { running: true, port, host: apiGateway.activeHost }",
    );
  });

  it("persists gateway host and port only after the replacement listener binds", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    const restartHandler = mainSource.slice(
      mainSource.indexOf("ipcMain.handle('gateway:restart'"),
      mainSource.indexOf("ipcMain.handle('gateway:setSingleModelMode'"),
    );

    expect(restartHandler).toContain("await apiGateway.restart(port, host)");
    expect(restartHandler).not.toContain("db.setSetting('gateway_port'");
    expect(restartHandler).not.toContain("db.setSetting('gateway_host'");
  });

  it("does not turn client disconnect EPIPE into the unexpected-error crash dialog", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    expect(mainSource).toContain("function isExpectedClientDisconnectError");
    expect(mainSource).toContain("code === 'EPIPE'");
    expect(mainSource).toContain("code === 'ERR_STREAM_WRITE_AFTER_END'");
    expect(mainSource).toContain("write EPIPE");
    expect(mainSource).toContain("broken pipe");
    expect(mainSource).toContain("const cause = (err as any)?.cause");
    expect(mainSource).toContain("const wrappedDisconnects = [");
    expect(mainSource).toContain("(err as any)?.reason");
    expect(mainSource).toContain("(err as any)?.error");
    expect(mainSource).toContain("(err as any)?.detail");
    expect(mainSource).toContain("wrappedDisconnects.some((nested) => isExpectedClientDisconnectError(nested))");
    expect(mainSource).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
    expect(mainSource).toContain("nestedErrors.some((nested) => isExpectedClientDisconnectError(nested))");
    expect(mainSource).toContain("if (isExpectedClientDisconnectError(error))");
    expect(mainSource).toContain("if (isExpectedClientDisconnectError(reason))");
  });

  it("guards child process stdio stream EPIPE across app-managed process lanes", () => {
    // This matcher used to be copied into all five lanes and this test asserted
    // the whole body appeared in each — i.e. it pinned the DUPLICATION. It is a
    // rule that decides whether an error is SWALLOWED, so five copies meant a
    // newly-observed disconnect shape added to one lane left the other four
    // reporting it as a fault, with nothing to announce the split. (At
    // extraction time four were byte-identical and ipc/models.ts differed only
    // in quote style.) Assert the rule at its owner, and that every lane reads
    // it.
    const guardSource = readFileSync(
      resolve(process.cwd(), "src/main/childProcessStreamGuards.ts"),
      "utf8",
    );
    for (const code of [
      "EPIPE",
      "ECONNRESET",
      "ERR_STREAM_DESTROYED",
      "ERR_STREAM_WRITE_AFTER_END",
    ]) {
      expect(guardSource).toMatch(new RegExp(`code === ['"]${code}['"]`));
    }
    for (const phrase of [
      "write EPIPE",
      "broken pipe",
      "socket hang up",
      "connection reset",
      "premature close",
    ]) {
      expect(guardSource).toContain(phrase);
    }
    // The disconnect can arrive wrapped; the matcher must still recurse.
    expect(guardSource).toContain("const cause = (err as any)?.cause");
    expect(guardSource).toContain("const wrappedDisconnects = [");
    expect(guardSource).toContain("(err as any)?.reason");
    expect(guardSource).toContain("(err as any)?.error");
    expect(guardSource).toContain("(err as any)?.detail");
    expect(guardSource).toContain(
      "wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested))",
    );
    expect(guardSource).toContain(
      "const nestedErrors = Array.isArray((err as any)?.errors)",
    );

    for (const rel of [
      "src/main/process-manager.ts",
      "src/main/engine-manager.ts",
      "src/main/ipc/developer.ts",
      "src/main/ipc/models.ts",
      "src/main/tools/executor.ts",
    ]) {
      const sourceText = readFileSync(resolve(process.cwd(), rel), "utf8");
      expect(sourceText).toContain("childProcessStreamGuards");
      expect(sourceText).toContain("attachChildProcessStreamErrorGuard(");
      expect(sourceText).not.toContain(
        "function isExpectedChildProcessStreamDisconnectError(",
      );
    }
  });

  it("swallows orphaned Electron process-pipe EIO without hiding filesystem EIO", () => {
    const pipeError = Object.assign(new Error("write EIO"), { code: "EIO" });
    const fileError = Object.assign(
      new Error("EIO: i/o error, write '/Volumes/EricsLLMDrive/model.safetensors'"),
      { code: "EIO" },
    );
    expect(isExpectedProcessStdioDisconnectError(pipeError)).toBe(true);
    expect(isExpectedProcessStdioDisconnectError(fileError)).toBe(false);

    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    expect(mainSource).toContain(
      "for (const stream of [process.stdout, process.stderr])",
    );
    expect(mainSource).toContain("attachProcessStdioErrorGuard(stream");
    expect(mainSource).not.toContain("code === 'EIO' ||");
  });

  it("guards live proof script child stdio EPIPE while collecting e2e evidence", () => {
    const liveRealUiProof = readFileSync(
      resolve(process.cwd(), "scripts/live-real-ui-model-proof.mjs"),
      "utf8",
    );
    const liveChatToolsProof = readFileSync(
      resolve(process.cwd(), "scripts/live-chat-tools-reasoning-proof.mjs"),
      "utf8",
    );

    for (const sourceText of [liveRealUiProof, liveChatToolsProof]) {
      expect(sourceText).toContain("function attachChildProcessStreamErrorGuard");
      expect(sourceText).toContain("isSocketDisconnectError(error)");
      expect(sourceText).toContain("stream?.on('error'");
      expect(sourceText).toContain(".stdout,");
      expect(sourceText).toContain(".stderr,");
    }
  });

  it("does not relay backend stderr EPIPE lines as user-visible session errors", () => {
    const sessionsSource = readFileSync(
      resolve(process.cwd(), "src/main/sessions.ts"),
      "utf8",
    );
    const backendStderrSource = readFileSync(
      resolve(process.cwd(), "src/main/backend-stderr.ts"),
      "utf8",
    );
    expect(backendStderrSource).toContain("function isExpectedBackendStderrDisconnectLine");
    expect(backendStderrSource).toContain("normalizeBackendStderrChunk");
    expect(backendStderrSource).toContain("write EPIPE");
    expect(backendStderrSource).toContain("broken pipe");
    expect(sessionsSource).toContain("normalizeBackendStderrChunk(");
    expect(sessionsSource).toContain(
      "data: BACKEND_STDERR_DISCONNECT_NORMALIZED_LINE",
    );
    expect(sessionsSource).toContain("return");
    const stderrHandler = sessionsSource.indexOf("proc.stderr?.on('data'");
    const disconnectNormalizer = sessionsSource.indexOf(
      "normalizeBackendStderrChunk(",
      stderrHandler,
    );
    const rawErrorLog = sessionsSource.indexOf(
      "console.error(`[SERVER] ${stderrText.trimEnd()}`)",
      stderrHandler,
    );
    expect(disconnectNormalizer).toBeGreaterThan(stderrHandler);
    expect(rawErrorLog).toBeGreaterThan(disconnectNormalizer);
  });

  it("does not leave raw chat IPC backend request finalization unguarded", () => {
    const chatSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/chat.ts"),
      "utf8",
    );
    expect(chatSource).toContain("function endChatBackendRequest");
    expect(chatSource).toContain("closed?: boolean");
    expect(chatSource).toContain("!anyReq.closed");
    expect(chatSource).toContain('code === "EPIPE"');
    expect(chatSource).toContain('code === "ERR_STREAM_DESTROYED"');
    expect(chatSource).toContain('code === "ERR_STREAM_WRITE_AFTER_END"');
    expect(chatSource).toContain("broken pipe");
    expect(chatSource).toContain("const cause = (err as any)?.cause");
    expect(chatSource).toContain("const wrappedDisconnects = [");
    expect(chatSource).toContain("(err as any)?.reason");
    expect(chatSource).toContain("(err as any)?.error");
    expect(chatSource).toContain("(err as any)?.detail");
    expect(chatSource).toContain("wrappedDisconnects.some((nested) => isExpectedChatBackendDisconnectError(nested))");
    expect(chatSource).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
    expect(chatSource).toContain("nestedErrors.some((nested) => isExpectedChatBackendDisconnectError(nested))");
    const rawEnds = chatSource.match(/req\.end\(bodyBuf\);/g) || [];
    expect(rawEnds.length).toBe(1);
    expect(chatSource).toContain("endChatBackendRequest(req, bodyBuf, reject);");
    expect(chatSource).toContain("isExpectedChatBackendDisconnectError(error)");
    expect(chatSource).toContain('errCode === "EPIPE"');
    expect(chatSource).toContain('errCode === "ERR_STREAM_DESTROYED"');
    expect(chatSource).toContain('errMsg.includes("write EPIPE")');
    expect(chatSource).toContain(
      'if (isExpectedChatBackendDisconnectError(err)) {',
    );
    expect(chatSource).toContain(
      '!projectedMetalHeadroomErrorContent &&\n          !promptTooLongErrorContent &&\n          !isExpectedChatBackendDisconnectError(error)',
    );
    expect(chatSource).toContain(
      'console.error("[CHAT] Error caught:",',
    );
    expect(chatSource).not.toContain(
      'console.error("[CHAT] Error caught:", {\n          message: _err?.message,',
    );
  });

  it("guards disconnect-shaped chat stream error chunks before raw logging", () => {
    const chatSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/chat.ts"),
      "utf8",
    );
    expect(chatSource).toContain("function expectedChatBackendDisconnectError");
    expect(chatSource).toContain(
      "if (isExpectedChatBackendDisconnectError(errDetail)) {",
    );
    expect(chatSource).toContain("throw expectedChatBackendDisconnectError();");
    expect(chatSource).toContain(
      "console.error(`[CHAT] Responses API error event: ${errDetail}`);",
    );
    expect(chatSource).toContain(
      "console.error(\n                  `[CHAT] Chat completions error chunk: ${errDetail}`,",
    );
    const responseErrorStart = chatSource.indexOf(
      'currentEventType === "error"',
    );
    const responseErrorLog = chatSource.indexOf(
      "console.error(`[CHAT] Responses API error event: ${errDetail}`);",
      responseErrorStart,
    );
    const chatErrorStart = chatSource.indexOf("// Handle error chunks from Chat Completions");
    const chatErrorLog = chatSource.indexOf(
      "`[CHAT] Chat completions error chunk: ${errDetail}`,",
      chatErrorStart,
    );
    expect(
      chatSource.indexOf("throw expectedChatBackendDisconnectError();", responseErrorStart),
    ).toBeLessThan(responseErrorLog);
    expect(
      chatSource.indexOf("throw expectedChatBackendDisconnectError();", chatErrorStart),
    ).toBeLessThan(chatErrorLog);
  });

  it("normalizes cache IPC endpoint EPIPE disconnects instead of surfacing raw unexpected errors", () => {
    const cacheSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/cache.ts"),
      "utf8",
    );
    expect(cacheSource).toContain("function isExpectedCacheEndpointDisconnectError");
    expect(cacheSource).toContain('code === "EPIPE"');
    expect(cacheSource).toContain('code === "ECONNRESET"');
    expect(cacheSource).toContain('code === "ERR_STREAM_DESTROYED"');
    expect(cacheSource).toContain('code === "ERR_STREAM_WRITE_AFTER_END"');
    expect(cacheSource).toContain("write EPIPE");
    expect(cacheSource).toContain("broken pipe");
    expect(cacheSource).toContain("const cause = (err as any)?.cause");
    expect(cacheSource).toContain("const wrappedDisconnects = [");
    expect(cacheSource).toContain("(err as any)?.reason");
    expect(cacheSource).toContain("(err as any)?.error");
    expect(cacheSource).toContain("(err as any)?.detail");
    expect(cacheSource).toContain("wrappedDisconnects.some((nested) => isExpectedCacheEndpointDisconnectError(nested))");
    expect(cacheSource).toContain("const nestedErrors = Array.isArray((err as any)?.errors)");
    expect(cacheSource).toContain("nestedErrors.some((nested) => isExpectedCacheEndpointDisconnectError(nested))");
    expect(cacheSource).toContain("async function fetchCacheJson");
    expect(cacheSource).toContain('code === "ECONNREFUSED"');
    expect(cacheSource).toContain("expectedUndiciTransportCodes.has(code)");
    const rawFetches = cacheSource.match(/await fetch\(/g) || [];
    expect(rawFetches.length).toBe(1);
    const guardedFetches = cacheSource.match(/await fetchCacheJson\(/g) || [];
    expect(guardedFetches.length).toBeGreaterThanOrEqual(4);
  });

  it("normalizes performance health EPIPE disconnects instead of surfacing raw unexpected errors", () => {
    const performanceSource = readFileSync(
      resolve(process.cwd(), "src/main/ipc/performance.ts"),
      "utf8",
    );
    expect(performanceSource).toContain("function isExpectedPerformanceEndpointDisconnectError");
    expect(performanceSource).toContain("code === 'EPIPE'");
    expect(performanceSource).toContain("code === 'ECONNRESET'");
    expect(performanceSource).toContain("code === 'ERR_STREAM_DESTROYED'");
    expect(performanceSource).toContain("code === 'ERR_STREAM_WRITE_AFTER_END'");
    expect(performanceSource).toContain("write EPIPE");
    expect(performanceSource).toContain("broken pipe");
    expect(performanceSource).toContain("const cause = anyErr?.cause");
    expect(performanceSource).toContain("const wrappedDisconnects = [");
    expect(performanceSource).toContain("anyErr?.reason");
    expect(performanceSource).toContain("anyErr?.error");
    expect(performanceSource).toContain("anyErr?.detail");
    expect(performanceSource).toContain("wrappedDisconnects.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))");
    expect(performanceSource).toContain("const nestedErrors = Array.isArray(anyErr?.errors)");
    expect(performanceSource).toContain("nestedErrors.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))");
    expect(performanceSource).toContain("Performance health connection lost. The model server may have stopped or restarted; retry after the session is healthy.");
  });

  it("aborts Ollama backend response streams when the client response closes", () => {
    expect(source).toContain("private abortProxyResponseOnClientClose");
    expect(source).toContain('res.on("close", () => {');
    expect(source).toContain("proxyRes.destroy();");
    const ollamaAbortCalls =
      source.match(/this\.abortProxyResponseOnClientClose\(res, proxyRes\);/g) || [];
    expect(ollamaAbortCalls.length).toBeGreaterThanOrEqual(3);
    expect(source).toContain(
      "this.abortProxyResponseOnClientClose(clientRes, proxyRes);",
    );
  });

  it("surfaces a usable LAN gateway URL instead of showing 0.0.0.0 to users", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    expect(mainSource).toContain("host === '0.0.0.0' ? getLanAddress()");
    expect(mainSource).toContain("displayHost");
    expect(dashboardSource).toContain("const gatewayDisplayHost = lanEnabled ? (gwLanHost || gwHost) : \"localhost\"");
    expect(dashboardSource).toContain("const gatewayUrl = `http://${gatewayDisplayHost}:${gwPort}`");
  });

  it("shows the gateway port that actually bound after a restart fallback", () => {
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    expect(dashboardSource).toContain("const activePort = status?.port ?? port");
    expect(dashboardSource).toContain("setGwPort(String(activePort))");
    expect(dashboardSource).toContain("setPortInput(String(activePort))");
    expect(dashboardSource).toContain("setGwPort(String(status.port))");
    expect(dashboardSource).not.toContain("setGwPort(String(port));");
  });

  it("exposes a synchronized single-loaded-model gateway mode in main, preload, API dashboard, server settings, and tray", () => {
    const mainSource = readFileSync(
      resolve(process.cwd(), "src/main/index.ts"),
      "utf8",
    );
    const preloadSource = readFileSync(
      resolve(process.cwd(), "src/preload/index.ts"),
      "utf8",
    );
    const envSource = readFileSync(
      resolve(process.cwd(), "src/env.d.ts"),
      "utf8",
    );
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    const drawerSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/sessions/ServerSettingsDrawer.tsx"),
      "utf8",
    );
    const traySource = readFileSync(
      resolve(process.cwd(), "src/main/tray.ts"),
      "utf8",
    );
    const gatewaySettingsKeys = readFileSync(
      resolve(process.cwd(), "src/shared/gatewaySettingsKeys.ts"),
      "utf8",
    );
    // The key now has ONE spelling in src/shared/gatewaySettingsKeys.ts. It was
    // hand-written at six call sites, and because every reader compares
    // `=== 'true'` against a getSetting that returns undefined for an unknown
    // key, a typo in any one of them would silently stop enforcing single-model
    // mode with nothing thrown and no test failing.
    expect(source).toContain("GATEWAY_SINGLE_MODEL_MODE_KEY");
    expect(gatewaySettingsKeys).toContain("'gateway_single_model_mode'");
    expect(source).toContain("enforceSingleModelMode");
    expect(source).toContain("sessionManager.stopSession(s.id)");
    expect(source).toContain("single_model_mode: this.singleModelMode");
    expect(mainSource).toContain("singleModelMode: apiGateway.singleModelMode");
    expect(mainSource).toContain("gateway:setSingleModelMode");
    expect(mainSource).toContain("gateway:singleModelModeChanged");
    expect(preloadSource).toContain("setSingleModelMode");
    expect(preloadSource).toContain("onSingleModelModeChanged");
    expect(envSource).toContain("singleModelMode: boolean");
    expect(envSource).toContain("onSingleModelModeChanged");
    expect(dashboardSource).toContain("singleModelMode");
    expect(dashboardSource).toContain("handleSingleModelModeToggle");
    expect(dashboardSource).toContain("window.api.gateway?.onSingleModelModeChanged");
    expect(drawerSource).toContain("handleGatewaySingleModelModeToggle");
    expect(drawerSource).toContain("window.api.gateway?.onSingleModelModeChanged");
    expect(traySource).toContain("main.tray.singleModelMode");
    expect(traySource).toContain("gateway:singleModelModeChanged");
  });

  it("keeps single-model gateway controls localized and tailwind-safe", () => {
    const dashboardSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/api/ApiDashboard.tsx"),
      "utf8",
    );
    const drawerSource = readFileSync(
      resolve(process.cwd(), "src/renderer/src/components/sessions/ServerSettingsDrawer.tsx"),
      "utf8",
    );
    expect(dashboardSource + drawerSource).not.toContain("translate-x-4.5");
    expect(drawerSource).toContain("useTranslation");
    expect(drawerSource).toContain("t('main.tray.singleModelMode')");
    expect(drawerSource).toContain("t('api.singleModelModeOn')");
    expect(drawerSource).toContain("t('api.singleModelModeOff')");
    expect(drawerSource).not.toContain("API Gateway Single Model");
    expect(drawerSource).not.toContain("Gateway requests unload other local models");
  });

  it("collapses OpenAI tool arguments back to Ollama object arguments", () => {
    expect(source).toContain("private openAIToolCallsToOllama");
    expect(source).toContain("JSON.parse(args)");
    expect(source).toContain(
      "function: { name: tc.function.name, arguments: args }",
    );
  });
});

describe("Gateway passthrough contracts for non-Ollama APIs", () => {
  it("proxies OpenAI, Anthropic, Responses, cache, audio, and MCP paths verbatim", () => {
    expect(source).toMatch(
      /return\s+this\.proxyRequest\(req,\s*res,\s*routedSession,\s*body\);?/,
    );
    expect(source).toContain("path: clientReq.url");
    expect(source).toContain("method: clientReq.method");
    expect(source).toContain("...clientReq.headers");
    expect(source).toContain("if (body.length > 0 && !this.writeProxyBody(proxyReq, body))");
  });

  it("preserves backend status, headers, and streaming response bytes", () => {
    expect(source).toContain("this.writeHeadResponse(");
    expect(source).toContain("proxyRes.statusCode || 502");
    expect(source).toContain("proxyRes.headers");
    expect(source).toContain("proxyRes.on(\"data\"");
    expect(source).toContain("this.writeResponse(clientRes");
    expect(source).not.toContain("proxyRes.pipe(clientRes)");
    expect(source).toContain("preserves SSE Content-Type");
  });

  it("guards gateway streaming writes against client disconnect EPIPE errors", () => {
    expect(source).toContain("private attachResponseErrorGuard");
    expect(source).toContain('code === "EPIPE"');
    expect(source).toContain("write EPIPE");
    expect(source).toContain("const wrappedDisconnects = [");
    expect(source).toContain("(anyErr as any)?.reason");
    expect(source).toContain("(anyErr as any)?.error");
    expect(source).toContain("(anyErr as any)?.detail");
    expect(source).toContain("wrappedDisconnects.some((nested) => this.isClientDisconnectError(nested))");
    expect(source).toContain("this.attachResponseErrorGuard(res)");
    expect(source).toContain("private writeHeadResponse");
    expect(source).toContain("private writeProxyBody");
    expect(source).toContain("private endProxyRequest");
    expect(source).toContain("private writeJsonLine");
    expect(source).toContain("if (!this.writeJsonLine(res, ollamaMsg))");
    expect(source).toContain("private guardProxyResponseLifecycle");
    expect(source).toContain('proxyRes.once("aborted"');
    expect(source).toContain('proxyRes.once("close"');
    expect(source).toContain("closed?: boolean");
    expect(source).toContain("!anyRes.closed");
    expect(source).toContain("!anyReq.closed");
    expect(source).toContain("if (body.length > 0 && !this.writeProxyBody(proxyReq, body))");
    expect(source).toContain("if (!this.endProxyRequest(proxyReq)) return;");
    expect(source).toContain("if (!this.writeProxyBody(proxyReq, JSON.stringify(openaiBody)))");
    expect(source).toContain("proxyRes.destroy()");
  });

  it("does not leave raw backend request end calls unguarded after disconnect", () => {
    expect(source).not.toContain("proxyReq.end()");
    expect(source).not.toContain("cancelReq.end()");
    expect(source).toContain("this.endProxyRequest(cancelReq)");
    expect(source).toContain("this.endProxyRequest(proxyReq)");
  });

  it("does not leave unguarded Ollama streaming json writes alive after disconnect", () => {
    const rawJsonLineWrites = source.match(/this\.writeJsonLine\(\s*res,/g) || [];
    const guardedJsonLineWrites =
      source.match(/if\s*\(\s*!this\.writeJsonLine\(\s*res,/g) || [];

    expect(rawJsonLineWrites.length).toBeGreaterThan(0);
    expect(guardedJsonLineWrites.length).toBe(rawJsonLineWrites.length);
  });

  it("guards every Ollama backend response with the shared lifecycle boundary", () => {
    const ollamaProxyHandlers =
      source.match(
        /const proxyReq = httpRequest\(proxyOpts, \(proxyRes\) => \{[\s\S]*?proxyReq\.on\("error"/g,
      ) || [];
    const guardedOllamaProxyHandlers = ollamaProxyHandlers.filter((handler) =>
      handler.includes("this.guardProxyResponseLifecycle"),
    );

    expect(ollamaProxyHandlers.length).toBeGreaterThanOrEqual(3);
    expect(guardedOllamaProxyHandlers.length).toBe(ollamaProxyHandlers.length);
  });

  it("resolves target sessions from POST body, capabilities URL, or query model", () => {
    expect(source).toContain("modelName = extractGatewayModelFromBody(body, req.headers[\"content-type\"]");
    expect(bodySource).toContain("return typeof parsed?.model === \"string\" ? parsed.model : undefined");
    expect(source).toContain('url.match(/^\\/v1\\/models\\/(.+)\\/capabilities');
    expect(source).toContain("new URLSearchParams(url.slice(qIdx))");
    expect(source).toContain('params.get("model")');
  });

  it("keeps multipart image edits binary-safe while still resolving model fields", () => {
    expect(source).toContain("private readBody(req: IncomingMessage): Promise<Buffer>");
    expect(source).toContain("resolve(Buffer.concat(chunks))");
    expect(bodySource).toContain("extractGatewayModelFromBody");
    expect(bodySource).toContain("extractMultipartFormField");
    expect(source).not.toContain("resolve(Buffer.concat(chunks).toString())");
    expect(source).not.toContain("name=\"model\"(?:\\r?\\n");
  });

  it("broadcasts cancel requests without requiring a model field", () => {
    expect(source).toContain('const isCancel = method === "POST" && /\\/cancel\\/?$/.test(url)');
    expect(source).toContain('accepted ? 200 : 404');
    expect(source).toContain('status: "cancelled"');
    expect(source).toContain('{ error: "Request ID not found on any backend" }');
  });
});

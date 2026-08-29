import { randomUUID } from 'crypto'
import { ipcMain } from 'electron'

import {
  buildBenchmarkMessages,
  getBenchmarkProfile,
  type BenchmarkFamilyId,
  type BenchmarkProfile,
  type BenchmarkProfileId,
  type BenchmarkScenario,
  type BenchmarkScenarioKind,
} from '../../shared/benchmarkProfiles'
import { db } from '../database'
import { connectHost, resolveUrl } from '../sessions'
import { getAuthHeaders } from './utils'

/**
 * Benchmark IPC handlers.
 *
 * Peak and representative profiles remain deliberately separate. A peak row
 * is an explicitly labelled best-case microbenchmark; it is never averaged
 * together with long-form or agentic work and presented as general speed.
 */

interface PromptResult {
  label: string
  scenarioId: string
  kind: BenchmarkScenarioKind
  profileId: BenchmarkProfileId
  profileLabel: string
  familyId: BenchmarkFamilyId
  familyLabel: string
  trial: number
  repetitions: number
  ttft: number
  tps: number
  promptTokens: number
  cachedPromptTokens: number
  uncachedPromptTokens: number
  completionTokens: number
  totalTime: number
  decodeTime: number
  ppSpeed: number
  maxTokens: number
  temperature: number
  thinkingDisabled: boolean
  error?: string
}

interface BenchmarkRunOptions {
  flushCache?: boolean
  profileId?: BenchmarkProfileId
}

function usageCachedTokens(usage: any): number {
  return Number(usage?.prompt_tokens_details?.cached_tokens || 0)
}

async function runSingleBenchmark(
  baseUrl: string,
  profile: BenchmarkProfile,
  scenario: BenchmarkScenario,
  trial: number,
  authHeaders: Record<string, string> = {},
): Promise<PromptResult> {
  const fetchStart = Date.now()
  let firstTokenTime: number | null = null
  let lastTokenTime: number | null = null
  let tokenCount = 0
  let promptTokens = 0
  let cachedPromptTokens = 0
  const messages = buildBenchmarkMessages(scenario, randomUUID())
  const requestBody: Record<string, any> = {
    model: 'default',
    messages,
    max_tokens: scenario.maxTokens,
    temperature: scenario.temperature,
    stream: true,
    stream_options: { include_usage: true },
  }
  if (scenario.disableThinking) requestBody.enable_thinking = false

  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders },
    body: JSON.stringify(requestBody),
    signal: AbortSignal.timeout(scenario.timeoutMs),
  })

  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(
      `Benchmark request failed: ${res.status} ${res.statusText}${detail ? ` — ${detail.slice(0, 240)}` : ''}`,
    )
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith('data: ')) continue
      const data = trimmed.slice(6)
      if (data === '[DONE]') continue

      try {
        const parsed = JSON.parse(data)
        let serverUsageThisChunk = false
        if (parsed.usage) {
          promptTokens = Number(parsed.usage.prompt_tokens || promptTokens)
          cachedPromptTokens = usageCachedTokens(parsed.usage)
          if (parsed.usage.completion_tokens != null) {
            tokenCount = Number(parsed.usage.completion_tokens)
            serverUsageThisChunk = true
          }
        }

        const delta = parsed.choices?.[0]?.delta
        const emitted =
          delta?.content || delta?.reasoning_content || delta?.reasoning
        if (emitted) {
          const now = Date.now()
          if (firstTokenTime == null) firstTokenTime = now
          lastTokenTime = now
          if (!serverUsageThisChunk) tokenCount++
        }
      } catch {
        // Ignore non-JSON SSE comments/heartbeats. Malformed terminal payloads
        // still surface through missing usage/zero-rate results below.
      }
    }
  }

  const requestEnd = Date.now()
  const totalTime = (requestEnd - fetchStart) / 1000
  const ttft =
    firstTokenTime == null ? totalTime : (firstTokenTime - fetchStart) / 1000
  const streamedDecodeTime =
    firstTokenTime != null &&
    lastTokenTime != null &&
    lastTokenTime > firstTokenTime
      ? (lastTokenTime - firstTokenTime) / 1000
      : 0
  const fallbackDecodeTime =
    firstTokenTime == null ? totalTime : (requestEnd - firstTokenTime) / 1000
  const decodeTime =
    streamedDecodeTime > 0 ? streamedDecodeTime : fallbackDecodeTime
  const tps =
    streamedDecodeTime > 0 && tokenCount > 1
      ? (tokenCount - 1) / streamedDecodeTime
      : decodeTime > 0.01
        ? tokenCount / decodeTime
        : 0
  const uncachedPromptTokens = Math.max(0, promptTokens - cachedPromptTokens)
  const ppSpeed =
    ttft > 0.001 && uncachedPromptTokens > 0 ? uncachedPromptTokens / ttft : 0

  return {
    label: scenario.label,
    scenarioId: scenario.id,
    kind: scenario.kind,
    profileId: profile.id,
    profileLabel: profile.label,
    familyId: profile.familyId,
    familyLabel: profile.familyLabel,
    trial,
    repetitions: scenario.repetitions,
    ttft,
    tps,
    promptTokens,
    cachedPromptTokens,
    uncachedPromptTokens,
    completionTokens: tokenCount,
    totalTime,
    decodeTime,
    ppSpeed,
    maxTokens: scenario.maxTokens,
    temperature: scenario.temperature,
    thinkingDisabled: scenario.disableThinking,
  }
}

function failedResult(
  profile: BenchmarkProfile,
  scenario: BenchmarkScenario,
  trial: number,
  error: unknown,
): PromptResult {
  return {
    label: scenario.label,
    scenarioId: scenario.id,
    kind: scenario.kind,
    profileId: profile.id,
    profileLabel: profile.label,
    familyId: profile.familyId,
    familyLabel: profile.familyLabel,
    trial,
    repetitions: scenario.repetitions,
    ttft: 0,
    tps: 0,
    promptTokens: 0,
    cachedPromptTokens: 0,
    uncachedPromptTokens: 0,
    completionTokens: 0,
    totalTime: 0,
    decodeTime: 0,
    ppSpeed: 0,
    maxTokens: scenario.maxTokens,
    temperature: scenario.temperature,
    thinkingDisabled: scenario.disableThinking,
    error: error instanceof Error ? error.message : String(error),
  }
}

export function registerBenchmarkHandlers(
  getWindow: () => Electron.BrowserWindow | null,
): void {
  ipcMain.handle(
    'benchmark:run',
    async (
      _,
      sessionId: string,
      endpoint: { host: string; port: number },
      modelPath: string,
      modelName?: string,
      options?: BenchmarkRunOptions,
    ) => {
      const baseUrl = await resolveUrl(
        `http://${connectHost(endpoint.host)}:${endpoint.port}`,
      )
      const authHeaders = getAuthHeaders(sessionId)
      const profile = getBenchmarkProfile(
        options?.profileId || 'peak',
        `${modelName || ''} ${modelPath}`,
      )
      const results: PromptResult[] = []
      const win = getWindow()
      const total = profile.scenarios.reduce(
        (count, scenario) => count + scenario.repetitions,
        0,
      )

      if (options?.flushCache) {
        try {
          const cacheRes = await fetch(`${baseUrl}/v1/cache`, {
            method: 'DELETE',
            headers: authHeaders,
            signal: AbortSignal.timeout(10_000),
          })
          if (cacheRes.ok) {
            console.log('[BENCHMARK] Prefix cache flushed before benchmark run')
          }
        } catch (error: any) {
          console.warn(
            '[BENCHMARK] Cache flush failed (non-fatal):',
            error.message,
          )
        }
      }

      let current = 0
      for (const scenario of profile.scenarios) {
        for (let trial = 1; trial <= scenario.repetitions; trial++) {
          current++
          if (win && !win.isDestroyed()) {
            win.webContents.send('benchmark:progress', {
              sessionId,
              current,
              total,
              label: `${scenario.label} · ${trial}/${scenario.repetitions}`,
            })
          }

          try {
            results.push(
              await runSingleBenchmark(
                baseUrl,
                profile,
                scenario,
                trial,
                authHeaders,
              ),
            )
          } catch (error) {
            results.push(failedResult(profile, scenario, trial, error))
            console.error(
              `[BENCHMARK] ${scenario.label} trial ${trial} failed:`,
              error,
            )
          }
        }
      }

      const benchmark = {
        id: randomUUID(),
        sessionId,
        modelPath,
        modelName,
        resultsJson: JSON.stringify(results),
        createdAt: Date.now(),
      }
      db.saveBenchmark(benchmark)

      return {
        id: benchmark.id,
        profileId: profile.id,
        profileLabel: profile.label,
        familyId: profile.familyId,
        familyLabel: profile.familyLabel,
        disclosure: profile.disclosure,
        results,
        createdAt: benchmark.createdAt,
      }
    },
  )

  ipcMain.handle('benchmark:history', async (_, modelPath?: string) => {
    const benchmarks = db.getBenchmarks(modelPath)
    return benchmarks.map((benchmark) => {
      const results = JSON.parse(benchmark.resultsJson) as PromptResult[]
      const first = results[0]
      return {
        id: benchmark.id,
        sessionId: benchmark.sessionId,
        modelPath: benchmark.modelPath,
        modelName: benchmark.modelName,
        profileId: first?.profileId || 'legacy',
        profileLabel: first?.profileLabel || 'Legacy',
        familyId: first?.familyId || 'generic',
        familyLabel: first?.familyLabel || benchmark.modelName || 'Model',
        results,
        createdAt: benchmark.createdAt,
      }
    })
  })

  ipcMain.handle('benchmark:delete', async (_, id: string) => {
    db.deleteBenchmark(id)
    return { success: true }
  })
}

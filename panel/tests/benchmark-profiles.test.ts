import { describe, expect, it } from 'vitest'

import {
  buildBenchmarkMessages,
  detectBenchmarkFamily,
  getBenchmarkProfile,
} from '../src/shared/benchmarkProfiles'
import { extractBenchmarkMtpSnapshot } from '../src/shared/benchmarkTelemetry'

describe('benchmark peak profiles', () => {
  it.each([
    ['jangq-ai/Qwen3.8-Flash-Next-JANG_4M', 'qwen38-flash-next'],
    ['jangq-ai/Qwen3.8-27B-JANG_4D', 'qwen38-27b'],
    ['DeepSeek-V4-Flash-0731-JANG-CRACK', 'dsv4-flash'],
    ['GLM-5.3-Flash-JANG-Affine', 'glm53-flash'],
    ['qwen4_exp_text', 'qwen38-flash-next'],
    ['glm5_next', 'glm53-flash'],
  ])('detects %s as %s', (identity, family) => {
    expect(detectBenchmarkFamily(identity)).toBe(family)
  })

  it('uses retained tuned prefill shapes for the two measured Qwen families', () => {
    const flash = getBenchmarkProfile('peak', 'Qwen3.8-Flash-Next-JANG_4M')
    const dense = getBenchmarkProfile('peak', 'Qwen3.8-27B-JANG_4D')

    expect(
      flash.scenarios.filter((row) => row.kind === 'prefill'),
    ).toMatchObject([{ targetPromptTokens: 12_000, repetitions: 3 }])
    expect(
      dense.scenarios.filter((row) => row.kind === 'prefill'),
    ).toMatchObject([{ targetPromptTokens: 6_000, repetitions: 3 }])
  })

  it('keeps the headline decode arm explicit, short, deterministic, and thinking-off', () => {
    const profile = getBenchmarkProfile('peak', 'DeepSeek-V4-Flash')
    const decode = profile.scenarios.find((row) => row.kind === 'decode')

    expect(decode).toMatchObject({
      id: 'peak-code-burst',
      maxTokens: 128,
      temperature: 0,
      repetitions: 3,
      disableThinking: true,
    })
    expect(profile.disclosure).toContain('not sustained agentic throughput')
  })

  it('puts a nonce before the prefill body so repeated runs cannot claim a warm prefix', () => {
    const profile = getBenchmarkProfile('peak', 'GLM-5.3-Flash')
    const scenario = profile.scenarios.find((row) => row.kind === 'prefill')!
    const messages = buildBenchmarkMessages(scenario, 'unique-proof')

    expect(messages[0].content.startsWith('[benchmark:unique-proof]')).toBe(
      true,
    )
    expect(messages[0].content.length).toBeGreaterThan(5_000)
  })

  it('keeps representative prompts separate from peak headline rows', () => {
    const profile = getBenchmarkProfile('representative', 'Qwen3.8-27B')

    expect(profile.scenarios).toHaveLength(4)
    expect(profile.scenarios.every((row) => row.repetitions === 1)).toBe(true)
    expect(profile.scenarios.some((row) => row.kind === 'decode')).toBe(false)
  })
})

describe('benchmark MTP telemetry', () => {
  it('accepts only the completed benchmark request telemetry', () => {
    const snapshot = extractBenchmarkMtpSnapshot(
      {
        mtp: {
          runtime_active: true,
          effective_depth: 3,
          effective_depth_source: 'bundle',
          request_policy: 'compatible-only',
        },
        scheduler: {
          batch_generator: {
            last_native_mtp: {
              request_id: 'chatcmpl-own',
              final_depth: 2,
              cycles: 20,
              drafted_tokens: 40,
              accepted_tokens: 34,
              acceptance_rate: 0.85,
              profiled_phase_timing: false,
              timings_ms: { total: 0 },
              forwards: { verify_main: 20, mtp: 40 },
            },
          },
        },
      },
      'chatcmpl-own',
    )

    expect(snapshot).toMatchObject({
      runtimeActive: true,
      effectiveDepth: 3,
      telemetryState: 'engaged',
      telemetryRequestId: 'chatcmpl-own',
      finalDepth: 2,
      draftedTokens: 40,
      acceptedTokens: 34,
      acceptanceRate: 0.85,
      phaseTimingProfiled: false,
      timingsMs: undefined,
    })
  })

  it('does not turn absent model depth into a fake D0', () => {
    const snapshot = extractBenchmarkMtpSnapshot({
      mtp: { runtime_active: false, effective_depth: null },
    })

    expect(snapshot.effectiveDepth).toBeUndefined()
  })

  it('labels another request telemetry as stale instead of attributing it', () => {
    const snapshot = extractBenchmarkMtpSnapshot(
      {
        mtp: { runtime_active: true },
        scheduler: {
          batch_generator: {
            last_native_mtp: { request_id: 'chatcmpl-other' },
          },
        },
      },
      'chatcmpl-own',
    )

    expect(snapshot.telemetryState).toBe('stale')
    expect(snapshot.finalDepth).toBeUndefined()
  })

  it('records an exact request skip without claiming MTP engagement', () => {
    const snapshot = extractBenchmarkMtpSnapshot(
      {
        mtp: { runtime_active: true, effective_depth: 2 },
        scheduler: {
          batch_generator: {
            last_native_mtp_skip: {
              request_id: 'chatcmpl-own',
              reason: 'concurrent_request',
            },
          },
        },
      },
      'chatcmpl-own',
    )

    expect(snapshot).toMatchObject({
      telemetryState: 'skipped',
      skipReason: 'concurrent_request',
    })
  })
})

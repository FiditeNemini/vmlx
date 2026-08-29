export type BenchmarkMtpTelemetryState =
  | 'engaged'
  | 'skipped'
  | 'stale'
  | 'missing'

export interface BenchmarkMtpSnapshot {
  runtimeActive?: boolean
  effectiveDepth?: number
  effectiveDepthSource?: string
  requestPolicy?: string
  telemetryState: BenchmarkMtpTelemetryState
  telemetryRequestId?: string
  finalDepth?: number
  cycles?: number
  draftedTokens?: number
  acceptedTokens?: number
  acceptanceRate?: number
  forwards?: Record<string, number>
  timingsMs?: Record<string, number>
  phaseTimingProfiled?: boolean
  skipReason?: string
}

function finiteNumber(value: unknown): number | undefined {
  if (value == null || value === '') return undefined
  const number = Number(value)
  return Number.isFinite(number) ? number : undefined
}

function requestId(payload: any): string | undefined {
  const value = payload?.request_id
  return typeof value === 'string' && value ? value : undefined
}

/** Extract only request-exact MTP telemetry from the post-request health row. */
export function extractBenchmarkMtpSnapshot(
  health: any,
  expectedRequestId?: string,
): BenchmarkMtpSnapshot {
  const modelMtp = health?.mtp || {}
  const batch = health?.scheduler?.batch_generator || {}
  const engagement = batch.last_native_mtp
  const skip = batch.last_native_mtp_skip
  const engagementId = requestId(engagement)
  const skipId = requestId(skip)
  const base: BenchmarkMtpSnapshot = {
    runtimeActive:
      typeof modelMtp.runtime_active === 'boolean'
        ? modelMtp.runtime_active
        : undefined,
    effectiveDepth: finiteNumber(modelMtp.effective_depth),
    effectiveDepthSource:
      typeof modelMtp.effective_depth_source === 'string'
        ? modelMtp.effective_depth_source
        : undefined,
    requestPolicy:
      typeof modelMtp.request_policy === 'string'
        ? modelMtp.request_policy
        : undefined,
    telemetryState: 'missing',
  }

  if (expectedRequestId && engagementId === expectedRequestId) {
    const phaseTimingProfiled = engagement.profiled_phase_timing === true
    return {
      ...base,
      telemetryState: 'engaged',
      telemetryRequestId: engagementId,
      finalDepth: finiteNumber(engagement.final_depth),
      cycles: finiteNumber(engagement.cycles),
      draftedTokens: finiteNumber(engagement.drafted_tokens),
      acceptedTokens: finiteNumber(engagement.accepted_tokens),
      acceptanceRate: finiteNumber(engagement.acceptance_rate),
      forwards: engagement.forwards,
      timingsMs: phaseTimingProfiled ? engagement.timings_ms : undefined,
      phaseTimingProfiled,
    }
  }
  if (expectedRequestId && skipId === expectedRequestId) {
    return {
      ...base,
      telemetryState: 'skipped',
      telemetryRequestId: skipId,
      skipReason:
        typeof skip.reason === 'string'
          ? skip.reason
          : typeof skip.skip_reason === 'string'
            ? skip.skip_reason
            : 'unspecified',
    }
  }
  if (engagementId || skipId) {
    return {
      ...base,
      telemetryState: 'stale',
      telemetryRequestId: engagementId || skipId,
    }
  }
  return base
}

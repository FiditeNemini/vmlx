export interface Dsv4ActivationQatStatus {
  requested?: boolean
  runtime_requested?: boolean
  effective?: boolean
  observed?: boolean | null
  attested?: boolean
  supported?: boolean
  matches_request?: boolean
  paths?: {
    attention_kv_and_pool_e4m3?: boolean | null
    indexer_hadamard128_fp4_e2m1?: boolean | null
  }
  fused_kernels?: {
    e4m3_available?: boolean
    indexer_available?: boolean
  }
}

export interface Dsv4ActivationQatDisplay {
  requestedEffective: string
  observedAttestation: string
  paths: string
  fusedKernels: string
}

function onOff(value: boolean | undefined): string {
  if (value == null) return 'unknown'
  return value ? 'on' : 'off'
}

function observedState(value: boolean | null | undefined): string {
  if (value == null) return 'pending'
  return onOff(value)
}

function availability(value: boolean | undefined): string {
  if (value == null) return 'unknown'
  return value ? 'ready' : 'missing'
}

/**
 * Keep the QAT control plane separate from runtime observation. Before the
 * first model forward, health intentionally reports no observation, so that
 * state must remain pending rather than being presented as a mismatch.
 */
export function describeDsv4ActivationQat(
  status: Dsv4ActivationQatStatus,
): Dsv4ActivationQatDisplay {
  const attested = status.attested === true
  const match = attested
    ? status.matches_request === true ? 'match' : 'mismatch'
    : 'match pending'

  return {
    requestedEffective: `saved ${onOff(status.requested)} · runtime ${onOff(status.runtime_requested)} · effective ${onOff(status.effective)} · ${status.supported === false ? 'unsupported' : status.supported === true ? 'supported' : 'support unknown'}`,
    observedAttestation: `observed ${observedState(status.observed)} · ${attested ? 'attested' : 'not attested'} · ${match}`,
    paths: `KV/pool ${observedState(status.paths?.attention_kv_and_pool_e4m3)} · indexer ${observedState(status.paths?.indexer_hadamard128_fp4_e2m1)}`,
    fusedKernels: `E4M3 ${availability(status.fused_kernels?.e4m3_available)} · indexer ${availability(status.fused_kernels?.indexer_available)}`,
  }
}

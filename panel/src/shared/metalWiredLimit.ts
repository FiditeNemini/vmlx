export const metalWiredLimitHelpText =
  'An out-of-memory error or SIGKILL alone does not establish that the wired limit is the cause. Check the measured Metal memory status. Close other apps or use a smaller quant if memory is exhausted. A wired-limit increase needs measured hardware headroom; do not set it equal to physical RAM. Any sysctl change requires an admin password and resets after reboot.'

/**
 * 2026-08-17: `block` was REMOVED from this union deliberately.
 *
 * This preflight must never be able to refuse a load. Loading big models on
 * unified memory is what this app is for, and a size-vs-freemem heuristic does
 * not get veto power over a model the user explicitly chose. Deleting the
 * variant (rather than leaving it unused at the call site) means no future
 * edit can reintroduce a refusal without changing this type on purpose.
 */
export type LargeModelMemoryPreflight = {
  action: 'ok' | 'warn'
  message: string
}

function formatGb(bytes: number): string {
  return (bytes / 1e9).toFixed(1)
}

export function classifyLargeModelMemoryPreflight(input: {
  modelSizeBytes: number
  availableBytes: number
  totalBytes: number
}): LargeModelMemoryPreflight {
  const { modelSizeBytes, availableBytes, totalBytes } = input
  if (modelSizeBytes <= 0 || availableBytes <= 0 || totalBytes <= 0) {
    return { action: 'ok', message: '' }
  }

  const modelGB = formatGb(modelSizeBytes)
  const availGB = formatGb(availableBytes)
  const usagePercent = ((totalBytes - availableBytes) / totalBytes) * 100
  const hugeModel = modelSizeBytes >= 50e9
  const effectivelyNoFreeRam = availableBytes < 2e9 && usagePercent >= 98

  if (hugeModel && effectivelyNoFreeRam) {
    return {
      action: 'warn',
      message: appendMetalWiredLimitGuidance(
        `Memory warning: only ${availGB} GB free for a ~${modelGB} GB model (${usagePercent.toFixed(0)}% used). Loading anyway; if it fails with an out of memory error, closing other apps or stopping running vMLX sessions frees memory.`
      ),
    }
  }

  if (modelSizeBytes > availableBytes * 0.9) {
    return {
      action: 'warn',
      message: `Memory warning: Model requires ~${modelGB} GB but only ${availGB} GB free. Loading may cause system instability or swap.`,
    }
  }

  if (modelSizeBytes > availableBytes * 0.7) {
    return {
      action: 'warn',
      message: `Note: Model (~${modelGB} GB) will use most available memory. KV cache may be limited.`,
    }
  }

  return { action: 'ok', message: '' }
}

export interface MeasuredMetalMemory {
  version: 1
  available: true
  source: 'mlx_active_working_set'
  pid: number
  measured_at_ms: number
  active_bytes: number
  limit_bytes: number
  device_limit_bytes: number
  physical_bytes: number | null
  reason: 'measurement' | 'guard_rejection'
  threshold_pct?: number
}

const finitePositive = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && n > 0

/** Unknown/stale/cross-PID measurements are not warnings. No file-size oracle. */
export function readMeasuredMetalMemory(value: unknown, pid: number, now = Date.now()): MeasuredMetalMemory | null {
  const v = value as MeasuredMetalMemory | undefined
  if (!v || v.version !== 1 || v.available !== true || v.source !== 'mlx_active_working_set' ||
      !Number.isSafeInteger(pid) || pid <= 0 || v.pid !== pid ||
      !finitePositive(v.measured_at_ms) || now - v.measured_at_ms > 30_000 || v.measured_at_ms > now + 1000 ||
      typeof v.active_bytes !== 'number' || !Number.isFinite(v.active_bytes) || v.active_bytes < 0 ||
      !finitePositive(v.limit_bytes) || !finitePositive(v.device_limit_bytes) || v.limit_bytes > v.device_limit_bytes ||
      !['measurement', 'guard_rejection'].includes(v.reason)) return null
  if (v.reason === 'guard_rejection' && (!finitePositive(v.threshold_pct) || v.threshold_pct > 100)) return null
  return { ...v, physical_bytes: finitePositive(v.physical_bytes) ? v.physical_bytes : null }
}

export function isMeasuredMetalPressure(v: MeasuredMetalMemory): boolean {
  // A load/health reading warns only at the actual limit. The earlier guard
  // threshold is eligible only when the engine really rejected a request.
  const fraction = v.reason === 'guard_rejection' ? v.threshold_pct! / 100 : 1
  return v.active_bytes >= v.limit_bytes * fraction
}

/** Optional manual suggestion, never executed. sysctl uses MiB, not decimal MB. */
export function measuredWiredLimitCommand(v: MeasuredMetalMemory): string | null {
  if (!isMeasuredMetalPressure(v) || !v.physical_bytes || v.limit_bytes !== v.device_limit_bytes) return null
  const MiB = 1024 ** 2
  const GiB = 1024 ** 3
  // Leave at least 8 GiB or 10% for the OS/apps; round DOWN inside the reserve.
  const ceiling = v.physical_bytes - Math.max(8 * GiB, v.physical_bytes * 0.1)
  const desired = v.active_bytes + Math.max(4 * GiB, v.active_bytes * 0.05)
  const mb = Math.floor(Math.min(ceiling, desired) / MiB / 512) * 512
  if (mb * MiB <= Math.max(v.device_limit_bytes, v.active_bytes)) return null
  return `sudo sysctl iogpu.wired_limit_mb=${mb}`
}

export interface MetalMemoryNotice {
  id: string
  sessionId: string
  modelName: string
  measurement: MeasuredMetalMemory
  command: string | null
}

const METAL_WIRED_LIMIT_RE =
  /(?:Command buffer execution failed|Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|Metal OOM|kernel-panic risk|SIGKILL|likely out of memory|out of memory)/i

export function appendMetalWiredLimitGuidance(message: string): string {
  if (!METAL_WIRED_LIMIT_RE.test(message)) return message
  if (message.includes(metalWiredLimitHelpText)) return message
  return `${message}\n\nMetal wired-memory limit help: ${metalWiredLimitHelpText}`
}

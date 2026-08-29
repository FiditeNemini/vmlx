export const metalWiredLimitCommand = 'sudo sysctl iogpu.wired_limit_mb=120000'

export const metalWiredLimitHelpText =
  `If a large model hits Metal OOM, SIGKILL, or kernel-panic risk, macOS may need a tuned Metal wired-memory limit. Example for large-memory Macs: ${metalWiredLimitCommand}. Do not set it equal to physical RAM; leave OS/WindowServer/app headroom. On 128GB Macs, 115000-120000 MB is usually safer than 128000 MB. The command requires an admin password and resets after reboot.`

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

/**
 * Wired-limit preflight: compare the model's resident footprint against the
 * user's EFFECTIVE Metal wired limit and, when the load would brush or exceed
 * it, produce a visual recommendation with the exact sysctl command.
 *
 * ADVISORY ONLY by type: there is no block/refuse arm and none may be added.
 * macOS/Metal fail loudly on a genuine miss; this exists so the user learns
 * the one-line fix BEFORE that happens instead of after.
 */
export type WiredLimitPreflight =
  | { action: 'ok' }
  | {
      action: 'recommend'
      recommendedMb: number
      command: string
      message: string
      detail: string
    }

// Apple Silicon's default GPU wired limit (iogpu.wired_limit_mb=0) is ~75% of
// physical RAM (measured: 128 GB Mac -> ~96 GiB Metal working-set budget).
const DEFAULT_WIRED_FRACTION = 0.75
// Runtime overhead on top of raw weights: KV/native state, scratch arenas,
// vision tower activations, allocator slack. Deliberately modest — this tunes
// when we ADVISE, never whether we load.
const RUNTIME_OVERHEAD_BYTES = 6e9
// Never recommend wiring everything: leave OS/WindowServer headroom.
const MIN_OS_HEADROOM_MB = 8000

export function classifyWiredLimitPreflight(input: {
  modelSizeBytes: number
  wiredLimitMb: number // current sysctl value; 0 or negative = macOS default
  totalBytes: number
}): WiredLimitPreflight {
  const { modelSizeBytes, wiredLimitMb, totalBytes } = input
  if (modelSizeBytes <= 0 || totalBytes <= 0) return { action: 'ok' }

  const totalMb = Math.floor(totalBytes / 1e6)
  const effectiveLimitMb =
    wiredLimitMb > 0 ? wiredLimitMb : Math.floor(totalMb * DEFAULT_WIRED_FRACTION)
  const neededMb = Math.ceil((modelSizeBytes + RUNTIME_OVERHEAD_BYTES) / 1e6)
  if (neededMb <= effectiveLimitMb) return { action: 'ok' }

  const cappedMb = Math.max(effectiveLimitMb, Math.min(totalMb - MIN_OS_HEADROOM_MB, neededMb))
  // Round up to the nearest 1000 MB for a clean, memorable command.
  const recommendedMb = Math.ceil(cappedMb / 1000) * 1000
  if (recommendedMb <= effectiveLimitMb) return { action: 'ok' }

  const command = `sudo sysctl iogpu.wired_limit_mb=${recommendedMb}`
  const modelGb = (modelSizeBytes / 1e9).toFixed(1)
  const limitGb = (effectiveLimitMb / 1000).toFixed(0)
  return {
    action: 'recommend',
    recommendedMb,
    command,
    message:
      `This model (~${modelGb} GB) is close to or above your Metal wired-memory ` +
      `limit (~${limitGb} GB${wiredLimitMb > 0 ? '' : ', the macOS default'}).`,
    detail:
      `Loading will proceed, but it may fail with a Metal out-of-memory error or ` +
      `run degraded. To raise the limit, run this in Terminal (admin password ` +
      `required; resets after reboot):\n\n${command}\n\nDo not set it equal to ` +
      `physical RAM — leave OS and app headroom.`,
  }
}

const METAL_WIRED_LIMIT_RE =
  /(?:Command buffer execution failed|Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|Metal OOM|kernel-panic risk|SIGKILL|likely out of memory|out of memory)/i

export function appendMetalWiredLimitGuidance(message: string): string {
  if (!METAL_WIRED_LIMIT_RE.test(message)) return message
  if (message.includes(metalWiredLimitCommand)) return message
  return `${message}\n\nMetal wired-memory limit help: ${metalWiredLimitHelpText}`
}

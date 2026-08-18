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

const METAL_WIRED_LIMIT_RE =
  /(?:Command buffer execution failed|Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|Metal OOM|kernel-panic risk|SIGKILL|likely out of memory|out of memory)/i

export function appendMetalWiredLimitGuidance(message: string): string {
  if (!METAL_WIRED_LIMIT_RE.test(message)) return message
  if (message.includes(metalWiredLimitCommand)) return message
  return `${message}\n\nMetal wired-memory limit help: ${metalWiredLimitHelpText}`
}

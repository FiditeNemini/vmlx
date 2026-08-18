import { describe, expect, it } from 'vitest'

import {
  appendMetalWiredLimitGuidance,
  classifyLargeModelMemoryPreflight,
  metalWiredLimitCommand,
  metalWiredLimitHelpText,
} from '../src/shared/metalWiredLimit'

describe('Metal wired-memory limit guidance', () => {
  it('includes the sudo sysctl command in user-facing help text', () => {
    expect(metalWiredLimitHelpText).toContain(metalWiredLimitCommand)
    expect(metalWiredLimitHelpText).toContain('115000-120000 MB')
    expect(metalWiredLimitHelpText).toContain('Do not set it equal to physical RAM')
    expect(metalWiredLimitHelpText).toContain('admin password')
    expect(metalWiredLimitHelpText).toContain('resets after reboot')
  })

  it('annotates the exact Metal command-buffer OOM startup error', () => {
    const message =
      'Process exited before becoming ready: libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)'

    const annotated = appendMetalWiredLimitGuidance(message)

    expect(annotated).toContain(message)
    expect(annotated).toContain('Metal wired-memory limit help')
    expect(annotated).toContain(metalWiredLimitCommand)
  })

  it('does not annotate unrelated backend errors', () => {
    expect(appendMetalWiredLimitGuidance('Process exited before becoming ready: ImportError: mlx missing')).toBe(
      'Process exited before becoming ready: ImportError: mlx missing',
    )
  })

  it('annotates the SIGKILL load failure users see when macOS kills the engine', () => {
    const message =
      'Process was killed (SIGKILL) — likely out of memory. Try a smaller/more quantized model, reduce cache size, or close other apps.'

    const annotated = appendMetalWiredLimitGuidance(message)

    expect(annotated).toContain('Metal wired-memory limit help')
    expect(annotated).toContain('115000-120000 MB')
  })

  // 2026-08-17: this used to assert action === 'block'. It now asserts the
  // opposite, and that inversion is the point.
  //
  // The refusal it pinned shipped in v1.6.28-v1.6.33 and made dots3-note
  // (101.6 GB, estimated 103.6 GB) unloadable for EVERY 128 GB user, along with
  // any other bundle over ~76 GB whose model_type the ratio table did not
  // recognise. Eric never asked for a RAM limit. Even at 0.1 GB free the
  // preflight only warns — the OS and the Metal allocator report a real OOM
  // recoverably, and that is a better failure than refusing to try.
  it('NEVER blocks, even with essentially no free RAM — it only warns', () => {
    const result = classifyLargeModelMemoryPreflight({
      modelSizeBytes: 132.3e9,
      availableBytes: 0.1e9,
      totalBytes: 137e9,
    })

    expect(result.action).toBe('warn')
    expect(result.message).not.toContain('Refusing to start')
    expect(result.message).toContain('0.1 GB free')
    expect(result.message).toContain(metalWiredLimitCommand)
  })

  it('keeps ordinary large-model overcommit as a warning, not a hard block', () => {
    const result = classifyLargeModelMemoryPreflight({
      modelSizeBytes: 132.3e9,
      availableBytes: 104.3e9,
      totalBytes: 137e9,
    })

    expect(result.action).toBe('warn')
    expect(result.message).toContain('Memory warning')
  })
})

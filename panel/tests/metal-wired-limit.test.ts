import { describe, expect, it } from 'vitest'

import {
  appendMetalWiredLimitGuidance,
  classifyLargeModelMemoryPreflight,
  classifyWiredLimitPreflight,
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

describe('classifyWiredLimitPreflight', () => {
  const GB = 1e9

  it('recommends the exact sysctl command when a GLM-5.3-class model exceeds the stock limit on 128GB', () => {
    // 128 GiB Mac (137.4e9 bytes), sysctl at 0 (macOS default ~= 75% =>
    // ~103,050 MB ~= 96 GiB budget), GLM-5.3-Flash bundle = 102.4e9 bytes
    // on disk (the real du measurement).
    const r = classifyWiredLimitPreflight({
      modelSizeBytes: 102.4 * GB,
      wiredLimitMb: 0,
      totalBytes: 137.4 * GB,
    })
    expect(r.action).toBe('recommend')
    expect(r.command).toMatch(/^sudo sysctl iogpu\.wired_limit_mb=\d+$/)
    expect(r.recommendedMb).toBeGreaterThan(96000)
    // never recommend wiring the whole machine
    expect(r.recommendedMb).toBeLessThanOrEqual(137400 - 8000 + 1000)
    expect(r.detail).toContain(r.command)
    expect(r.detail).toContain('Loading will proceed')
  })

  it('stays quiet when the model fits the current limit comfortably', () => {
    const r = classifyWiredLimitPreflight({
      modelSizeBytes: 18 * GB,
      wiredLimitMb: 0,
      totalBytes: 137.4 * GB,
    })
    expect(r.action).toBe('ok')
  })

  it('respects a user-raised limit that already covers the model', () => {
    const r = classifyWiredLimitPreflight({
      modelSizeBytes: 102.4 * GB,
      wiredLimitMb: 120000,
      totalBytes: 137.4 * GB,
    })
    expect(r.action).toBe('ok')
  })

  it('recommends raising a user limit that is set but too low', () => {
    const r = classifyWiredLimitPreflight({
      modelSizeBytes: 102.4 * GB,
      wiredLimitMb: 90000,
      totalBytes: 137.4 * GB,
    })
    expect(r.action).toBe('recommend')
    expect(r.recommendedMb).toBeGreaterThan(90000)
  })

  it('has no block arm by type — the union is ok|recommend only', () => {
    const r = classifyWiredLimitPreflight({
      modelSizeBytes: 300 * GB,
      wiredLimitMb: 0,
      totalBytes: 137.4 * GB,
    })
    expect(['ok', 'recommend']).toContain(r.action)
  })
})

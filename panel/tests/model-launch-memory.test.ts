import { mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'
import { tmpdir } from 'os'
import { describe, expect, it } from 'vitest'
import {
  MODEL_LAUNCH_FIXED_OVERHEAD_BYTES,
  estimateModelFileBytes,
  estimateModelLaunchResidentBytes,
  estimateMacReclaimableMemoryBytesFromVmStat,
  launchResidentProfileForModelType,
  modelLaunchReserveWarning,
  unsafeModelLaunchOverrideEnabled,
  unsafeModelLaunchReason,
} from '../src/main/modelLaunchMemory'

function makeModelDir(name: string): string {
  const dir = join(tmpdir(), `vmlx-model-launch-${name}-${Date.now()}-${Math.random().toString(16).slice(2)}`)
  mkdirSync(dir, { recursive: true })
  return dir
}

function writeBytes(path: string, bytes: number): void {
  writeFileSync(path, Buffer.alloc(bytes, 1))
}

describe('model launch memory admission', () => {
  it('recursively counts model files for launch estimates', () => {
    const dir = makeModelDir('recursive')
    mkdirSync(join(dir, 'sub'))
    writeBytes(join(dir, 'model.safetensors'), 11)
    writeBytes(join(dir, 'sub', 'extra.safetensors'), 7)

    expect(estimateModelFileBytes(dir)).toBe(18)
  })

  it('residency is keyed to the FAMILY loader, not the weight format', () => {
    // Measured (vmmap 2026-08-11): weights are COPIED into dirty Metal
    // buffers on every ordinary load — a jang_config.json does NOT make a
    // bundle lazy. The old format-keyed ×0.7 under-admitted full-resident
    // affine/mxtq/mxfp8 bundles by ~45%.
    expect(launchResidentProfileForModelType('nemotron_h')).toEqual({ ratio: 1.0, streamsWeights: false })
    expect(launchResidentProfileForModelType('minimax_m2')).toEqual({ ratio: 1.0, streamsWeights: false })
    expect(launchResidentProfileForModelType('llama')).toEqual({ ratio: 1.0, streamsWeights: false })
    expect(launchResidentProfileForModelType('')).toEqual({ ratio: 1.0, streamsWeights: false })
    // Expert-streaming loaders genuinely stay below file size (DSV4-Flash
    // measured ~0.26×, MM3 measured ~0.80×).
    expect(launchResidentProfileForModelType('deepseek_v4')).toEqual({ ratio: 0.7, streamsWeights: true })
    expect(launchResidentProfileForModelType('minimax_m3_vl')).toEqual({ ratio: 0.85, streamsWeights: true })
  })

  it('estimates full-resident bundles at fileBytes + fixed overhead', () => {
    const dir = makeModelDir('affine-jang')
    writeFileSync(join(dir, 'jang_config.json'), '{}')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'nanbeige' }))
    writeBytes(join(dir, 'weights.safetensors'), 100)

    const gib = 1024 ** 3
    expect(estimateModelLaunchResidentBytes(dir, 10 * gib, 128 * gib))
      .toBe(10 * gib + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES)
  })

  it('keeps the streaming discount for expert-streaming families', () => {
    const dir = makeModelDir('dsv4-flash')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'deepseek_v4' }))
    writeBytes(join(dir, 'weights.safetensors'), 100)

    const gib = 1024 ** 3
    expect(estimateModelLaunchResidentBytes(dir, 100 * gib, 128 * gib))
      .toBe(Math.round(100 * gib * 0.7) + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES)
  })

  it('caps the estimate at total machine memory', () => {
    const dir = makeModelDir('huge')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'llama' }))

    const gib = 1024 ** 3
    expect(estimateModelLaunchResidentBytes(dir, 200 * gib, 128 * gib)).toBe(128 * gib)
  })

  it('does not hard-block when only the 16 GiB reserve is tight', () => {
    const gib = 1024 ** 3
    const launchResident = 73.2 * gib
    const available = 88.3 * gib

    expect(unsafeModelLaunchReason(launchResident, available, {})).toBeNull()
    expect(modelLaunchReserveWarning(launchResident, available)).toContain('16.0 GB system safety headroom')
  })

  it('still hard-blocks when estimated launch resident exceeds free RAM', () => {
    const gib = 1024 ** 3

    expect(unsafeModelLaunchReason(91 * gib, 88.3 * gib, {})).toContain('exceeds currently free RAM')
  })

  it('credits bounded reclaimable macOS page cache for EVERY launch', () => {
    // Clean page cache is droppable no matter what is allocating; the credit
    // used to be gated on the (refuted) lazy-mmap JANG theory, which denied
    // it to the full-resident bundles that need it most.
    const gib = 1024 ** 3

    expect(
      unsafeModelLaunchReason(60.2 * gib, 51.3 * gib, {}, {
        reclaimableBytes: 12 * gib,
        totalBytes: 128 * gib,
      }),
    ).toBeNull()
    // The credit stays bounded at 15% of total RAM — a hot file cache cannot
    // talk the preflight into admitting a model the box cannot hold.
    expect(
      unsafeModelLaunchReason(80 * gib, 51.3 * gib, {}, {
        reclaimableBytes: 60 * gib,
        totalBytes: 128 * gib,
      }),
    ).toContain('exceeds currently free RAM')
  })

  it('parses conservative reclaimable memory from macOS vm_stat output', () => {
    const vmStat = [
      'Mach Virtual Memory Statistics: (page size of 16384 bytes)',
      'Pages inactive:                            1499093.',
      'Pages purgeable:                             19462.',
      'Pages speculative:                         1746605.',
      'File-backed pages:                         5995950.',
    ].join('\n')

    expect(estimateMacReclaimableMemoryBytesFromVmStat(vmStat)).toBe((1_499_093 + 19_462) * 16_384)
  })

  it('accepts canonical and legacy unsafe launch override env names', () => {
    const gib = 1024 ** 3

    expect(unsafeModelLaunchOverrideEnabled({ VMLX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBe(true)
    expect(unsafeModelLaunchOverrideEnabled({ VMLINUX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBe(true)
    expect(unsafeModelLaunchReason(91 * gib, 88.3 * gib, { VMLX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBeNull()
    expect(unsafeModelLaunchReason(91 * gib, 88.3 * gib, { VMLINUX_ALLOW_UNSAFE_MODEL_LAUNCH: '1' })).toBeNull()
  })
})

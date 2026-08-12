import { mkdirSync, readFileSync, writeFileSync } from 'fs'
import { join, resolve } from 'path'
import { tmpdir } from 'os'
import { describe, expect, it } from 'vitest'
import {
  MODEL_LAUNCH_FIXED_OVERHEAD_BYTES,
  estimateModelFileBytes,
  estimateModelLaunchAdmissionBytes,
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
    // `admissionRatio` is what a REFUSAL may use; `ratio` stays the
    // conservative bound behind the graded warnings. For full-resident loads
    // there is nothing to discount, so the two are equal.
    expect(launchResidentProfileForModelType('nemotron_h')).toEqual({ ratio: 1.0, admissionRatio: 1.0, streamsWeights: false })
    expect(launchResidentProfileForModelType('minimax_m2')).toEqual({ ratio: 1.0, admissionRatio: 1.0, streamsWeights: false })
    expect(launchResidentProfileForModelType('llama')).toEqual({ ratio: 1.0, admissionRatio: 1.0, streamsWeights: false })
    expect(launchResidentProfileForModelType('')).toEqual({ ratio: 1.0, admissionRatio: 1.0, streamsWeights: false })
    // Expert-streaming loaders genuinely stay below file size (DSV4-Flash
    // measured ~0.26×, MM3 measured ~0.80×), and those measurements — not the
    // padded bounds — are what admission is allowed to refuse on.
    expect(launchResidentProfileForModelType('deepseek_v4')).toEqual({ ratio: 0.7, admissionRatio: 0.35, streamsWeights: true })
    expect(launchResidentProfileForModelType('minimax_m3_vl')).toEqual({ ratio: 0.85, admissionRatio: 0.8, streamsWeights: true })
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

describe('launch admission is actually wired into the launch path', () => {
  const sessions = readFileSync(
    resolve(__dirname, '../src/main/sessions.ts'),
    'utf8',
  )

  it('consults unsafeModelLaunchReason before spawning the engine', () => {
    // These three were written, unit-tested, and never called from src/ — the
    // launch path relied solely on classifyLargeModelMemoryPreflight, whose
    // block arm requires modelSizeBytes >= 50GB AND availableBytes < 2GB AND
    // >= 98% used. MEASURED on a 137GB box at 42.2GB free: a 91.9GB model and a
    // 73.4GB model both classified as merely "warn", so the app would start a
    // 92GB model into 42GB of free RAM.
    expect(sessions).toContain('unsafeModelLaunchReason(')
    expect(sessions).toContain('modelLaunchReserveWarning(')
    expect(sessions).toContain('unsafeModelLaunchOverrideHint()')
    expect(sessions).toMatch(
      /import \{[^}]*unsafeModelLaunchReason[^}]*\} from '\.\/modelLaunchMemory'/s,
    )
  })

  it('refuses BEFORE the graded warnings, and names the override', () => {
    const refusalAt = sessions.indexOf('unsafeModelLaunchReason(')
    const classifyAt = sessions.indexOf('classifyLargeModelMemoryPreflight({')
    expect(refusalAt).toBeGreaterThan(-1)
    expect(classifyAt).toBeGreaterThan(-1)
    expect(refusalAt).toBeLessThan(classifyAt)

    const block = sessions.slice(refusalAt, classifyAt)
    expect(block).toContain('Refusing to start this model')
    expect(block).toContain('unsafeModelLaunchOverrideHint()')
    expect(block).toContain("status: 'error'")
    expect(block).toContain('throw new Error')
  })

  it('the refusal threshold is the one that actually fits, not near-death', () => {
    const gib = 1024 ** 3
    // 92GB model, 42GB free, 20GB reclaimable on a 137GB box -> must refuse.
    expect(
      unsafeModelLaunchReason(92 * gib, 42 * gib, {}, {
        reclaimableBytes: 20 * gib,
        totalBytes: 137 * gib,
      }),
    ).toContain('exceeds currently free RAM')
    // A 21GB model in the same conditions must still be admitted.
    expect(
      unsafeModelLaunchReason(21 * gib, 42 * gib, {}, {
        reclaimableBytes: 20 * gib,
        totalBytes: 137 * gib,
      }),
    ).toBeNull()
  })
})

describe('a refusal must be keyed to the measurement, not the worst case', () => {
  /**
   * `unsafeModelLaunchReason` was wired to the CONSERVATIVE residency estimate,
   * so the safety upper bound became a mandatory admission threshold. For a
   * 96 GB DSV4-Flash bundle that is 0.7 * 96 + 2 = 69.2 GB — refused on a box
   * with ~62.8 GB effective free, even though the same bundle measures ~25 GB
   * resident there and serves requests fine. The bound is for warnings; the
   * block needs the measurement.
   */
  const GB = 1e9

  function dsv4Dir(): string {
    const dir = makeModelDir('dsv4-admission')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'deepseek_v4' }))
    return dir
  }

  function plainDir(): string {
    const dir = makeModelDir('plain-admission')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'llama' }))
    return dir
  }

  it('admits a 96 GB DSV4-Flash bundle with ~62.8 GB effective free', () => {
    const dir = dsv4Dir()
    const admission = estimateModelLaunchAdmissionBytes(dir, 96 * GB, 128 * GB)
    // 0.35 * 96 + 2 = 35.6 GB, comfortably inside the real headroom.
    expect(admission).toBeLessThan(40 * GB)
    expect(unsafeModelLaunchReason(admission, 62.8 * GB, {})).toBeNull()
  })

  it('still refuses a DSV4 bundle that genuinely cannot fit', () => {
    const dir = dsv4Dir()
    const admission = estimateModelLaunchAdmissionBytes(dir, 400 * GB, 128 * GB)
    expect(unsafeModelLaunchReason(admission, 20 * GB, {})).not.toBeNull()
  })

  it('does not loosen admission for ordinary full-resident bundles', () => {
    const dir = plainDir()
    const resident = estimateModelLaunchResidentBytes(dir, 92 * GB, 128 * GB)
    const admission = estimateModelLaunchAdmissionBytes(dir, 92 * GB, 128 * GB)
    expect(admission).toBe(resident)
    // 92 GB of dirty Metal buffers into 42 GB free must still be refused.
    expect(unsafeModelLaunchReason(admission, 42 * GB, {})).not.toBeNull()
  })

  it('never lets the block threshold exceed the warning threshold', () => {
    for (const modelType of ['deepseek_v4', 'minimax_m3', 'llama', 'gemma3', '']) {
      const profile = launchResidentProfileForModelType(modelType)
      expect(profile.admissionRatio).toBeLessThanOrEqual(profile.ratio)
    }
  })
})

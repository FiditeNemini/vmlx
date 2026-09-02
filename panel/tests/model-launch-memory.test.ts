import { mkdirSync, readFileSync, writeFileSync } from 'fs'
import { join, resolve } from 'path'
import { tmpdir } from 'os'
import { describe, expect, it } from 'vitest'
import { classifyWiredLimitPreflight } from '../src/shared/metalWiredLimit'
import {
  MODEL_LAUNCH_FIXED_OVERHEAD_BYTES,
  estimateModelFileBytes,
  estimateModelLaunchAdmissionBytes,
  estimateModelLaunchResidentBytes,
  estimateMacReclaimableMemoryBytesFromVmStat,
  launchResidentProfileForModel,
  launchResidentProfileForModelType,
  modelLaunchReserveWarning,
  unsafeModelLaunchOverrideEnabled,
  unsafeModelLaunchReason,
} from '../src/main/modelLaunchMemory'
import { classifyLargeModelMemoryPreflight } from '../src/shared/metalWiredLimit'

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

  it('still COMPUTES the memory estimate and surfaces it', () => {
    expect(sessions).toContain('unsafeModelLaunchReason(')
    expect(sessions).toContain('modelLaunchReserveWarning(')
    expect(sessions).toMatch(
      /import \{[^}]*unsafeModelLaunchReason[^}]*\} from '\.\/modelLaunchMemory'/s,
    )
  })

  /**
   * 2026-08-17 — THE REGRESSION THIS FILE NOW EXISTS TO PREVENT.
   *
   * Shipped in 1.6.33: a ~101 GB bundle was REFUSED on a 128 GB box with
   * "estimated launch resident ~103.6 GB exceeds currently free RAM 78.2 GB".
   * The 103.6 figure is fileBytes x 1.0 — i.e. the per-family residency ratio
   * did not recognise the bundle and fell back to the worst case, so a FAILED
   * DETECTION became a user-facing wall on the exact models this app exists to
   * run. Eric never asked for a RAM limit.
   *
   * The estimate may inform. It may not refuse. These assertions are
   * deliberately phrased as absence-of-refusal so that reintroducing a block
   * fails the suite.
   */
  it('NEVER refuses a launch on a memory estimate', () => {
    const estimateAt = sessions.indexOf('unsafeModelLaunchReason(')
    const spawnAt = sessions.indexOf('await this.ensureOwnedSessionPortAvailable(session)')
    expect(estimateAt).toBeGreaterThan(-1)
    expect(spawnAt).toBeGreaterThan(estimateAt)

    // Everything between computing the estimate and actually launching.
    const preflight = sessions.slice(estimateAt, spawnAt)

    // No refusal, by any spelling.
    expect(preflight).not.toContain('Refusing to start this model')
    expect(preflight).not.toContain('Refusing to start this large model')
    // No throwing, and no marking the session failed before it ever ran.
    expect(preflight).not.toContain('throw new Error')
    expect(preflight).not.toContain("status: 'error'")
    // An override env is not consent — it is a wall with a secret door.
    expect(preflight).not.toContain('unsafeModelLaunchOverrideHint()')
    // It must still SAY something: advice is the whole remaining job.
    expect(preflight).toMatch(/session:log/)
  })

  it('the preflight classifier cannot even express a refusal', () => {
    const shared = readFileSync(
      resolve(__dirname, '../src/shared/metalWiredLimit.ts'),
      'utf8',
    )
    // `block` was deleted from the union so no call site can reintroduce it.
    expect(shared).toMatch(/action:\s*'ok'\s*\|\s*'warn'\s*$/m)
    expect(shared).not.toContain("action: 'block'")
    expect(sessions).not.toContain("memoryPreflight.action === 'block'")

    // And the classifier's worst case is a warning that still starts.
    const gib = 1024 ** 3
    const verdict = classifyLargeModelMemoryPreflight({
      modelSizeBytes: 101 * gib,
      availableBytes: 1 * gib,
      totalBytes: 128 * gib,
    })
    expect(verdict.action).toBe('warn')
    expect(verdict.message).not.toContain('Refusing')
  })

  it('a 101 GB bundle on a 128 GB box is admitted even undetected', () => {
    // The exact shape of the 1.6.33 failure: unknown model_type -> 1.0x ratio.
    const dir = makeModelDir('unknown-family-101gb')
    writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'some_new_family' }))

    const gib = 1024 ** 3
    const admission = estimateModelLaunchAdmissionBytes(dir, 101 * gib, 128 * gib)
    // The estimate is still pessimistic — that is fine, it is only advice now.
    expect(admission).toBeGreaterThan(100 * gib)

    // What must NOT happen: that estimate stopping the launch. The launch path
    // has no refusal at all, which the preceding test pins at the source level.
    const verdict = classifyLargeModelMemoryPreflight({
      modelSizeBytes: 101 * gib,
      availableBytes: 78.2 * gib,
      totalBytes: 128 * gib,
    })
    expect(verdict.action).not.toBe('block')
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

describe('an unidentifiable model still yields a usable estimate', () => {
  /**
   * The catch path returned a hand-written `{ ratio, streamsWeights }` that
   * omitted `admissionRatio`. `modelFileBytes * undefined` is NaN, so the
   * admission estimate was NaN and the refusal message read "estimated launch
   * resident ~NaN GB". Caught by `npm run typecheck`, which the vitest run does
   * not perform — so the suite was green with this shipped.
   */
  const GB = 1e9

  it('falls back to the full-resident profile with both ratios set', () => {
    const dir = makeModelDir('no-config')
    // No config.json at all — readFileSync throws.
    const profile = launchResidentProfileForModel(dir)
    expect(profile.ratio).toBe(1.0)
    expect(profile.admissionRatio).toBe(1.0)
    expect(profile.streamsWeights).toBe(false)
  })

  it('produces a finite admission estimate, not NaN', () => {
    const dir = makeModelDir('bad-config')
    writeFileSync(join(dir, 'config.json'), '{ this is not json')
    const admission = estimateModelLaunchAdmissionBytes(dir, 50 * GB, 128 * GB)
    expect(Number.isFinite(admission)).toBe(true)
    expect(admission).toBe(50 * GB + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES)
    // And the refusal it feeds must be a real decision, not "~NaN GB".
    expect(unsafeModelLaunchReason(admission, 100 * GB, {})).toBeNull()
    expect(unsafeModelLaunchReason(admission, 10 * GB, {})).toContain('GB')
    expect(unsafeModelLaunchReason(admission, 10 * GB, {})).not.toContain('NaN')
  })
})

describe('qwen4_exp SSD-backed PLE residency profile', () => {
  it('discounts the SSD-resident PLE from the launch estimate', () => {
    const profile = launchResidentProfileForModelType('qwen4_exp')
    expect(profile.streamsWeights).toBe(true)
    expect(profile.ratio).toBe(0.85)
    expect(profile.admissionRatio).toBe(0.78)
    expect(profile.admissionRatio).toBeLessThanOrEqual(profile.ratio)
  })

  it('keeps a real 4M-sized bundle under the wired-limit recommendation', () => {
    // 96 GiB bundle on a 128 GB box with an ~111 GB wired limit: counting the
    // whole bundle (105 GB estimate + 6 GB overhead) tripped the modal; the
    // measured PLE discount keeps it clearly below the limit.
    const fileBytes = 103.1e9
    const estimated = Math.round(fileBytes * 0.85) + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES
    const preflight = classifyWiredLimitPreflight({
      modelSizeBytes: estimated,
      wiredLimitMb: 111000,
      totalBytes: 137e9,
    })
    expect(preflight.action).toBe('ok')
  })
})

import { execFileSync } from 'child_process'
import { readdirSync, readFileSync, statSync } from 'fs'
import { join } from 'path'

/** Estimate local model file bytes recursively. Returns 0 if unknown. */
export function estimateModelFileBytes(modelPath: string): number {
  try {
    const entries = readdirSync(modelPath, { withFileTypes: true })
    let totalBytes = 0
    for (const entry of entries) {
      const fullPath = join(modelPath, entry.name)
      if (entry.isDirectory()) {
        totalBytes += estimateModelFileBytes(fullPath)
      } else if (entry.isFile()) {
        totalBytes += statSync(fullPath).size
      }
    }
    return totalBytes
  } catch (_) {
    return 0
  }
}

/**
 * Launch residency is a property of the FAMILY's loader, not of the weight
 * format. Measured (vmmap + RSS, 2026-08-11 and 2026-07-12 live loads):
 *
 *  - Every ordinary load COPIES weights into dirty Metal buffers. vmmap on a
 *    3.09 GB affine bundle right after load: mapped-file resident 400 KB,
 *    IOAccelerator(graphics) 2.9 GB dirty. Affine/MXTQ/MXFP8/plain all
 *    measured resident ≈ fileBytes + ~0.9 GB process overhead; MM2.7 (86 GB
 *    MoE) measured 0.96×. Dirty Metal memory is NOT evictable — the old
 *    "lazy mmap, ~70% resident" model under-admitted these by ~45%.
 *  - deepseek_v4 flash MoE streams experts from SSD on demand: 96 GB bundle
 *    measured ~25 GB resident. 0.7 is kept as the conservative bound.
 *  - minimax_m3* manages expert residency (F128): 84 GB measured 67 GB
 *    resident (0.80×); 0.85 keeps a margin.
 */
export const MODEL_LAUNCH_FIXED_OVERHEAD_BYTES = 2e9

export interface LaunchResidentProfile {
  /**
   * Conservative upper bound on the fraction of file bytes resident after load.
   * Used for the graded WARNINGS — erring high there costs the user a scary
   * message, which is cheap.
   */
  ratio: number
  /**
   * Fraction to use when REFUSING a launch outright. Never above `ratio`.
   *
   * A safety upper bound is not sound as a mandatory admission threshold. For
   * the streaming families the conservative number is far above what was
   * actually measured — DSV4-Flash was measured at ~0.26× — so refusing on 0.7
   * turned away a 96 GB DSV4-Flash bundle at an estimated 69.2 GB when its real
   * residency is ~25 GB and it runs fine on this box. Refusal has to be keyed
   * to the measurement, with a margin, not to the worst case.
   */
  admissionRatio: number
  /** True when the loader keeps weights on SSD and streams them on demand. */
  streamsWeights: boolean
}

export function launchResidentProfileForModelType(modelType: string): LaunchResidentProfile {
  const type = String(modelType || '').toLowerCase()
  // DSV4-Flash: measured ~0.26×. 0.35 keeps ~35% headroom over the measurement
  // for expert residency growing during a long session, and is still half the
  // conservative warning bound.
  if (type.startsWith('deepseek_v4')) {
    return { ratio: 0.7, admissionRatio: 0.35, streamsWeights: true }
  }
  // MM3: measured 0.80×; 0.85 is the warning margin, so refuse on the measured
  // value rather than the padded one.
  if (type.startsWith('minimax_m3')) {
    return { ratio: 0.85, admissionRatio: 0.8, streamsWeights: true }
  }
  // Ordinary loads COPY weights into dirty Metal buffers, so file size IS the
  // residency. Nothing to discount.
  return { ratio: 1.0, admissionRatio: 1.0, streamsWeights: false }
}

export function launchResidentProfileForModel(modelPath: string): LaunchResidentProfile {
  try {
    const config = JSON.parse(readFileSync(join(modelPath, 'config.json'), 'utf8'))
    return launchResidentProfileForModelType(String(config?.model_type || ''))
  } catch (_) {
    // An unreadable config.json means we cannot identify the family, so assume
    // the full-resident profile — the conservative choice for both numbers.
    //
    // This must go through launchResidentProfileForModelType rather than
    // hand-rolling the object: written out by hand it omitted admissionRatio,
    // and `modelFileBytes * undefined` is NaN, so estimateModelLaunchAdmissionBytes
    // returned NaN and the refusal read "estimated launch resident ~NaN GB".
    return launchResidentProfileForModelType('')
  }
}

export function estimateModelLaunchResidentBytes(modelPath: string, modelFileBytes: number, totalBytes: number): number {
  if (modelFileBytes <= 0) return 0
  const { ratio } = launchResidentProfileForModel(modelPath)
  const residentBytes = Math.round(modelFileBytes * ratio) + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES
  return totalBytes > 0 ? Math.min(residentBytes, totalBytes) : residentBytes
}

/**
 * The estimate a REFUSAL may be based on — measurement-keyed, not worst-case.
 *
 * Always <= estimateModelLaunchResidentBytes for the same inputs, so the
 * warning thresholds stay strictly more cautious than the block threshold and
 * a model can never be refused without first having been warned about.
 */
export function estimateModelLaunchAdmissionBytes(modelPath: string, modelFileBytes: number, totalBytes: number): number {
  if (modelFileBytes <= 0) return 0
  const { admissionRatio } = launchResidentProfileForModel(modelPath)
  const residentBytes = Math.round(modelFileBytes * admissionRatio) + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES
  return totalBytes > 0 ? Math.min(residentBytes, totalBytes) : residentBytes
}

export function formatGb(bytes: number): string {
  return (bytes / 1e9).toFixed(1)
}

export const MODEL_LAUNCH_SAFETY_RESERVE_BYTES = 16 * 1024 ** 3
export const UNSAFE_MODEL_LAUNCH_OVERRIDE_ENV = 'VMLX_ALLOW_UNSAFE_MODEL_LAUNCH'
export const UNSAFE_MODEL_LAUNCH_LEGACY_OVERRIDE_ENV = 'VMLINUX_ALLOW_UNSAFE_MODEL_LAUNCH'

export interface LaunchAdmissionOptions {
  reclaimableBytes?: number
  totalBytes?: number
}

export function unsafeModelLaunchOverrideEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env[UNSAFE_MODEL_LAUNCH_OVERRIDE_ENV] === '1' || env[UNSAFE_MODEL_LAUNCH_LEGACY_OVERRIDE_ENV] === '1'
}

export function unsafeModelLaunchOverrideHint(): string {
  return `${UNSAFE_MODEL_LAUNCH_OVERRIDE_ENV}=1 (or legacy ${UNSAFE_MODEL_LAUNCH_LEGACY_OVERRIDE_ENV}=1)`
}

export function unsafeModelLaunchReason(
  modelSizeBytes: number,
  availableBytes: number,
  env: NodeJS.ProcessEnv = process.env,
  options: LaunchAdmissionOptions = {},
): string | null {
  if (modelSizeBytes <= 0 || availableBytes <= 0) return null
  if (unsafeModelLaunchOverrideEnabled(env)) return null
  const effectiveAvailableBytes = effectiveLaunchAvailableBytes(availableBytes, options)
  if (modelSizeBytes <= effectiveAvailableBytes) return null
  return (
    `estimated launch resident ~${formatGb(modelSizeBytes)} GB exceeds ` +
    `currently free RAM ${formatGb(availableBytes)} GB`
  )
}

export function effectiveLaunchAvailableBytes(
  availableBytes: number,
  options: LaunchAdmissionOptions = {},
): number {
  // Clean page-cache pages are droppable no matter what kind of allocation is
  // asking — this credit used to be gated on the (refuted) lazy-mmap JANG
  // theory, but freemem() under-counting reclaimable cache applies to every
  // launch equally. Keep the credit bounded so a hot file cache can't talk the
  // preflight into admitting a model the box can't actually hold.
  if (availableBytes <= 0) return availableBytes
  const reclaimableBytes = Math.max(0, options.reclaimableBytes || 0)
  if (reclaimableBytes <= 0) return availableBytes
  const totalBytes = Math.max(0, options.totalBytes || 0)
  const reclaimableCap = totalBytes > 0 ? Math.round(totalBytes * 0.15) : reclaimableBytes
  return availableBytes + Math.min(reclaimableBytes, reclaimableCap)
}

export function estimateMacReclaimableMemoryBytesFromVmStat(output: string): number {
  const pageSizeMatch = output.match(/page size of\s+(\d+)\s+bytes/i)
  const pageSize = pageSizeMatch ? Number(pageSizeMatch[1]) : 4096
  if (!Number.isFinite(pageSize) || pageSize <= 0) return 0

  const pagesFor = (label: string): number => {
    const match = output.match(new RegExp(`${label}:\\s+(\\d+)\\.`, 'i'))
    if (!match) return 0
    const pages = Number(match[1])
    return Number.isFinite(pages) && pages > 0 ? pages : 0
  }

  // Node's os.freemem() already includes free/speculative memory on macOS.
  // Count only conservative extra reclaimable pages so lazy-mmap launches do
  // not get blocked by cache-heavy but low-pressure systems.
  return (pagesFor('Pages inactive') + pagesFor('Pages purgeable')) * pageSize
}

export function estimateMacReclaimableMemoryBytes(): number {
  if (process.platform !== 'darwin') return 0
  try {
    const output = execFileSync('vm_stat', [], {
      encoding: 'utf8',
      timeout: 1000,
    })
    return estimateMacReclaimableMemoryBytesFromVmStat(output)
  } catch (_) {
    return 0
  }
}

export function modelLaunchReserveWarning(modelSizeBytes: number, availableBytes: number): string | null {
  if (modelSizeBytes <= 0 || availableBytes <= 0) return null
  const requiredFreeBytes = modelSizeBytes + MODEL_LAUNCH_SAFETY_RESERVE_BYTES
  if (requiredFreeBytes <= availableBytes) return null
  return (
    `estimated launch resident ~${formatGb(modelSizeBytes)} GB leaves less ` +
    `than 16.0 GB system safety headroom (${formatGb(availableBytes)} GB free)`
  )
}

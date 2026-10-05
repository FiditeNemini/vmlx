/**
 * DFlash2 block drafters (z-lab) for external speculative decoding.
 *
 * Only Qwen3.8-27B has a published DFlash2 drafter among the Qwen3.8 MTP
 * families (z-lab/Qwen3.8-27B-DFlash2); Flash-Next affine/JANGH have none and
 * keep native MTP. Measured 2026-10-05 on an M5 Max (served, greedy, novel
 * prompts, AR on the same engine): 27B JANG_2D/4D/6D prose 1.15x/1.55x/2.07x,
 * code 2.05x/2.58x/3.32x of AR.
 */

export const QWEN38_27B_DFLASH2_REPO = 'z-lab/Qwen3.8-27B-DFlash2'

/** Qwen3.8 27B by name/path (keeps Qwen3.6-27B, whose drafter is DFlash v1, apart). */
export function isQwen38Dense27b(modelIdentity: string | undefined | null): boolean {
  const text = String(modelIdentity || '')
  return /qwen[-_ ]?3[._-]?8[^/\\]*?27b/i.test(text)
}

/** A drafter path that the engine will run on the DFlash2 lane. */
export function looksLikeDflash2Drafter(path: string | undefined | null): boolean {
  return /dflash2/i.test(String(path || ''))
}

/** config.json content check — the authoritative test when the file is readable. */
export function configDeclaresDflash2(config: unknown): boolean {
  const arch = (config as { architectures?: unknown } | null)?.architectures
  return Array.isArray(arch) && arch.some(a => a === 'DFlash2DraftModel')
}

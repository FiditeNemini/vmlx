import { existsSync, readdirSync, readFileSync } from 'fs'
import { join } from 'path'

/**
 * True when a drafter config.json declares DSpark: openbmb's MiniCPM-V-4.7 drafters (`Qwen3DSparkModel`,
 * `dflash_config.projector_type: "dspark"`) and SpecForge's Naive N0.5 drafter (`DSparkDraftModel`).
 */
export function configDeclaresDspark(cfg: any): boolean {
  if (!cfg || typeof cfg !== 'object') return false
  const archs: string[] = Array.isArray(cfg.architectures) ? cfg.architectures.map(String) : []
  if (archs.some(a => /dspark/i.test(a))) return true
  const projector = cfg.dflash_config?.projector_type ?? cfg.projector_type
  return String(projector || '').toLowerCase() === 'dspark'
}

/**
 * DSpark default for a bundle: ON whenever the bundle ships a drafter (Eric 2026-10-13: "check model files if it has
 * or not ... enabled if it has it and if not don't use it"). The jang_config `speculative.default` stamp is
 * informational only; the engine applies the same rule (`--dspark auto`, models/mllm._load_dspark_runtime).
 */
export function bundledDsparkDefault(modelPath: string | undefined | null): boolean {
  return findBundledDsparkDrafter(modelPath) !== null
}

/**
 * A DSpark drafter shipped INSIDE the model bundle (convention `<bundle>/dspark`; any immediate subfolder whose
 * config.json declares DSpark). The engine loads it with the model and runs it on the single-request lane; bundles
 * without one return null and launch exactly as before.
 */
export function findBundledDsparkDrafter(modelPath: string | undefined | null): string | null {
  const root = String(modelPath || '').trim()
  if (!root) return null
  let entries: string[] = []
  try {
    entries = readdirSync(root)
  } catch {
    return null
  }
  const ordered = [...entries].sort((a, b) => (a === 'dspark' ? -1 : b === 'dspark' ? 1 : a.localeCompare(b)))
  for (const entry of ordered) {
    if (entry.startsWith('.')) continue
    const cfg = join(root, entry, 'config.json')
    try {
      if (existsSync(cfg) && configDeclaresDspark(JSON.parse(readFileSync(cfg, 'utf8')))) return join(root, entry)
    } catch {
      /* not a drafter */
    }
  }
  return null
}

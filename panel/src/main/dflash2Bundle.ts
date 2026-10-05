import { existsSync, readdirSync, readFileSync } from 'fs'
import { join } from 'path'
import { configDeclaresDflash2 } from '../shared/dflash2Drafter'

/**
 * A DFlash2 drafter shipped INSIDE the model bundle: any immediate subfolder
 * whose config.json declares DFlash2DraftModel (convention: `<bundle>/dflash2`).
 * Bundles without one return null and launch exactly as before.
 */
export function findBundledDflash2Drafter(modelPath: string | undefined | null): string | null {
  const root = String(modelPath || '').trim()
  if (!root) return null
  let entries: string[] = []
  try {
    entries = readdirSync(root)
  } catch {
    return null
  }
  const ordered = [...entries].sort((a, b) => (a === 'dflash2' ? -1 : b === 'dflash2' ? 1 : a.localeCompare(b)))
  for (const entry of ordered) {
    if (entry.startsWith('.')) continue
    const cfg = join(root, entry, 'config.json')
    try {
      if (existsSync(cfg) && configDeclaresDflash2(JSON.parse(readFileSync(cfg, 'utf8')))) {
        return join(root, entry)
      }
    } catch {
      /* not a drafter */
    }
  }
  return null
}

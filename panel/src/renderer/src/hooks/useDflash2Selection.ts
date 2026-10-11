import { useEffect, useState } from 'react'
import { looksLikeDflash2Drafter } from '../../../shared/dflash2Drafter'

/** Use the launcher's local config-first decision, including authoritative false. */
export function useDflash2Selection(path: string): boolean {
  const [result, setResult] = useState<{ path: string; value: boolean } | null>(null)
  useEffect(() => {
    let active = true
    if (!path || typeof window.api.models.classifyDflash2 !== 'function') return
    window.api.models.classifyDflash2(path).then(value => {
      if (active) setResult({ path, value })
    }).catch(() => { /* retain name fallback when metadata is unavailable */ })
    return () => { active = false }
  }, [path])
  return result?.path === path ? result.value : looksLikeDflash2Drafter(path)
}

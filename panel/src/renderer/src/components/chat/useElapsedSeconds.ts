import { useEffect, useState } from 'react'

/**
 * Whole seconds elapsed since `startAt` (a real producer timestamp), ticking
 * once per second while `active`. Returns null when there is no start time:
 * a legacy row without timestamps shows an activity indicator only, never an
 * invented number. Frozen at the last value once `active` turns false.
 */
export function useElapsedSeconds(startAt: number | undefined, active: boolean): number | null {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active || startAt == null) return
    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [active, startAt])
  if (startAt == null || !Number.isFinite(startAt)) return null
  return Math.max(0, Math.floor((now - startAt) / 1000))
}

export function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  const minutes = Math.floor(ms / 60_000)
  const seconds = Math.round((ms % 60_000) / 1000)
  return `${minutes}m ${seconds}s`
}

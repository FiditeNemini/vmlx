import { useSyncExternalStore } from 'react'

const listeners = new Set<() => void>()
let intervalId: ReturnType<typeof setInterval> | null = null
let tick = 0

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  if (intervalId === null) {
    intervalId = setInterval(() => {
      tick++
      listeners.forEach((l) => l())
    }, 30_000)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0 && intervalId !== null) {
      clearInterval(intervalId)
      intervalId = null
    }
  }
}

/** Keeps a relative-time label fresh while the component stays mounted.
 *  One shared 30s interval drives all subscribers; the snapshot is the
 *  formatted string, so memoized components only re-render when the
 *  label text actually changes. */
export function useRelativeTime(timestamp: number, format: (ts: number) => string): string {
  const getSnapshot = (): string => format(timestamp)
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}

/** Re-render trigger for components that format many timestamps inline
 *  (lists mapping over items where a per-item hook is not possible). */
export function useRelativeTimeTick(): number {
  const getSnapshot = (): number => tick
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}

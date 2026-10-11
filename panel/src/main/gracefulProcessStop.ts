export interface GracefulProcessStop {
  isAlive: () => boolean
  terminate: () => void
  forceKill: () => void
  onSlow: (signal: AbortSignal) => Promise<'wait' | 'force'>
  graceMs?: number
  retryMs?: number
  pollMs?: number
  forceExitMs?: number
}

/** A grace deadline asks the user; it never authorizes discarding pending writes. */
export async function stopProcessGracefully(options: GracefulProcessStop): Promise<void> {
  const { isAlive, terminate, forceKill, onSlow } = options
  const pollMs = options.pollMs ?? 100
  const delay = () => new Promise<void>(resolve => setTimeout(resolve, pollMs))
  if (!isAlive()) return
  terminate()
  let deadline = Date.now() + (options.graceMs ?? 10_000)
  while (isAlive()) {
    if (Date.now() < deadline) {
      await delay()
      continue
    }
    const controller = new AbortController()
    const watcher = setInterval(() => {
      if (!isAlive()) controller.abort()
    }, pollMs)
    let decision: 'wait' | 'force'
    try {
      decision = await onSlow(controller.signal)
    } finally {
      clearInterval(watcher)
    }
    if (!isAlive()) return
    if (decision === 'force') {
      forceKill()
      const forcedDeadline = Date.now() + (options.forceExitMs ?? 5_000)
      while (isAlive() && Date.now() < forcedDeadline) await delay()
      if (isAlive()) throw new Error('Server process did not exit after force stop')
      return
    }
    deadline = Date.now() + (options.retryMs ?? 30_000)
  }
}

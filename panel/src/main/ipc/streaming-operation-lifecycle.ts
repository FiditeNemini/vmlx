export interface StreamingOperationResult {
  success: boolean
  cancelled?: boolean
  error?: string
}

export interface KillableProcess {
  kill(signal?: NodeJS.Signals | number): boolean
}

export interface StreamingOperationHandle<TProcess extends KillableProcess> {
  process: TProcess
  cancelled: boolean
  settled: boolean
  forceKillTimer: ReturnType<typeof setTimeout> | null
}

type Schedule = (
  callback: () => void,
  delayMs: number,
) => ReturnType<typeof setTimeout>

/**
 * Owns exactly one long-running Developer Tools child process.
 *
 * Keeping cancellation and settlement on the handle prevents a late close/error
 * from an older child from clearing or completing a newer operation. It also
 * gives the renderer a durable terminal result after navigation.
 */
export class StreamingOperationLifecycle<TProcess extends KillableProcess> {
  private active: StreamingOperationHandle<TProcess> | null = null
  private terminalResult: StreamingOperationResult | null = null

  get isRunning(): boolean {
    return this.active !== null
  }

  get lastResult(): StreamingOperationResult | null {
    return this.terminalResult
  }

  start(process: TProcess): StreamingOperationHandle<TProcess> {
    if (this.active) throw new Error('Another operation is already running')
    const handle: StreamingOperationHandle<TProcess> = {
      process,
      cancelled: false,
      settled: false,
      forceKillTimer: null,
    }
    this.active = handle
    this.terminalResult = null
    return handle
  }

  cancel(schedule: Schedule = setTimeout): boolean {
    const handle = this.active
    if (!handle) return false
    if (handle.cancelled) return true

    handle.cancelled = true
    try { handle.process.kill('SIGTERM') } catch { /* child already exited */ }
    handle.forceKillTimer = schedule(() => {
      if (handle.settled) return
      try { handle.process.kill('SIGKILL') } catch { /* child already exited */ }
    }, 3000)
    return true
  }

  finish(
    handle: StreamingOperationHandle<TProcess>,
    result: StreamingOperationResult,
  ): StreamingOperationResult | null {
    if (handle.settled) return null
    handle.settled = true
    if (handle.forceKillTimer) clearTimeout(handle.forceKillTimer)

    const terminal = handle.cancelled
      ? { success: false, cancelled: true, error: 'Cancelled' }
      : result
    if (this.active === handle) this.active = null
    this.terminalResult = terminal
    return terminal
  }
}

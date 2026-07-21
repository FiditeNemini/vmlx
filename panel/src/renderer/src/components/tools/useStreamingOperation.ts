import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_LOG_LINES = 2000

export interface StreamingOperationCompletion {
  success: boolean
  cancelled?: boolean
  error?: string
}

export interface StreamingOperationDescriptor {
  command: string
  args: string[]
}

interface StreamingOperationResult {
  running: boolean
  logLines: string[]
  wasCancelled: boolean
  completion: StreamingOperationCompletion | null
  operation: StreamingOperationDescriptor | null
  /** Start a streaming operation. Returns the IPC result and accumulated log lines. */
  start: (ipcCall: () => Promise<any>) => Promise<{ ipcResult: any; allLines: string[] }>
  cancel: () => Promise<void>
}

/**
 * Shared hook for ModelDoctor and ModelConverter streaming operations.
 * Handles mounted guards, event listener lifecycle, log accumulation, and cancel.
 *
 * @param onLogUpdate Optional callback called with accumulated lines on each log event (e.g., for parsing).
 */
export function useStreamingOperation(onLogUpdate?: (lines: string[]) => void): StreamingOperationResult {
  const [running, setRunning] = useState(false)
  const [logLines, setLogLines] = useState<string[]>([])
  const [wasCancelled, setWasCancelled] = useState(false)
  const [completion, setCompletion] = useState<StreamingOperationCompletion | null>(null)
  const [operation, setOperation] = useState<StreamingOperationDescriptor | null>(null)
  const mountedRef = useRef(true)
  const cleanupRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    mountedRef.current = true

    // Subscribe before taking the snapshot so a completion between the snapshot
    // request and listener setup cannot strand the restored UI in "running".
    let completionReceived = false
    const unsubLog = window.api.developer.onLog((data: any) => {
      if (!mountedRef.current) return
      const lines = data.data.split('\n').filter((l: string) => l.length > 0)
      setLogLines(prev => {
        const updated = [...prev, ...lines].slice(-MAX_LOG_LINES)
        onLogUpdate?.(updated)
        return updated
      })
    })
    const unsubComplete = window.api.developer.onComplete((result: StreamingOperationCompletion) => {
      completionReceived = true
      if (!mountedRef.current) return
      setRunning(false)
      setCompletion(result)
      setWasCancelled(!!result.cancelled)
      unsubLog()
      unsubComplete()
      cleanupRef.current = null
    })
    cleanupRef.current = () => { unsubLog(); unsubComplete() }

    // Restore both active and completed operations. Long conversions may finish
    // while the user is on another page; their logs/result must not disappear.
    window.api.developer.getBufferedLogs().then((result: {
      lines: string[]
      running: boolean
      result: StreamingOperationCompletion | null
      operation: StreamingOperationDescriptor | null
    }) => {
      if (!mountedRef.current) return
      const restoredLines = result.lines.slice(-MAX_LOG_LINES)
      if (restoredLines.length > 0) {
        setLogLines(restoredLines)
        onLogUpdate?.(restoredLines)
      }
      if (!completionReceived) {
        setRunning(result.running)
        setCompletion(result.result)
        setWasCancelled(!!result.result?.cancelled)
        setOperation(result.operation)
        if (!result.running) {
          unsubLog()
          unsubComplete()
          cleanupRef.current = null
        }
      }
    }).catch(() => {
      if (!completionReceived) setRunning(false)
    })

    return () => {
      mountedRef.current = false
      cleanupRef.current?.()
    }
  }, [])

  const start = useCallback(async (ipcCall: () => Promise<any>): Promise<{ ipcResult: any; allLines: string[] }> => {
    setRunning(true)
    setLogLines([])
    setWasCancelled(false)
    setCompletion(null)

    const allLines: string[] = []

    const unsubLog = window.api.developer.onLog((data: any) => {
      if (!mountedRef.current) return
      const lines = data.data.split('\n').filter((l: string) => l.length > 0)
      allLines.push(...lines)
      setLogLines(prev => {
        const updated = [...prev, ...lines].slice(-MAX_LOG_LINES)
        onLogUpdate?.(updated)
        return updated
      })
    })

    cleanupRef.current = () => { unsubLog() }

    let ipcResult: any
    try {
      ipcResult = await ipcCall()
    } catch (error) {
      ipcResult = {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      }
    } finally {
      unsubLog()
      cleanupRef.current = null
      if (mountedRef.current) {
        setRunning(false)
        setCompletion(ipcResult ?? { success: false, error: 'Operation returned no result' })
        if (ipcResult?.cancelled) setWasCancelled(true)
      }
    }
    return { ipcResult, allLines }
  }, [onLogUpdate])

  const cancel = useCallback(async () => {
    try {
      await window.api.developer.cancelOp()
    } catch { /* main process unavailable */ }
  }, [])

  return { running, logLines, wasCancelled, completion, operation, start, cancel }
}

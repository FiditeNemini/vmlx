/**
 * ONE definition of "this child-process stream error is the peer going away,
 * not a fault".
 *
 * This pair existed as FIVE copies — process-manager.ts, engine-manager.ts,
 * tools/executor.ts, ipc/developer.ts and ipc/models.ts — the largest
 * duplication in the panel. At extraction time four were byte-identical and
 * ipc/models.ts differed only in quote style and semicolons, so there was no
 * behavioural divergence yet.
 *
 * It is still the worst kind of rule to leave duplicated, because it decides
 * whether an error is SWALLOWED. Add a newly-observed disconnect shape to one
 * copy and the other four keep reporting it as a fault; narrow one copy and
 * that surface starts hiding real failures. Neither direction announces itself.
 *
 * The matcher is deliberately generous about where the disconnect hides: some
 * runtimes wrap it in `cause`/`reason`/`error`/`detail`, and AggregateError
 * puts it in `errors`, so it recurses through all of those.
 */

export function isExpectedChildProcessStreamDisconnectError(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException)?.code
  const message = String((err as Error)?.message || '').toLowerCase()
  const cause = (err as any)?.cause
  const wrappedDisconnects = [
    cause,
    (err as any)?.reason,
    (err as any)?.error,
    (err as any)?.detail,
  ].filter(Boolean)
  const nestedErrors = Array.isArray((err as any)?.errors) ? (err as any).errors : []
  return (
    code === 'EPIPE' ||
    code === 'ECONNRESET' ||
    code === 'ERR_STREAM_DESTROYED' ||
    code === 'ERR_STREAM_WRITE_AFTER_END' ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedChildProcessStreamDisconnectError(nested)) ||
    nestedErrors.some((nested: unknown) => isExpectedChildProcessStreamDisconnectError(nested))
  )
}

/**
 * A dev/proof Electron process is often itself attached to a parent harness by
 * stdout/stderr pipes. On macOS, writing after that parent exits can surface
 * as `write EIO` rather than `EPIPE`. Scope that exception to the Electron
 * process stdio streams: a filesystem EIO must still reach the crash handler.
 */
export function isExpectedProcessStdioDisconnectError(err: unknown): boolean {
  if (isExpectedChildProcessStreamDisconnectError(err)) return true
  const code = String((err as NodeJS.ErrnoException)?.code || '')
  const message = String((err as Error)?.message || '').trim()
  return code === 'EIO' && /^(?:read|write) EIO$/i.test(message)
}

export function attachProcessStdioErrorGuard(
  stream: NodeJS.ReadableStream | NodeJS.WritableStream | null | undefined,
  onUnexpected: (err: Error) => void,
): void {
  stream?.on('error', (err: Error) => {
    if (isExpectedProcessStdioDisconnectError(err)) return
    onUnexpected(err)
  })
}

/**
 * Attach an `error` listener that reports only unexpected failures.
 *
 * An unhandled `error` on a child stream takes the whole main process down, so
 * every child stream needs one of these — the point is that the "expected"
 * judgement is shared rather than re-decided per call site.
 */
export function attachChildProcessStreamErrorGuard(
  stream: NodeJS.ReadableStream | null | undefined,
  onUnexpected: (err: Error) => void,
): void {
  stream?.on('error', (err: Error) => {
    if (isExpectedChildProcessStreamDisconnectError(err)) return
    onUnexpected(err)
  })
}

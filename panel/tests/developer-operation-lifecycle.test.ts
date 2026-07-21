import { describe, expect, it, vi } from 'vitest'
import { StreamingOperationLifecycle } from '../src/main/ipc/streaming-operation-lifecycle'

class FakeProcess {
  signals: Array<NodeJS.Signals | number | undefined> = []
  kill(signal?: NodeJS.Signals | number): boolean {
    this.signals.push(signal)
    return true
  }
}

describe('Developer Tools streaming operation lifecycle', () => {
  it('persists a terminal result after the child exits', () => {
    const lifecycle = new StreamingOperationLifecycle<FakeProcess>()
    const handle = lifecycle.start(new FakeProcess())

    expect(lifecycle.finish(handle, { success: true })).toEqual({ success: true })
    expect(lifecycle.isRunning).toBe(false)
    expect(lifecycle.lastResult).toEqual({ success: true })
  })

  it('turns cancellation into one durable cancelled result', () => {
    const lifecycle = new StreamingOperationLifecycle<FakeProcess>()
    const process = new FakeProcess()
    const handle = lifecycle.start(process)
    const schedule = vi.fn(() => 1 as unknown as ReturnType<typeof setTimeout>)

    expect(lifecycle.cancel(schedule)).toBe(true)
    expect(process.signals).toEqual(['SIGTERM'])
    expect(lifecycle.finish(handle, { success: false, error: 'Conversion failed' })).toEqual({
      success: false,
      cancelled: true,
      error: 'Cancelled',
    })
    expect(lifecycle.lastResult?.cancelled).toBe(true)
  })

  it('ignores a duplicate error after close instead of emitting twice', () => {
    const lifecycle = new StreamingOperationLifecycle<FakeProcess>()
    const handle = lifecycle.start(new FakeProcess())

    expect(lifecycle.finish(handle, { success: false, error: 'spawn failed' })).not.toBeNull()
    expect(lifecycle.finish(handle, { success: false, error: 'closed' })).toBeNull()
    expect(lifecycle.lastResult?.error).toBe('spawn failed')
  })

  it('does not let a late old-child completion clear the next operation', () => {
    const lifecycle = new StreamingOperationLifecycle<FakeProcess>()
    const oldHandle = lifecycle.start(new FakeProcess())
    lifecycle.finish(oldHandle, { success: true })
    const currentHandle = lifecycle.start(new FakeProcess())

    expect(lifecycle.finish(oldHandle, { success: false, error: 'late close' })).toBeNull()
    expect(lifecycle.isRunning).toBe(true)
    expect(lifecycle.lastResult).toBeNull()
    expect(lifecycle.finish(currentHandle, { success: true })).toEqual({ success: true })
  })

  it('rejects concurrent operations and reports cancel with no child', () => {
    const lifecycle = new StreamingOperationLifecycle<FakeProcess>()
    lifecycle.start(new FakeProcess())
    expect(() => lifecycle.start(new FakeProcess())).toThrow('Another operation is already running')

    const empty = new StreamingOperationLifecycle<FakeProcess>()
    expect(empty.cancel()).toBe(false)
  })
})

import { afterEach, describe, expect, it, vi } from 'vitest'
import { stopProcessGracefully } from '../src/main/gracefulProcessStop'
afterEach(() => vi.useRealTimers())
describe('graceful process stop durability', () => {
  it('does not kill a slow process without an explicit force decision', async () => {
    vi.useFakeTimers()
    let alive = true
    const forceKill = vi.fn()
    const onSlow = vi.fn(async () => 'wait' as const)
    const work = stopProcessGracefully({isAlive:()=>alive, terminate:vi.fn(),forceKill,onSlow})
    await vi.advanceTimersByTimeAsync(15_000)
    expect(onSlow).toHaveBeenCalledOnce()
    expect(forceKill).not.toHaveBeenCalled()
    alive=false;await vi.advanceTimersByTimeAsync(100);await work
  })
  it('closes the pending choice when the process finishes its drain', async () => {
    vi.useFakeTimers()
    let alive=true
    let signal: AbortSignal | undefined
    const onSlow=vi.fn((s:AbortSignal)=>new Promise<'wait'>(resolve=>{
      signal=s;s.addEventListener('abort',()=>resolve('wait'))
    }))
    const forceKill=vi.fn()
    const work=stopProcessGracefully({isAlive:()=>alive,terminate:vi.fn(),forceKill,onSlow})
    await vi.advanceTimersByTimeAsync(10_000)
    alive=false;await vi.advanceTimersByTimeAsync(100);await work
    expect(signal?.aborted).toBe(true);expect(forceKill).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })
  it('forces only on explicit choice and waits for actual process exit', async () => {
    vi.useFakeTimers();let alive=true
    const forceKill=vi.fn(()=>{setTimeout(()=>{alive=false},200)})
    const work=stopProcessGracefully({isAlive:()=>alive,terminate:vi.fn(),forceKill,onSlow:async()=> 'force'})
    await vi.advanceTimersByTimeAsync(10_300);await work
    expect(forceKill).toHaveBeenCalledOnce();expect(alive).toBe(false)
  })
  it('rejects instead of reporting success if force stop does not exit', async () => {
    vi.useFakeTimers()
    const work=stopProcessGracefully({isAlive:()=>true,terminate:vi.fn(),forceKill:vi.fn(),onSlow:async()=> 'force'})
    const rejected=expect(work).rejects.toThrow('did not exit')
    await vi.advanceTimersByTimeAsync(15_100)
    await rejected
  })
  it('does not signal a process which has already exited', async () => {
    const terminate=vi.fn(),forceKill=vi.fn(),onSlow=vi.fn()
    await stopProcessGracefully({isAlive:()=>false,terminate,forceKill,onSlow})
    expect(terminate).not.toHaveBeenCalled();expect(onSlow).not.toHaveBeenCalled()
  })
})

import { beforeEach, describe, expect, it, vi } from 'vitest'

const clipboard = vi.hoisted(() => ({ readText: vi.fn(), writeText: vi.fn() }))
vi.mock('electron', () => ({ clipboard }))
vi.mock('../src/main/database', () => ({ db: { getSetting: vi.fn() } }))

import { executeBuiltinTool } from '../src/main/tools/executor'

describe('asynchronous Electron clipboard tools', () => {
  beforeEach(() => vi.resetAllMocks())

  it('awaits the native read and preserves whitespace and Unicode', async () => {
    const text = '  keep\n한글\\n\n'
    clipboard.readText.mockResolvedValue(text)
    expect(await executeBuiltinTool('clipboard_read', {}, '')).toEqual({
      content: `Clipboard (${text.length} chars):\n\n${text}`, is_error: false,
    })
  })

  it('reports a resolved empty clipboard, not a Promise', async () => {
    clipboard.readText.mockResolvedValue('')
    expect(await executeBuiltinTool('clipboard_read', {}, '')).toEqual({
      content: '(clipboard is empty)', is_error: false,
    })
  })

  it('does not report success before the native write completes', async () => {
    let finish!: () => void
    clipboard.writeText.mockReturnValue(new Promise<void>(resolve => { finish = resolve }))
    let settled = false
    const result = executeBuiltinTool('clipboard_write', { text: 'Amber' }, '')
    void result.then(() => { settled = true })
    await Promise.resolve()
    await Promise.resolve()
    expect(settled).toBe(false)
    expect(clipboard.writeText).toHaveBeenCalledWith('Amber')
    finish()
    expect(await result).toEqual({ content: 'Written 5 characters to clipboard.', is_error: false })
  })

  it.each(['read', 'write'] as const)('returns a tool error for a rejected native %s', async mode => {
    clipboard[mode === 'read' ? 'readText' : 'writeText'].mockRejectedValue(new Error('clipboard unavailable'))
    const result = await executeBuiltinTool(`clipboard_${mode}`, { text: 'Amber' }, '')
    expect(result.is_error).toBe(true)
    expect(result.content).toContain('clipboard unavailable')
  })

  it('does not write when the required value is missing', async () => {
    const result = await executeBuiltinTool('clipboard_write', {}, '')
    expect(result.is_error).toBe(true)
    expect(clipboard.writeText).not.toHaveBeenCalled()
  })
})

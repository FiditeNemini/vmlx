import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'

// Execute the actual renderer callbacks, not a duplicate implementation.
// Model start/repair and visual progress remain separate main-process/live proof.
const surfaces = [
  ['chat', 'src/renderer/src/components/chat/ChatInterface.tsx'],
  ['toolbar', 'src/renderer/src/components/layout/ChatModeToolbar.tsx'],
] as const

function callbackFor(surface: string, path: string) {
  const source = readFileSync(path, 'utf8')
  const ast = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  const callbacks: string[] = []
  const visit = (node: ts.Node) => {
    if (surface === 'chat' && ts.isJsxAttribute(node) && node.name.getText(ast) === 'onClick' &&
        node.initializer && ts.isJsxExpression(node.initializer)) {
      const expression = node.initializer.expression
      if (expression && ts.isArrowFunction(expression) &&
          expression.getText(ast).includes('window.api.sessions.start(sessionId)')) {
        callbacks.push(expression.getText(ast))
      }
    }
    if (surface === 'toolbar' && ts.isVariableDeclaration(node) &&
        node.name.getText(ast) === 'handleStart' && node.initializer) {
      callbacks.push(node.initializer.getText(ast))
    }
    ts.forEachChild(node, visit)
  }
  visit(ast)
  if (callbacks.length !== 1) throw new Error(`Expected one ${surface} start handler, got ${callbacks.length}`)
  const code = ts.transpileModule(`return (${callbacks[0]})`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText
  return { source, code }
}

for (const [surface, path] of surfaces) {
  describe(`${surface} actual model-start callback`, () => {
    const { source, code } = callbackFor(surface, path)
    function setup(sessionId: string | null = 'owned-session') {
      const start = vi.fn()
      const showToast = vi.fn()
      const t = (key: string) => key
      const run = new Function('window', 'sessionId', 'activeSessionId', 'showToast', 't', code)(
        { api: { sessions: { start } } }, sessionId, sessionId, showToast, t,
      ) as () => Promise<void>
      return { start, showToast, run }
    }

    it('surfaces a resolved IPC lifecycle failure rather than silently treating it as success', async () => {
      const f = setup()
      f.start.mockResolvedValue({ success: false, error: 'Bundle validation failed: shard-02' })
      await f.run()
      expect(f.start).toHaveBeenCalledExactlyOnceWith('owned-session')
      expect(f.showToast).toHaveBeenCalledExactlyOnceWith(
        'error', 'chat.interface.toast.failedToStart', 'Bundle validation failed: shard-02',
      )
    })

    it('also surfaces IPC promise rejection without an unhandled rejection or retry', async () => {
      const f = setup()
      f.start.mockRejectedValue(new Error('IPC transport unavailable'))
      await expect(f.run()).resolves.toBeUndefined()
      expect(f.start).toHaveBeenCalledExactlyOnceWith('owned-session')
      expect(f.showToast).toHaveBeenCalledExactlyOnceWith(
        'error', 'chat.interface.toast.failedToStart', 'IPC transport unavailable',
      )
    })

    it('uses the existing translated fallback when a failed result has no detail', async () => {
      const f = setup()
      f.start.mockResolvedValue({ success: false })
      await f.run()
      expect(f.showToast).toHaveBeenCalledExactlyOnceWith(
        'error', 'chat.interface.toast.failedToStart', 'chat.interface.toast.failedToStart',
      )
    })

    it('does not credit a missing IPC result as successful', async () => {
      const f = setup()
      f.start.mockResolvedValue(undefined)
      await f.run()
      expect(f.showToast).toHaveBeenCalledExactlyOnceWith(
        'error', 'chat.interface.toast.failedToStart', 'chat.interface.toast.failedToStart',
      )
    })

    it('awaits one exact-session start and does not fabricate ready/success feedback', async () => {
      const f = setup('selected-session')
      let resolve!: (result: { success: boolean }) => void
      f.start.mockReturnValue(new Promise(r => { resolve = r }))
      let settled = false
      const pending = f.run().then(() => { settled = true })
      await Promise.resolve()
      expect(settled).toBe(false)
      expect(f.start).toHaveBeenCalledExactlyOnceWith('selected-session')
      expect(f.showToast).not.toHaveBeenCalled()
      resolve({ success: true })
      await pending
      expect(f.showToast).not.toHaveBeenCalled()
      expect(f.start).toHaveBeenCalledTimes(1)
    })

    it('keeps the callback on the actual control and provides its toast hook', () => {
      expect(source).toContain('const { showToast } = useToast()')
      if (surface === 'toolbar') expect(source).toContain('onClick={handleStart}')
      else expect(source).toContain("t('chat.interface.loadModelButton')")
    })

    if (surface === 'toolbar') {
      it('does not dispatch without a selected owner', async () => {
        const f = setup(null)
        await f.run()
        expect(f.start).not.toHaveBeenCalled()
        expect(f.showToast).not.toHaveBeenCalled()
      })
    }
  })
}

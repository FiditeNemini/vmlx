import { readFileSync } from 'node:fs'
import ts from 'typescript'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { InlineToolCall } from '../src/renderer/src/components/chat/InlineToolCall'

function extracted(path: string, predicate: (node: ts.Node, file: ts.SourceFile) => boolean): string {
  const source = readFileSync(path, 'utf8')
  const file = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  let found: ts.Node | undefined
  function visit(node: ts.Node) {
    if (predicate(node, file)) found = node
    ts.forEachChild(node, visit)
  }
  visit(file)
  if (!found) throw new Error(`Owning production node missing: ${path}`)
  return ts.transpileModule(`return (${found.getText(file)});`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
}
function hydrate(events: unknown[]) {
  const expression = extracted('src/renderer/src/components/chat/ChatInterface.tsx', (node, file) =>
    ts.isBinaryExpression(node) && node.left.getText(file) === 'restoredTools[m.id]')
  return new Function('parsed', 'restoredTools', 'm', expression)(events, {}, { id: 'msg', timestamp: 99999 })
}
function html(statuses: any[]) {
  return renderToStaticMarkup(React.createElement(InlineToolCall, { group: { name: 'read_file', statuses }, isStreaming: false }))
}

describe('executor timestamps survive persisted tool history', () => {
  it('uses producer time during live rendering rather than delayed IPC receipt time', () => {
    const expression = extracted('src/renderer/src/components/chat/ChatInterface.tsx', node => ts.isArrowFunction(node) &&
      ts.isVariableDeclaration(node.parent) && node.parent.name.getText() === 'handleToolStatus')
    let state: Record<string, any[]> = {}
    const handle = new Function('chatId', 'isCurrentChat', 'setMessages', 'setToolStatusMap', expression)(
      'chat', () => true, () => undefined,
      (update: (prior: typeof state) => typeof state) => { state = update(state) })
    const now = vi.spyOn(Date, 'now').mockReturnValue(99999)
    try {
      handle({ chatId: 'chat', messageId: 'msg', phase: 'executing', timestamp: 1000 })
      handle({ chatId: 'chat', messageId: 'msg', phase: 'result', timestamp: 1257 })
      expect(state.msg.map(s => s.timestamp)).toEqual([1000, 1257])
      expect(now).not.toHaveBeenCalled()
    } finally { now.mockRestore() }
  })

  it('uses one producer timestamp for IPC and storage, preserving duration through JSON hydration', () => {
    const expression = extracted('src/main/ipc/chat.ts', node => ts.isArrowFunction(node) &&
      ts.isVariableDeclaration(node.parent) && node.parent.name.getText() === 'emitToolStatus')
    const saved: any[] = []
    const sent: any[] = []
    const now = vi.spyOn(Date, 'now').mockReturnValueOnce(1000).mockReturnValueOnce(1257)
    try {
      const emit = new Function('collectedToolStatuses', 'displayTimeline', 'getWindow', 'chatId', 'assistantMessage',
        `let toolStatusNeedsFlush = false; let lastEmittedContentLength = 0; ${expression}`)(
          saved, { snapshot: () => undefined, tool: () => undefined },
          () => ({ isDestroyed: () => false, webContents: { send: (_channel: string, event: unknown) => sent.push(event) } }),
          'chat', { id: 'msg' })
      emit('executing', 'read_file', undefined, 1, 'call')
      emit('result', 'read_file', 'done', 1, 'call')
    } finally { now.mockRestore() }
    expect(saved.map(x => x.timestamp)).toEqual([1000, 1257])
    expect(sent.map(x => x.timestamp)).toEqual([1000, 1257])
    expect(html(hydrate(JSON.parse(JSON.stringify(saved))))).toContain('data-vmlx-proof-tool-duration-ms="257"')
  })
  it('does not invent zero duration from legacy rows without event times', () => {
    const restored = hydrate([{ phase: 'executing', toolName: 'read_file' }, { phase: 'result', toolName: 'read_file' }])
    expect(restored.every((x: any) => x.timestamp === undefined)).toBe(true)
    expect(html(restored)).not.toContain('data-vmlx-proof-tool-duration-ms')
  })
  it('preserves a genuine measured zero duration and rejects malformed timing', () => {
    const statuses = [{ phase: 'executing', toolName: 'read_file', timestamp: 0 }, { phase: 'result', toolName: 'read_file', timestamp: 0 }]
    expect(html(hydrate(statuses))).toContain('data-vmlx-proof-tool-duration-ms="0"')
    expect(html(hydrate(statuses.map(s => ({ ...s, timestamp: '123' }))))).not.toContain('data-vmlx-proof-tool-duration-ms')
  })
})

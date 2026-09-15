import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'
import { restoreUserMessageContent } from '../src/renderer/src/components/chat/messageReplay'

// Execute the actual handler (not a test-side copy or string-presence proxy).
// React rendering and transport are covered separately by the live UI receipt.
function editHandler(options: { content?: string; role?: string; deleteError?: Error } = {}) {
  const source = readFileSync('src/renderer/src/components/chat/ChatInterface.tsx', 'utf8')
  const ast = ts.createSourceFile('ChatInterface.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  let expression = ''
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(ast) === 'handleEdit') {
      expression = node.initializer!.getText(ast)
    }
    ts.forEachChild(node, visit)
  }
  visit(ast)
  expect(expression).not.toBe('')
  const send = vi.fn().mockResolvedValue(undefined)
  const truncate = options.deleteError
    ? vi.fn().mockRejectedValue(options.deleteError)
    : vi.fn().mockResolvedValue(undefined)
  const update = vi.fn()
  const toast = vi.fn()
  const messages = [{ id: 'original', role: options.role ?? 'user', timestamp: 123,
    content: options.content ?? 'original question' }]
  const dependencies: Record<string, unknown> = {
    chatId: 'chat', chatIdRef: { current: 'chat' }, loading: false, messages,
    window: { api: { chat: { deleteMessagesFrom: truncate } } },
    setMessages: update, handleSend: send, showToast: toast,
    t: (key: string) => key, formatChatSendErrorMessage: (e: Error) => e.message,
    restoreUserMessageContent,
  }
  const code = ts.transpileModule(`return ${expression}`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText
  const handler = new Function(...Object.keys(dependencies), code)(...Object.values(dependencies))
  return { handler, send, truncate, update, toast }
}

describe('Edit and resend uses the original user turn', () => {
  it('preserves the image while replacing the prompt exactly once', async () => {
    const run = editHandler({ content: JSON.stringify([
      { type: 'text', text: 'original question' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,YWJj' } },
    ]) })
    await run.handler('original', 'new question')
    expect(run.truncate).toHaveBeenCalledExactlyOnceWith('chat', 123)
    expect(run.send).toHaveBeenCalledTimes(1)
    expect(run.send.mock.calls[0][0]).toBe('new question')
    expect(run.send.mock.calls[0][1]).toEqual([
      expect.objectContaining({ kind: 'image', dataUrl: 'data:image/png;base64,YWJj' }),
    ])
  })

  it('does not send or discard visible history if truncation fails', async () => {
    const run = editHandler({ deleteError: new Error('database is locked') })
    await run.handler('original', 'new question')
    expect(run.send).not.toHaveBeenCalled()
    expect(run.update).not.toHaveBeenCalled()
    expect(run.toast).toHaveBeenCalledOnce()
  })

  it('does not replay an assistant message as a user message', async () => {
    const run = editHandler({ role: 'assistant' })
    await run.handler('original', 'new question')
    expect(run.truncate).not.toHaveBeenCalled()
    expect(run.send).not.toHaveBeenCalled()
  })
})

describe('Persisted user content replay', () => {
  it.each(['a plain prompt', '[1, 2]', '[]', '[not JSON', '  indented\ntext\n'])('keeps literal text %j unchanged', content => {
    expect(restoreUserMessageContent(content)).toEqual({ content })
  })

  it('retains ordered images, video and audio with exact payloads', () => {
    const parts = [
      { type: 'image_url', image_url: { url: 'https://example.invalid/a.png?sig=unchanged' } },
      { type: 'text', text: '  prompt\n' },
      { type: 'video_url', video_url: { url: 'data:video/mp4;base64,YWJj' } },
      { type: 'input_audio', input_audio: { data: 'ZGVm', format: 'mp3' } },
      { type: 'image_url', image_url: { url: 'data:image/webp;base64,Z2hp' } },
      { type: 'text', text: '[Attached file: note.txt]\n  file bytes\n' },
    ]
    const before = JSON.stringify(parts)
    const result = restoreUserMessageContent(before)
    expect(result.content).toBe('  prompt\n\n\n[Attached file: note.txt]\n  file bytes\n')
    expect(result.attachments?.map(a => [a.kind, a.dataUrl])).toEqual([
      ['image', parts[0].image_url!.url], ['video', parts[2].video_url!.url],
      ['audio', 'data:audio/mpeg;base64,ZGVm'], ['image', parts[4].image_url!.url],
    ])
    expect(new Set(result.attachments?.map(a => a.id)).size).toBe(4)
    expect(JSON.stringify(parts)).toBe(before)
  })

  it.each(['wav', 'mp3', 'flac', 'ogg', 'm4a'])('restores bare %s audio without changing its bytes', format => {
    const result = restoreUserMessageContent(JSON.stringify([{ type: 'input_audio', input_audio: { data: 'YWJj', format } }]))
    expect(result.content).toBe('')
    expect(result.attachments?.[0].dataUrl).toBe(`data:audio/${format === 'mp3' ? 'mpeg' : format};base64,YWJj`)
  })

  it('does not double-prefix stored audio data URLs', () => {
    const url = 'data:audio/wav;base64,YWJj'
    const result = restoreUserMessageContent(JSON.stringify([{ type: 'input_audio', input_audio: { data: url, format: 'wav' } }]))
    expect(result.attachments?.[0].dataUrl).toBe(url)
  })

  it('exposes every text block and text-file payload to the actual edit UI', () => {
    const result = restoreUserMessageContent(JSON.stringify([
      { type: 'text', text: 'question' }, { type: 'text', text: '[Attached file: a.json]\n{"n": 2}' },
    ]))
    expect(result).toEqual({ content: 'question\n\n[Attached file: a.json]\n{"n": 2}', attachments: undefined })
    const bubble = readFileSync('src/renderer/src/components/chat/MessageBubble.tsx', 'utf8')
    expect(bubble).toContain('setEditText(restoreUserMessageContent(message.content).content)')
  })

  it('uses the same reconstruction for Regenerate', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatInterface.tsx', 'utf8')
    expect(source).toContain('const { content, attachments } = restoreUserMessageContent(lastUser.content)')
    expect(source).toContain('await handleSend(content, attachments)')
  })
})

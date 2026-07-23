import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readPanelSource = (path: string) =>
  readFileSync(new URL(`../${path}`, import.meta.url), 'utf8')

describe('bounded answer-pass phase', () => {
  it('resets only the physical decode clock when the SSE comment arrives', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const start = source.indexOf('const beginAnswerPass = () =>')
    const block = source.slice(start, source.indexOf('// Client-side tool call buffering', start))

    expect(start).toBeGreaterThan(0)
    expect(block).toContain('lastTokenTime = null')
    expect(block).toContain('tpsSnapshots.length = 0')
    expect(block).toContain('tpsTokenBase = tokenCount')
    expect(block).toContain('win.webContents.send("chat:answerPass"')
    expect(source).toContain('trimmed === ": vmlx-answer-pass-start"')
    expect(source).toContain('beginAnswerPass();')
  })

  it('shows a distinct finalizing state and clears it on content or completion', () => {
    const preload = readPanelSource('src/preload/index.ts')
    const chat = readPanelSource('src/renderer/src/components/chat/ChatInterface.tsx')
    const bubble = readPanelSource('src/renderer/src/components/chat/MessageBubble.tsx')

    expect(preload).toContain("ipcRenderer.on('chat:answerPass'")
    expect(chat).toContain('window.api.chat.onAnswerPass(handleAnswerPass)')
    expect(chat).toContain('[data.messageId]: true')
    expect(chat).toContain('[data.messageId]: false')
    expect(bubble).toContain('answerPassPending && isStreaming && !message.content')
    expect(bubble).toContain("t('chat.bubble.finalizingAnswer')")
  })

  it('ships the answer-pass label in every locale', () => {
    for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
      const messages = JSON.parse(
        readPanelSource(`src/renderer/src/i18n/locales/${locale}.json`),
      )
      expect(messages.chat.bubble.finalizingAnswer).toBeTruthy()
    }
  })
})

import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const source = readFileSync('src/renderer/src/components/chat/ChatInterface.tsx', 'utf8')
const start = source.indexOf('      // Reload messages from DB to restore consistent state')
const end = source.indexOf('\n    } finally {', start)
if (start < 0 || end < 0) throw new Error('Chat error reconciliation not found')
const run = async (persisted: any[], initial: any[], switched = false) => {
  let messages = initial
  const context = {
    window: { api: { chat: { getMessages: async () => persisted } } },
    chatId: 'current', chatIdRef: { current: switched ? 'other' : 'current' }, tempId: 'temporary-user',
    hydrateMessages: (values: any[]) => values.map(value => ({ ...value, hydrated: true })),
    setMessages: (value: any) => { messages = typeof value === 'function' ? value(messages) : value },
  }
  await new Function(...Object.keys(context), `return (async () => {${source.slice(start, end)}})()`)(...Object.values(context))
  return messages
}

describe('chat error history reconciliation', () => {
  it('removes the typing assistant when a rejected first request persisted no messages', async () => {
    const initial = [{ id: 'temporary-user', role: 'user' }, { id: 'typing-assistant', role: 'assistant', content: '' }]
    expect(await run([], initial)).toEqual([])
  })
  it('preserves persisted partial output and older history after an error', async () => {
    const persisted = [{ id: 'prior', role: 'user', content: 'question' }, { id: 'partial', role: 'assistant', content: 'Partial answer', reasoningContent: 'Saved reasoning' }]
    expect(await run(persisted, [])).toEqual(persisted.map(value => ({ ...value, hydrated: true })))
  })
  it('does not overwrite a different chat after navigation', async () => {
    const initial = [{ id: 'other-chat-message' }]
    expect(await run([], initial, true)).toEqual(initial)
  })
})

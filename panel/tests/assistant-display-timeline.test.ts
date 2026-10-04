import { describe, expect, it } from 'vitest'
import { AssistantDisplayTimelineRecorder, readAssistantDisplayTimeline, validateAssistantDisplayTimeline } from '../src/shared/assistantDisplayTimeline'

describe('recorded assistant display order', () => {
  it('keeps reasoning, content, executed calls and post-result reasoning in emitted order', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeReasoning(['Plan'])
    record.observeReasoning(['Plan first'])
    record.observeContent('Before write.\n')
    record.tool('write')
    record.observeReasoning(['Plan first', 'Check written file'])
    record.observeContent('Before write.\nBefore read.')
    record.tool('read')
    record.observeReasoning(['Plan first', 'Check written file', 'Both results verified'])
    record.observeContent('Before write.\nBefore read.\nFinal answer.')
    const snapshot = record.finalize('Before write.\nBefore read.\nFinal answer.', ['Plan first', 'Check written file', 'Both results verified'])!
    expect(snapshot.items.map(item => item.kind === 'tool' ? item.callId : item.kind)).toEqual([
      'reasoning', 'content', 'write', 'reasoning', 'content', 'read', 'reasoning', 'content',
    ])
    expect(validateAssistantDisplayTimeline(JSON.parse(JSON.stringify(snapshot)), 'Before write.\nBefore read.\nFinal answer.', ['Plan first', 'Check written file', 'Both results verified'])).toEqual(snapshot)
  })

  it('preserves empty reasoning slots without guessing the next segment from tool count', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeReasoning(['First'])
    record.tool('a')
    record.observeReasoning(['First', ''])
    record.tool('b')
    record.observeReasoning(['First', '', 'Final'])
    expect(record.snapshot()!.items).toEqual([
      { kind: 'reasoning', segment: 0, start: 0, end: 5 },
      { kind: 'tool', callId: 'a' }, { kind: 'tool', callId: 'b' },
      { kind: 'reasoning', segment: 2, start: 0, end: 5 },
    ])
  })

  it('records content before later reasoning within the same assistant pass', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeContent('Visible first.')
    record.observeReasoning(['Then reason'])
    record.observeContent('Visible first. Continue.')
    record.observeReasoning(['Then reason more'])
    expect(record.snapshot()!.items.map(item => item.kind)).toEqual(['content', 'reasoning', 'content', 'reasoning'])
  })

  it('does not mutate an earlier IPC snapshot when later deltas arrive', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeContent('a')
    const snapshot = record.snapshot()
    record.observeContent('ab')
    expect(snapshot!.items).toEqual([{ kind: 'content', start: 0, end: 1 }])
  })

  it('allows final whitespace cleanup while preserving all content boundaries', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeContent('\n before ')
    record.tool('a')
    record.observeContent('\n before after \n')
    const final = record.finalize('before after', [])!
    expect(final.items).toEqual([
      { kind: 'content', start: 0, end: 7 }, { kind: 'tool', callId: 'a' },
      { kind: 'content', start: 7, end: 12 },
    ])
    expect(validateAssistantDisplayTimeline(final, 'before after', [])).toBe(final)
  })

  it.each(['content rewrite', 'reasoning rewrite', 'unobserved final text', 'missing call identity'])('falls back instead of inventing order for %s', change => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeContent('a')
    record.observeReasoning(['r'])
    if (change === 'content rewrite') record.observeContent('b')
    if (change === 'reasoning rewrite') record.observeReasoning(['s'])
    if (change === 'missing call identity') record.tool(undefined)
    expect(record.finalize(change === 'unobserved final text' ? 'unseen' : 'a', ['r'])).toBeUndefined()
  })

  it('rejects missing, malformed and out-of-range metadata without fabricating legacy order', () => {
    expect(readAssistantDisplayTimeline(undefined)).toBeUndefined()
    expect(readAssistantDisplayTimeline({ version: 1, items: [{ kind: 'reasoning', start: 0, end: 2 }] })).toBeUndefined()
    expect(validateAssistantDisplayTimeline({ version: 1, items: [{ kind: 'content', start: 1, end: 2 }] }, 'ab', [])).toBeUndefined()
    expect(validateAssistantDisplayTimeline({ version: 1, items: [] }, 'ab', [])).toBeUndefined()
  })
})

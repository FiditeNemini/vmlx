import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { accumulateChatToolCallDelta, type StreamedChatToolCall } from '../src/shared/chatToolCallDeltas'

function production(chunks: any[]): StreamedChatToolCall[] {
  const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
  const start = source.indexOf('              if (choice?.tool_calls && Array.isArray(choice.tool_calls)) {')
  const end = source.indexOf('\n            }\n          } catch', start)
  expect(start).toBeGreaterThan(0)
  expect(end).toBeGreaterThan(start)
  const run = new Function('choice', 'receivedToolCalls', 'uuidv4', 'console', 'accumulateChatToolCallDelta', source.slice(start,end))
  const calls: StreamedChatToolCall[] = []
  for (const delta of chunks) run({tool_calls:[delta]},calls,()=> 'fallback-id',{log(){}},accumulateChatToolCallDelta)
  return calls
}

describe('real Chat SSE tool accumulator', () => {
  it('preserves the provider id in the captured id-only -> function chunk shape', () => {
    expect(production([
      {index:0,id:'call_provider',type:'function'},
      {index:0,function:{name:'read_file',arguments:'{"path":"fixture.json"}'}},
    ])).toEqual([{id:'call_provider',function:{name:'read_file',arguments:'{"path":"fixture.json"}'}}])
  })
  it('accepts a late id without losing earlier name or arguments', () => {
    expect(production([
      {index:0,function:{name:'read_',arguments:'{"path":'}},
      {index:0,id:'late',function:{name:'file',arguments:'"fixture.json"}'}},
    ])).toEqual([{id:'late',function:{name:'read_file',arguments:'{"path":"fixture.json"}'}}])
  })
  it('keeps interleaved calls independent and preserves JSON escapes byte-for-byte', () => {
    const calls: StreamedChatToolCall[] = []
    for(const d of [
      {index:1,id:'b'}, {index:0,id:'a'},
      {index:0,function:{name:'write_file',arguments:'{"content":"a\\n'}},
      {index:1,function:{name:'read_file',arguments:'{}'}},
      {index:0,function:{arguments:'b"}'}},
    ]) accumulateChatToolCallDelta(calls,d,()=> 'fallback')
    expect(calls.map(c=>c.id)).toEqual(['a','b'])
    expect(calls[0].function.arguments).toBe('{"content":"a\\nb"}')
    expect(JSON.parse(calls[0].function.arguments).content).toBe('a\nb')
  })
  it('retains support for complete unindexed legacy calls', () => {
    expect(production([{id:'whole',function:{name:'read_file',arguments:'{}'}}]))
      .toEqual([{id:'whole',function:{name:'read_file',arguments:'{}'}}])
  })
})

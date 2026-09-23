import { describe, expect, it } from 'vitest'
import { captureGenerationPass, renderSessionMarkdown, parseSessionMarkdown, literalBlock } from '../src/main/session-export'
import type { Chat, Message } from '../src/main/database'

const chat: Chat = { id: 'chat', title: 'Test 한국어', createdAt: 1, updatedAt: 2, modelId: 'alias', modelPath: '/models/mimo' }
const message = (extra: Partial<Message>): Message => ({ id: 'm', chatId: 'chat', timestamp: 3, role: 'assistant', content: '', ...extra })

describe('session request provenance', () => {
  it('freezes every pass with actual instructions, schemas and explicit zero/false overrides', () => {
    const body = { model: 'alias', temperature: 0, enable_thinking: false, chat_template_kwargs: { enable_thinking: false }, instructions: 'system at generation', tools: [{name:'lookup',parameters:{type:'object'}}], authorization: 'SECRET' }
    const server = { maxContextLength: 32768, defaultTemperature: 100, pagedCacheBlockSize: 64, maxCacheBlocks: 513, blockDiskCacheMaxPercent: 0.5, kvCacheQuantization: 'none', apiKey: 'SECRET' }
    const health = { effective_defaults: {temperature:1,top_p:0.95}, sampling_defaults:{temperature:1}, active_parsers:{tool_call_parser:'xml_function'} }
    const first = captureGenerationPass({body,serverConfig:server,health,wireApi:'responses',modelPath:'/models/mimo',now:123})
    body.instructions = 'changed later'; body.chat_template_kwargs.enable_thinking = true; server.maxContextLength = 999; health.effective_defaults.temperature = 8
    expect(first.requestSettings).toMatchObject({temperature:0,enable_thinking:false,chat_template_kwargs:{enable_thinking:false}})
    expect(first.serverDefaults).toEqual({temperature:1,top_p:0.95})
    expect(first.instructions).toBe('system at generation')
    expect(first.maxPromptTokens).toBe(32768)
    expect(first.serverSettings).toMatchObject({defaultTemperature:100,pagedCacheBlockSize:64,maxCacheBlocks:513,blockDiskCacheMaxPercent:0.5,kvCacheQuantization:'none'})
    expect(JSON.stringify(first)).not.toContain('SECRET')
    const second = captureGenerationPass({body:{model:'other',messages:[{role:'system',content:'replacement'},{role:'user',content:'hi'}]},serverConfig:{},wireApi:'completions',modelPath:'/models/nemotron'})
    expect(second.modelPath).toBe('/models/nemotron')
    expect(second.serverDefaults).toBeNull()
    expect(second.systemMessages).toEqual([{role:'system',content:'replacement'}])
  })
})

describe('full session Markdown', () => {
  it('round trips identity, timestamps and tool IDs without overwriting content', () => {
    const rows = [message({ content: 'Original answer', toolCallId: 'call-한글', toolCapabilityFingerprint: 'schema-v1' })]
    const parsed = parseSessionMarkdown(renderSessionMarkdown(chat, rows, new Date(0)))!
    expect(parsed).toMatchObject({ modelId: chat.modelId, modelPath: chat.modelPath, createdAt: chat.createdAt })
    expect(parsed.messages[0]).toMatchObject({ content: 'Original answer', timestamp: 3, toolCallId: 'call-한글', toolCapabilityFingerprint: 'schema-v1' })
  })
  it('retains reasoning-only turns, tool history, status and literal embedded fences in order', () => {
    const reason = 'Reflect.\n```json\n{"x":1}\n```\n## 999. assistant\n</details>'
    const rows = [message({id:'u',role:'user',content:'안녕'}),message({id:'a',reasoningContent:reason,reasoningSegmentsJson:JSON.stringify(['first','','last']),generationRecordJson:JSON.stringify({version:1,status:'interrupted',passes:[{modelPath:'/models/mimo',toolExchange:[{role:'tool',tool_call_id:'id-1',content:'full result'}]}]}),toolCallsOaiJson:'[{"id":"id-1","function":{"name":"lookup","arguments":"{}"}}]',warningsJson:'["length"]'})]
    const md = renderSessionMarkdown(chat,rows,new Date(0))
    expect(md).toContain(reason)
    expect(md).toContain('full result')
    expect(md).toContain('interrupted')
    expect(md.indexOf('## 1. user')).toBeLessThan(md.indexOf('## 2. assistant'))
    const parsed = parseSessionMarkdown(md)!
    expect(parsed.title).toBe(chat.title)
    expect(parsed.messages).toHaveLength(2)
    expect(parsed.messages[1].reasoningContent).toBe(reason)
    expect(parsed.messages[1].content).toBe('')
    expect(JSON.parse(parsed.messages[1].generationRecordJson!)).toEqual(JSON.parse(rows[1].generationRecordJson!))
    expect(JSON.parse(parsed.messages[1].toolCallsOaiJson!)[0].id).toBe('id-1')
  })
  it('labels legacy settings unknown and never fetches current defaults', () => {
    const md = renderSessionMarkdown(chat,[message({content:'old answer'})])
    expect(md).toContain('unavailable (not recorded for this message)')
    expect(md).not.toContain('temperature')
    expect(parseSessionMarkdown('# old export\n## User\nhello')).toBeNull()
  })
  it('uses a fence longer than all recorded backtick runs', () => {
    expect(literalBlock('````\n## injected')).toBe('`````text\n````\n## injected\n`````')
  })
})

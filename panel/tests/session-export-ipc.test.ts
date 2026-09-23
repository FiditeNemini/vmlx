import { beforeEach, describe, expect, it, vi } from 'vitest'

const { handlers, dialog, writeFileSync, readFileSync, statSync, db } = vi.hoisted(() => ({
  handlers: new Map<string, (...args: any[]) => any>(),
  dialog: { showSaveDialog: vi.fn(), showOpenDialog: vi.fn() },
  writeFileSync: vi.fn(),
  readFileSync: vi.fn(), statSync: vi.fn(),
  db: { getChat:vi.fn(), getMessages:vi.fn(), createChat:vi.fn(), addMessage:vi.fn() },
}))
vi.mock('electron',()=>({ipcMain:{handle:(name:string,fn:any)=>handlers.set(name,fn)},dialog}))
vi.mock('fs',()=>({writeFileSync,readFileSync,statSync}))
vi.mock('../src/main/database',()=>({db}))
import { registerExportHandlers } from '../src/main/ipc/export'

beforeEach(()=>{
  vi.clearAllMocks(); handlers.clear(); registerExportHandlers()
  db.getChat.mockReturnValue({id:'c',title:'Session',modelId:'m',modelPath:'/models/m',createdAt:1})
  db.getMessages.mockReturnValue([{id:'a',role:'assistant',timestamp:2,content:'final',reasoningContent:'full reasoning'}])
  statSync.mockReturnValue({size:1000})
})
describe('session export native save flow',()=>{
  it('does not write or report success after cancellation',async()=>{
    dialog.showSaveDialog.mockResolvedValue({canceled:true})
    expect(await handlers.get('chat:export')!(null,'c','markdown')).toEqual({success:false})
    expect(writeFileSync).not.toHaveBeenCalled()
  })
  it('saves full Markdown and returns the actual selected path',async()=>{
    dialog.showSaveDialog.mockResolvedValue({canceled:false,filePath:'/tmp/session.md'})
    expect(await handlers.get('chat:export')!(null,'c','markdown')).toEqual({success:true,path:'/tmp/session.md'})
    expect(dialog.showSaveDialog.mock.calls[0][0].title).toBe('Export session')
    expect(writeFileSync.mock.calls[0][1]).toContain('full reasoning')
    expect(writeFileSync.mock.calls[0][1]).toContain('/models/m')
  })
  it('reports a write failure instead of claiming export succeeded',async()=>{
    dialog.showSaveDialog.mockResolvedValue({canceled:false,filePath:'/tmp/session.md'})
    writeFileSync.mockImplementationOnce(()=>{throw new Error('disk full')})
    await expect(handlers.get('chat:export')!(null,'c','markdown')).rejects.toThrow('Failed to write file: disk full')
  })
  it('keeps ShareGPT compatible',async()=>{
    dialog.showSaveDialog.mockResolvedValue({canceled:false,filePath:'/tmp/session.json'})
    await handlers.get('chat:export')!(null,'c','sharegpt')
    expect(JSON.parse(writeFileSync.mock.calls[0][1])).toEqual({conversations:[{from:'gpt',value:'final'}]})
  })
  it.each(['markdown', 'json'])('round trips %s identity, original time and generation provenance', async format => {
    const path = format === 'markdown' ? '/tmp/session.md' : '/tmp/session.json'
    const record = {version:1,status:'interrupted',passes:[{modelPath:'/models/original',requestSettings:{temperature:0},instructions:'original system'}]}
    db.getMessages.mockReturnValue([{id:'a',role:'assistant',timestamp:2,content:'',reasoningContent:'한국어 ```thinking```',generationRecordJson:JSON.stringify(record)}])
    dialog.showSaveDialog.mockResolvedValue({canceled:false,filePath:path})
    await handlers.get('chat:export')!(null,'c',format)
    readFileSync.mockReturnValue(writeFileSync.mock.calls[0][1])
    dialog.showOpenDialog.mockResolvedValue({canceled:false,filePaths:[path]})
    await handlers.get('chat:import')!(null)
    expect(db.createChat).toHaveBeenCalledWith(expect.objectContaining({modelId:'m',modelPath:'/models/m',createdAt:1}))
    expect(db.addMessage).toHaveBeenCalledWith(expect.objectContaining({content:'',timestamp:2,reasoningContent:'한국어 ```thinking```'}))
    expect(JSON.parse(db.addMessage.mock.calls[0][0].generationRecordJson)).toEqual(record)
    // A deliberately chosen import target changes the association, not saved provenance.
    await handlers.get('chat:import')!(null,'/models/selected')
    expect(db.createChat.mock.calls[1][0]).toMatchObject({modelId:'default',modelPath:'/models/selected'})
    expect(JSON.parse(db.addMessage.mock.calls[1][0].generationRecordJson)).toEqual(record)
  })
  it('round trips Markdown tool associations and ordered segments', async () => {
    const original = {id:'a',role:'assistant',timestamp:2,content:'answer',toolCallId:'call-1',toolCapabilityFingerprint:'schema-1',toolCallsOaiJson:'[{"id":"call-1","function":{"name":"lookup","arguments":"{}"}}]',toolResultsOaiJson:'[{"tool_call_id":"call-1","content":"value"}]',reasoningSegmentsJson:'["before","after"]'}
    db.getMessages.mockReturnValue([original])
    dialog.showSaveDialog.mockResolvedValue({canceled:false,filePath:'/tmp/session.md'})
    await handlers.get('chat:export')!(null,'c','markdown')
    readFileSync.mockReturnValue(writeFileSync.mock.calls[0][1])
    dialog.showOpenDialog.mockResolvedValue({canceled:false,filePaths:['/tmp/session.md']})
    await handlers.get('chat:import')!(null)
    const imported = db.addMessage.mock.calls[0][0]
    expect(imported).toMatchObject({content:'answer',toolCallId:'call-1',toolCapabilityFingerprint:'schema-1',timestamp:2})
    for (const field of ['toolCallsOaiJson','toolResultsOaiJson','reasoningSegmentsJson'] as const) expect(JSON.parse(imported[field])).toEqual(JSON.parse(original[field]))
  })
})

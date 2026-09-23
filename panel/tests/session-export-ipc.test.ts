import { beforeEach, describe, expect, it, vi } from 'vitest'

const { handlers, dialog, writeFileSync, db } = vi.hoisted(() => ({
  handlers: new Map<string, (...args: any[]) => any>(),
  dialog: { showSaveDialog: vi.fn(), showOpenDialog: vi.fn() },
  writeFileSync: vi.fn(),
  db: { getChat:vi.fn(), getMessages:vi.fn() },
}))
vi.mock('electron',()=>({ipcMain:{handle:(name:string,fn:any)=>handlers.set(name,fn)},dialog}))
vi.mock('fs',()=>({writeFileSync,readFileSync:vi.fn(),statSync:vi.fn()}))
vi.mock('../src/main/database',()=>({db}))
import { registerExportHandlers } from '../src/main/ipc/export'

beforeEach(()=>{
  vi.clearAllMocks(); handlers.clear(); registerExportHandlers()
  db.getChat.mockReturnValue({id:'c',title:'Session',modelId:'m',modelPath:'/models/m',createdAt:1})
  db.getMessages.mockReturnValue([{id:'a',role:'assistant',timestamp:2,content:'final',reasoningContent:'full reasoning'}])
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
})

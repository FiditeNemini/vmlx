import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
const h = vi.hoisted(() => {
  const emitter = () => {
    const events = new Map<string, Function[]>()
    return { on: (key: string, fn: Function) => events.set(key, [...(events.get(key) || []), fn]),
      off: (key: string, fn: Function) => events.set(key, (events.get(key) || []).filter(x => x !== fn)),
      emit: (key: string, data?: any) => events.get(key)?.forEach(fn => fn(data)) }
  }
  return { menus: [] as any[][], db: { getSessions: vi.fn(), getSetting: vi.fn() },
    clipboard: { writeText: vi.fn().mockResolvedValue(undefined) }, dialog: { showMessageBox: vi.fn().mockResolvedValue({}) },
    manager: { ...emitter(), list: vi.fn(), totalMemoryMB: vi.fn(), kill: vi.fn() },
    sessions: { ...emitter(), softSleep: vi.fn(), wakeSession: vi.fn(), stopSession: vi.fn() },
    gateway: { ...emitter(), running: false, activePort: 9123, activeHost: '::', setSingleModelMode: vi.fn() } }
})
vi.mock('electron', () => ({
  Tray: class { setContextMenu(menu: any) { h.menus.push(menu) }; setToolTip() {}; setImage() {}; destroy() {} },
  Menu: { buildFromTemplate: (x: any) => x },
  nativeImage: { createFromBuffer: () => ({ resize: () => ({}) }) },
  clipboard: h.clipboard, dialog: h.dialog, app: { quit: vi.fn(), emit: vi.fn() },
}))
vi.mock('../src/main/database', () => ({ db: h.db }))
vi.mock('../src/main/sessions', () => ({ sessionManager: h.sessions }))
vi.mock('../src/main/api-gateway', () => ({ apiGateway: h.gateway }))
vi.mock('../src/main/application-menu', () => ({ navigateFromMenu: vi.fn() }))
import { createTray, destroyTray } from '../src/main/tray'
const menu = () => h.menus.at(-1)!
const item = (label: string) => menu().flatMap(x => [x, ...(x.submenu || [])]).find(x => x.label === label)
let rows: any[]
beforeEach(() => {
  vi.clearAllMocks(); h.menus.length = 0
  rows = [{ id: 's', host: '127.0.0.1', port: 9001, status: 'running', modelName: 'Test', config: '{}' }]
  h.db.getSessions.mockImplementation(() => rows); h.db.getSetting.mockReturnValue('8080')
  h.manager.list.mockReturnValue([]); h.manager.totalMemoryMB.mockReturnValue(0)
  h.gateway.running = false; h.gateway.activePort = 9123; h.gateway.activeHost = '::'
})
afterEach(() => destroyTray())
describe('native tray actions and live updates', () => {
  it('refreshes from live gateway events and copies explicit URLs rather than stale settings', async () => {
    createTray(h.manager as any, () => null)
    expect(item('Copy OpenAI Base URL').enabled).toBe(false)
    h.gateway.running = true; h.gateway.emit('started')
    expect(item('Copy OpenAI Base URL').enabled).toBe(true)
    item('Copy OpenAI Base URL').click()
    expect(h.clipboard.writeText).toHaveBeenLastCalledWith('http://[::1]:9123/v1')
    item('Copy Server URL').click()
    expect(h.clipboard.writeText).toHaveBeenLastCalledWith('http://[::1]:9123')
    h.gateway.running = false; h.gateway.emit('stopped')
    expect(item('Copy Server URL').enabled).toBe(false)
  })
  it('retains soft sleep weights and refreshes measured standby memory', () => {
    createTray(h.manager as any, () => null)
    h.sessions.emit('session:health', { sessionId: 's', memory: { active_mb: 4096 } })
    expect(menu().some(x => x.label?.startsWith('Memory: 4.0 /'))).toBe(true)
    rows[0].status = 'standby'; h.sessions.emit('session:standby', { sessionId: 's' })
    expect(menu()[0].label).toContain('0 running, 0 loading, 1 sleeping')
    expect(menu().some(x => x.label?.startsWith('Memory: 4.0 /'))).toBe(true)
    h.sessions.emit('session:memory', { sessionId: 's', memory: { active_mb: 0 } })
    expect(menu().some(x => x.label?.startsWith('Memory: 0.0 /'))).toBe(true)
    expect(item('Wake')).toBeDefined()
    expect(item('Sleep')).toBeUndefined()
  })
  it('prevents duplicate actions, surfaces failures and restores controls', async () => {
    let reject!: (e: Error) => void
    h.sessions.softSleep.mockImplementation(() => new Promise((_, fail) => { reject = fail }))
    createTray(h.manager as any, () => null)
    const action = item('Sleep').click
    const pending = action(); await action()
    expect(h.sessions.softSleep).toHaveBeenCalledTimes(1)
    expect(item('Sleep').enabled).toBe(false)
    reject(new Error('Busy with an active request'))
    await pending
    expect(h.dialog.showMessageBox).toHaveBeenCalledWith(expect.objectContaining({ detail: 'Error: Busy with an active request' }))
    expect(item('Sleep').enabled).toBe(true)
  })
  it('surfaces structured server failures as well as rejected promises', async () => {
    h.sessions.softSleep.mockResolvedValue({ success: false, error: 'Engine busy — generation in progress' })
    createTray(h.manager as any, () => null)
    await item('Sleep').click()
    expect(h.dialog.showMessageBox).toHaveBeenCalledWith(expect.objectContaining({ detail: 'Error: Engine busy — generation in progress' }))
    expect(item('Sleep').enabled).toBe(true)
  })

})

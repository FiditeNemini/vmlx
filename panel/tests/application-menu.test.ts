import { beforeEach, describe, expect, it, vi } from 'vitest'
const { app, Menu } = vi.hoisted(() => ({
  app: { name: 'vMLX', emit: vi.fn() },
  Menu: { buildFromTemplate: vi.fn(x => x), setApplicationMenu: vi.fn() },
}))
vi.mock('electron', () => ({ app, Menu }))
import { acknowledgeMenuReadiness, installApplicationMenu, navigateFromMenu, resetMenuReadiness } from '../src/main/application-menu'
import { isNativeNavigationAction } from '../src/shared/nativeNavigation'

function windowMock(id = 1) {
  return { isDestroyed: () => false, isMinimized: () => true, restore: vi.fn(), show: vi.fn(), focus: vi.fn(),
    webContents: { id, isLoadingMainFrame: () => false, send: vi.fn() } } as any
}
beforeEach(() => { vi.clearAllMocks(); resetMenuReadiness() })
describe('native menu navigation lifecycle', () => {
  it('restores a minimized window and waits for the subscribed renderer', () => {
    const win = windowMock()
    navigateFromMenu('preferences', () => win)
    expect(win.restore).toHaveBeenCalledOnce()
    expect(win.webContents.send).not.toHaveBeenCalled()
    acknowledgeMenuReadiness(win)
    expect(win.webContents.send).toHaveBeenCalledExactlyOnceWith('app:navigate', 'preferences')
    acknowledgeMenuReadiness(win)
    expect(win.webContents.send).toHaveBeenCalledTimes(1)
  })
  it('recreates a closed window and keeps the most recent navigation intent', () => {
    let win: any = null
    app.emit.mockImplementation(() => { win = windowMock(2) })
    navigateFromMenu('models', () => win)
    navigateFromMenu('servers', () => win)
    expect(app.emit).toHaveBeenCalledWith('activate')
    acknowledgeMenuReadiness(win)
    expect(win.webContents.send).toHaveBeenCalledExactlyOnceWith('app:navigate', 'servers')
  })
  it('queues during reload until readiness is acknowledged again', () => {
    const win = windowMock()
    acknowledgeMenuReadiness(win); resetMenuReadiness()
    navigateFromMenu('new-chat', () => win)
    expect(win.webContents.send).not.toHaveBeenCalled()
    acknowledgeMenuReadiness(win)
    expect(win.webContents.send).toHaveBeenCalledWith('app:navigate', 'new-chat')
  })
  it('wires real template shortcuts to the navigation path and retains native editing roles', () => {
    const win = windowMock(); acknowledgeMenuReadiness(win); installApplicationMenu(() => win)
    const template = Menu.buildFromTemplate.mock.calls[0][0]
    expect(template.some((x: any) => x.role === 'editMenu')).toBe(true)
    for (const [key, action] of [['N', 'new-chat'], ['1', 'chat'], ['2', 'servers'], ['3', 'models'], ['4', 'api']]) {
      const item = template.flatMap((x: any) => x.submenu || []).find((x: any) => x.accelerator === `CmdOrCtrl+${key}`)
      item.click()
      expect(win.webContents.send).toHaveBeenLastCalledWith('app:navigate', action)
    }
  })
  it('rejects unknown renderer navigation payloads', () => {
    expect(isNativeNavigationAction('new-chat')).toBe(true)
    expect(isNativeNavigationAction({ mode: 'servers' })).toBe(false)
    expect(isNativeNavigationAction('run-command')).toBe(false)
  })
})

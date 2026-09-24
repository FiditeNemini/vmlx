import { app, Menu, type BrowserWindow } from 'electron'
import { t } from './i18n'
import type { NativeNavigationAction } from '../shared/nativeNavigation'

let readyContentsId: number | null = null
let pendingAction: NativeNavigationAction | null = null

export function resetMenuReadiness(): void { readyContentsId = null }

export function acknowledgeMenuReadiness(window: BrowserWindow): void {
  if (window.isDestroyed()) return
  readyContentsId = window.webContents.id
  if (pendingAction) {
    const action = pendingAction
    pendingAction = null
    window.webContents.send('app:navigate', action)
  }
}

export function navigateFromMenu(action: NativeNavigationAction, getWindow: () => BrowserWindow | null): void {
  let window = getWindow()
  if (!window || window.isDestroyed()) {
    app.emit('activate')
    window = getWindow()
  }
  if (!window || window.isDestroyed()) return
  if (window.isMinimized()) window.restore()
  window.show()
  window.focus()
  if (readyContentsId === window.webContents.id && !window.webContents.isLoadingMainFrame()) {
    window.webContents.send('app:navigate', action)
  } else {
    // The renderer acknowledges only after its listener is installed. Keep
    // the most recent navigation intent through a reload or window recreation.
    pendingAction = action
  }
}

export function installApplicationMenu(getWindow: () => BrowserWindow | null): void {
  const go = (action: NativeNavigationAction) => () => navigateFromMenu(action, getWindow)
  const preferences = { label: t('console.preferences'), accelerator: 'CmdOrCtrl+,', click: go('preferences') }
  const template: Electron.MenuItemConstructorOptions[] = [
    ...(process.platform === 'darwin' ? [{ label: app.name, submenu: [
      { role: 'about' }, { type: 'separator' }, preferences,
      { type: 'separator' }, { role: 'services' }, { type: 'separator' },
      { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' },
      { type: 'separator' }, { role: 'quit' },
    ] } as Electron.MenuItemConstructorOptions] : []),
    { role: 'fileMenu', submenu: [
      { label: t('chat.interface.newChat'), accelerator: 'CmdOrCtrl+N', click: go('new-chat') },
      ...(process.platform !== 'darwin' ? [preferences] : []),
      { type: 'separator' }, { role: 'close' },
    ] },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { label: t('console.pages'), submenu: [
      { label: t('console.chatImages'), accelerator: 'CmdOrCtrl+1', click: go('chat') },
      { label: t('console.serversApi'), accelerator: 'CmdOrCtrl+2', click: go('servers') },
      { label: t('console.models'), accelerator: 'CmdOrCtrl+3', click: go('models') },
      { label: 'API', accelerator: 'CmdOrCtrl+4', click: go('api') },
    ] },
    { role: 'windowMenu' },
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

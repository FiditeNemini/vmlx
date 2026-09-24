/**
 * Menu bar tray for vMLX.
 *
 * Shows model status, memory usage, and provides quick controls.
 * Window close keeps app alive in system tray when enabled.
 *
 * Tray icon states:
 * - Green: at least one model serving
 * - Yellow: starting up / loading model
 * - Gray: no models loaded
 */

import { GATEWAY_SINGLE_MODEL_MODE_KEY, isGatewaySettingEnabled } from '../shared/gatewaySettingsKeys'
import { app, Tray, Menu, nativeImage, BrowserWindow, clipboard, dialog } from 'electron'
import type { ProcessManager } from './process-manager'
import { db } from './database'
import { sessionManager } from './sessions'
import { apiGateway } from './api-gateway'
import { t } from './i18n'
import { navigateFromMenu } from './application-menu'
import { localApiUrl, sessionShownByProcess, summarizeTrayState } from '../shared/trayState'

let tray: Tray | null = null
const busySessions = new Set<string>()
let boundProcessManager: ProcessManager | null = null
const boundListeners: Array<{ event: string; fn: (...args: any[]) => void }> = []

/** Track GPU memory per session (updated from session:health events) */
const sessionMemoryMB: Map<string, number> = new Map()

/**
 * Create a colored circle icon for tray.
 * Uses nativeImage.createFromDataURL with a simple SVG circle.
 */
function createTrayIcon(color: 'green' | 'yellow' | 'red' | 'gray'): Electron.NativeImage {
  // Generate a 44x44 PNG via Canvas-like pixel buffer.
  // macOS tray icons are 22x22 logical pixels; we provide @2x for retina.
  const s = 44
  const cx = s / 2, cy = s / 2, r = s / 2 - 4

  // RGBA pixel buffer
  const buf = Buffer.alloc(s * s * 4, 0)

  const colorValues: Record<string, [number, number, number]> = {
    green:  [34, 197, 94],
    yellow: [234, 179, 8],
    red:    [239, 68, 68],
    gray:   [255, 255, 255],
  }
  const [cr, cg, cb] = colorValues[color]

  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const dx = x - cx, dy = y - cy
      const dist = Math.sqrt(dx * dx + dy * dy)
      const offset = (y * s + x) * 4

      if (color === 'gray') {
        // Ring outline for idle state
        const ringOuter = r, ringInner = r - 3
        if (dist <= ringOuter && dist >= ringInner) {
          // Anti-alias edges
          let alpha = 255
          if (dist > ringOuter - 1) alpha = Math.round((ringOuter - dist) * 255)
          else if (dist < ringInner + 1) alpha = Math.round((dist - ringInner) * 255)
          alpha = Math.max(0, Math.min(255, alpha))
          buf[offset] = cr; buf[offset + 1] = cg; buf[offset + 2] = cb; buf[offset + 3] = alpha
        }
      } else {
        // Filled circle for active states
        if (dist <= r) {
          let alpha = 255
          if (dist > r - 1) alpha = Math.round((r - dist) * 255)
          alpha = Math.max(0, Math.min(255, alpha))
          buf[offset] = cr; buf[offset + 1] = cg; buf[offset + 2] = cb; buf[offset + 3] = alpha
        }
      }
    }
  }

  const img = nativeImage.createFromBuffer(buf, { width: s, height: s })
  return img.resize({ width: 18, height: 18 })
}

function reportActionError(error: unknown): void {
  console.error('[TRAY] Action failed:', error)
  void dialog.showMessageBox({ type: 'error', title: 'vMLX', message: t('main.tray.actionFailed'), detail: String(error) }).catch(failure => console.warn('[TRAY] Could not show error:', failure))
}

async function runSessionAction(id: string, action: () => Promise<unknown>, processManager: ProcessManager, getWindow: () => BrowserWindow | null): Promise<void> {
  if (busySessions.has(id)) return
  busySessions.add(id)
  rebuildMenu(processManager, getWindow)
  try {
    const result = await action()
    if (result && typeof result === 'object' && 'success' in result && result.success === false) {
      throw new Error('error' in result ? String(result.error) : t('main.tray.actionFailed'))
    }
  } catch (error) { reportActionError(error) }
  finally { busySessions.delete(id); rebuildMenu(processManager, getWindow) }
}

/**
 * Build the tray context menu.
 */
function buildMenu(
  processManager: ProcessManager,
  getWindow: () => BrowserWindow | null,
): Electron.Menu {
  const processes = processManager.list()
  const totalGB = Math.round(require('os').totalmem() / (1024 ** 3))
  const sessions = db.getSessions()
  const summary = summarizeTrayState(processes, sessions, sessionMemoryMB)
  const totalMemMB = summary.memoryMB
  const gatewayRoot = localApiUrl(apiGateway.activeHost, apiGateway.activePort)
  const singleModelMode = isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))
  const items: Electron.MenuItemConstructorOptions[] = [
    {
      label: t('main.tray.statusSummary', summary),
      enabled: false,
    },
    {
      label: apiGateway.running ? gatewayRoot : t('main.tray.gatewayStopped'),
      enabled: false,
    },
    {
      label: t('main.tray.copyOpenAiUrl'),
      enabled: apiGateway.running,
      click: () => { void clipboard.writeText(`${gatewayRoot}/v1`).catch(reportActionError) },
    },
    {
      label: t('main.tray.copyServerUrl'),
      enabled: apiGateway.running,
      click: () => { void clipboard.writeText(gatewayRoot).catch(reportActionError) },
    },
    {
      label: t('main.tray.singleModelMode'),
      type: 'checkbox',
      checked: singleModelMode,
      click: () => {
        const next = !singleModelMode
        apiGateway.setSingleModelMode(next)
        try {
          const win = getWindow()
          if (win && !win.isDestroyed()) {
            win.webContents.send('gateway:singleModelModeChanged', { singleModelMode: next })
          }
        } catch (_) {}
        rebuildMenu(processManager, getWindow)
      },
    },
    { type: 'separator' },
  ]

  // Per-model entries
  for (const proc of processes) {
    const statusIcon = proc.status === 'running' ? '●'
      : proc.status === 'starting' ? '◐'
        : proc.status === 'error' ? '✕' : '○'

    const memLabel = proc.gpuMemoryMB > 0
      ? ` — ${(proc.gpuMemoryMB / 1024).toFixed(1)} GB`
      : ''

    const modelName = proc.model.split('/').pop() || proc.model

    items.push({
      label: `${statusIcon} ${modelName} (:${proc.port})${memLabel}`,
      submenu: [
        {
          label: proc.status === 'running'
            ? t('main.tray.statusRunning')
            : proc.status === 'starting'
              ? t('main.tray.statusStarting')
              : proc.status === 'error'
                ? t('main.tray.statusError')
                : proc.status,
          enabled: false,
        },
        {
          label: t('main.tray.port', { port: proc.port }),
          enabled: false,
        },
        { type: 'separator' },
        {
          label: t('main.tray.copyOpenAiUrl'),
          enabled: proc.status === 'running',
          click: () => {
            void clipboard.writeText(localApiUrl('127.0.0.1', proc.port, true)).catch(error => {
              console.warn('[TRAY] Failed to copy API URL:', error)
            })
          },
        },
        {
          label: proc.pinned ? t('main.tray.unpin') : t('main.tray.pin'),
          click: () => {
            processManager.setPinned(proc.id, !proc.pinned)
            rebuildMenu(processManager, getWindow)
          },
        },
        { type: 'separator' },
        {
          label: t('main.tray.stop'),
          enabled: !busySessions.has(`process:${proc.id}`),
          click: () => runSessionAction(`process:${proc.id}`, () => processManager.kill(proc.id), processManager, getWindow),
        },
      ],
    })
  }

  // Add SessionManager sessions (includes image servers not in ProcessManager)
  try {
    const sessions = db.getSessions().filter(s => s.status === 'running' || s.status === 'standby' || s.status === 'loading')
    for (const s of sessions) {
      // Skip if already shown via ProcessManager
      const alreadyShown = sessionShownByProcess(s, processes)
      if (alreadyShown) continue

      let isImage = false
      try { isImage = JSON.parse(s.config || '{}').modelType === 'image' } catch {}
      const isSleeping = s.status === 'standby'
      const isLoading = s.status === 'loading'
      const icon = isSleeping ? '💤' : isLoading ? '◐' : isImage ? '🖼' : '●'
      const modelName = s.modelName || s.modelPath?.split('/').pop() || 'Unknown'
      const sessMem = sessionMemoryMB.get(s.id) || 0
      const sessMemLabel = sessMem > 0 ? ` — ${(sessMem / 1024).toFixed(1)} GB` : ''

      items.push({
        label: `${icon} ${modelName} (:${s.port})${sessMemLabel}`,
        submenu: [
          {
            label: t('main.tray.copyOpenAiUrl'),
            enabled: !isLoading,
            click: () => {
              void clipboard.writeText(localApiUrl(s.host, s.port, true)).catch(error => {
                console.warn('[TRAY] Failed to copy API URL:', error)
              })
            },
          },
          { type: 'separator' },
          ...(!isLoading ? (isSleeping ? [{
            label: t('main.tray.wake'),
            enabled: !busySessions.has(s.id),
            click: () => runSessionAction(s.id, () => sessionManager.wakeSession(s.id), processManager, getWindow),
          }] : [{
            label: t('main.tray.sleep'),
            enabled: !busySessions.has(s.id),
            click: () => runSessionAction(s.id, () => sessionManager.softSleep(s.id), processManager, getWindow),
          }]) : []),
          { type: 'separator' as const },
          {
            label: t('main.tray.stop'),
            enabled: !busySessions.has(s.id),
            click: () => runSessionAction(s.id, () => sessionManager.stopSession(s.id), processManager, getWindow),
          },
        ],
      })
    }
  } catch (_) {}

  if (summary.running + summary.loading + summary.standby === 0) {
    items.push({
      label: t('main.tray.noModels'),
      enabled: false,
    })
  }

  items.push(
    { type: 'separator' },
    ...(['servers', 'models', 'api', 'preferences'] as const).map(action => ({
      label: action === 'api' ? 'API' : t(action === 'servers' ? 'console.serversApi' : `console.${action}`),
      click: () => navigateFromMenu(action, getWindow),
    })),
    { type: 'separator' },
    {
      label: t('main.tray.memory', { used: (totalMemMB / 1024).toFixed(1), total: String(totalGB) }),
      enabled: false,
    },
    { type: 'separator' },
    {
      label: t('main.tray.openWindow'),
      click: () => {
        const win = getWindow()
        if (win && !win.isDestroyed()) {
          if (win.isMinimized()) win.restore()
          win.show()
          win.focus()
        } else {
          // Window was closed — recreate
          app.emit('activate')
        }
      },
    },
    {
      label: t('main.tray.quit'),
      click: () => {
        app.quit()
      },
    },
  )

  return Menu.buildFromTemplate(items)
}

/**
 * Rebuild the tray menu and update icon (call after process state changes).
 */
export function rebuildMenu(
  processManager: ProcessManager,
  getWindow: () => BrowserWindow | null,
): void {
  if (!tray) return
  const processes = processManager.list()

  const summary = summarizeTrayState(processes, db.getSessions(), sessionMemoryMB)
  const iconColor = summary.running ? 'green' : (summary.loading || summary.standby) ? 'yellow' : 'gray'
  tray.setImage(createTrayIcon(iconColor))
  tray.setContextMenu(buildMenu(processManager, getWindow))
  tray.setToolTip(t('main.tray.statusSummary', summary))
}

/**
 * Create the system tray.
 */
export function createTray(
  processManager: ProcessManager,
  getWindow: () => BrowserWindow | null,
): Tray {
  if (tray) return tray

  tray = new Tray(createTrayIcon('gray'))
  tray.setToolTip(t('main.tray.tooltipNoModels'))
  tray.setContextMenu(buildMenu(processManager, getWindow))

  // macOS: clicking tray icon opens the context menu automatically (setContextMenu).
  // No 'click' handler — adding one causes BOTH the menu AND window to open on every click.
  // "Open vMLX Window" menu item handles showing the window explicitly.

  // Listen for process state changes to update tray (store refs for cleanup)
  boundProcessManager = processManager
  boundListeners.length = 0
  const events = ['process:spawn', 'process:ready', 'process:exit', 'process:killed', 'process:evicted', 'process:pinChanged']
  for (const event of events) {
    const fn = () => rebuildMenu(processManager, getWindow)
    processManager.on(event, fn)
    boundListeners.push({ event, fn })
  }

  // Also listen for SessionManager events (sessions started from Server/Image tabs)
  const sessionRebuild = () => {
    const live = new Set(db.getSessions().filter(s => ['running', 'loading', 'standby'].includes(s.status)).map(s => s.id))
    for (const id of sessionMemoryMB.keys()) if (!live.has(id)) sessionMemoryMB.delete(id)
    rebuildMenu(processManager, getWindow)
  }
  for (const event of ['session:created', 'session:starting', 'session:ready', 'session:stopped', 'session:error', 'session:deleted', 'session:standby']) {
    sessionManager.on(event, sessionRebuild)
    boundListeners.push({ event, fn: sessionRebuild })
  }

  for (const event of ['started', 'stopped']) {
    apiGateway.on(event, sessionRebuild)
    boundListeners.push({ event: `gateway:${event}`, fn: sessionRebuild })
  }

  // Track session memory from health events — only rebuild tray when memory changes
  const sessionHealthFn = (data: any) => {
    if (data.memory?.active_mb != null) {
      const prev = sessionMemoryMB.get(data.sessionId) || 0
      const curr = Math.round(data.memory.active_mb)
      if ((curr === 0 && prev !== 0) || Math.abs(curr - prev) >= 10) {  // Only rebuild if change >= 10 MB
        sessionMemoryMB.set(data.sessionId, curr)
        rebuildMenu(processManager, getWindow)
      } else if (prev === 0 && curr > 0) {
        sessionMemoryMB.set(data.sessionId, curr)
        rebuildMenu(processManager, getWindow)
      }
    }
  }
  sessionManager.on('session:memory', sessionHealthFn)
  boundListeners.push({ event: 'session:memory', fn: sessionHealthFn })
  sessionManager.on('session:health', sessionHealthFn)
  boundListeners.push({ event: 'session:health', fn: sessionHealthFn })

  // Clean up session memory tracking when sessions stop or are deleted
  const sessionCleanupFn = (data: any) => {
    if (data?.sessionId) sessionMemoryMB.delete(data.sessionId)
  }
  sessionManager.on('session:stopped', sessionCleanupFn)
  sessionManager.on('session:deleted', sessionCleanupFn)
  boundListeners.push({ event: 'session:stopped', fn: sessionCleanupFn })
  boundListeners.push({ event: 'session:deleted', fn: sessionCleanupFn })

  return tray
}

/**
 * Destroy the tray and clean up event listeners.
 */
export function destroyTray(): void {
  // Remove ProcessManager listeners to prevent leaks on recreate
  if (boundProcessManager) {
    for (const { event, fn } of boundListeners) {
      // Process events go to ProcessManager, session events go to SessionManager
      if (event.startsWith('gateway:')) {
        apiGateway.off(event.slice('gateway:'.length), fn)
      } else if (event.startsWith('session:')) {
        sessionManager.off(event, fn)
      } else {
        boundProcessManager.off(event, fn)
      }
    }
    boundListeners.length = 0
    boundProcessManager = null
  }
  sessionMemoryMB.clear()
  if (tray) {
    tray.destroy()
    tray = null
  }
}

/**
 * Check if tray exists.
 */
export function hasTray(): boolean {
  return tray !== null
}

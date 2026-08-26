import { ipcMain, BrowserWindow, dialog } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import { sessionManager } from '../sessions'
import { db } from '../database'
import {
  StreamingOperationLifecycle,
  type StreamingOperationResult,
} from './streaming-operation-lifecycle'
import { attachChildProcessStreamErrorGuard } from '../childProcessStreamGuards'
import { normalizeHfTokenSetting } from '../../shared/hfSettings'

const operationLifecycle = new StreamingOperationLifecycle<ChildProcess>()
let bufferedLogLines: string[] = []  // Persists across component mounts for reconnection
let bufferedOperationDescriptor: { command: string; args: string[] } | null = null



function buildCliEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = {
    ...process.env,
    PYTHONDONTWRITEBYTECODE: '1',
    PYTHONNOUSERSITE: '1',
    PYTHONPATH: undefined,
  }
  try {
    // Same normalize-on-read contract as the session spawn path.
    const hfToken = normalizeHfTokenSetting(db.getSetting('hf_api_key'))
    if (hfToken) env.HF_TOKEN = hfToken
  } catch { /* DB not ready yet — skip token injection */ }
  return env
}

/** Resolve the CLI spawn command + args using the same path as sessions.ts */
function resolveCliSpawn(subcommandArgs: string[]): { cmd: string; args: string[]; env: Record<string, string | undefined> } {
  const engineResult = sessionManager.findEnginePath()
  const env = buildCliEnv()
  if (engineResult?.type === 'bundled' || engineResult?.type === 'development') {
    if (engineResult.type === 'development') {
      env.PYTHONPATH = engineResult.sourceRoot
    }
    return {
      cmd: engineResult.pythonPath,
      args: ['-B', '-s', '-m', 'vmlx_engine.cli', ...subcommandArgs],
      env,
    }
  }
  if (engineResult?.type === 'system') {
    return {
      cmd: engineResult.binaryPath,
      args: subcommandArgs,
      env,
    }
  }
  // Last resort: bare python3 (may not have vmlx_engine installed)
  return {
    cmd: 'python3',
    args: ['-B', '-s', '-m', 'vmlx_engine.cli', ...subcommandArgs],
    env,
  }
}

function emitLog(getWin: () => BrowserWindow | null, data: string) {
  const win = getWin()
  if (win && !win.isDestroyed()) {
    win.webContents.send('developer:log', { data })
  }
}

function emitComplete(getWin: () => BrowserWindow | null, result: StreamingOperationResult) {
  const win = getWin()
  if (win && !win.isDestroyed()) {
    win.webContents.send('developer:complete', result)
  }
}

/** Run a quick CLI command that buffers output and returns it. 30s timeout. */
async function runQuickCommand(subcommand: string, args: string[]): Promise<{ success: boolean; output: string; error?: string }> {
  const spawn_info = resolveCliSpawn([subcommand, ...args])
  return new Promise((resolve) => {
    const proc = spawn(spawn_info.cmd, spawn_info.args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: spawn_info.env,
    })
    let stdout = ''
    let stderr = ''
    let resolved = false

    proc.stdout?.on('data', (d) => { stdout += d.toString() })
    proc.stderr?.on('data', (d) => { stderr += d.toString() })
    attachChildProcessStreamErrorGuard(proc.stdout, (err) => {
      stderr += `\nstdout stream error: ${err.message}`
    })
    attachChildProcessStreamErrorGuard(proc.stderr, (err) => {
      stderr += `\nstderr stream error: ${err.message}`
    })

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true
        proc.kill('SIGKILL')
        resolve({ success: false, output: '', error: 'Command timed out after 30 seconds' })
      }
    }, 30000)

    proc.on('close', (code) => {
      clearTimeout(timer)
      if (!resolved) {
        resolved = true
        resolve({
          success: code === 0,
          output: stdout,
          error: code !== 0 ? stderr || 'Command failed' : undefined,
        })
      }
    })
    proc.on('error', (err) => {
      clearTimeout(timer)
      if (!resolved) {
        resolved = true
        resolve({ success: false, output: '', error: err.message })
      }
    })
  })
}

/** Run a long-running CLI command that streams log output via IPC events. */
async function runStreamingCommand(
  getWin: () => BrowserWindow | null,
  cliArgs: string[],
  failureMessage: string,
): Promise<{ success: boolean; cancelled?: boolean; error?: string }> {
  if (operationLifecycle.isRunning) {
    return { success: false, error: 'Another operation is already running' }
  }

  const spawn_info = resolveCliSpawn(cliArgs)
  bufferedLogLines = []  // Clear previous buffer
  bufferedOperationDescriptor = {
    command: cliArgs[0] || '',
    args: cliArgs.slice(1),
  }

  return new Promise((resolve) => {
    const proc = spawn(spawn_info.cmd, spawn_info.args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: spawn_info.env,
    })
    const operation = operationLifecycle.start(proc)
    const settle = (result: StreamingOperationResult) => {
      const terminal = operationLifecycle.finish(operation, result)
      if (!terminal) return
      emitComplete(getWin, terminal)
      resolve(terminal)
    }

    const bufferAndEmit = (raw: string) => {
      const lines = raw.split('\n').filter((l: string) => l.length > 0)
      bufferedLogLines.push(...lines)
      // Cap buffer to prevent unbounded growth
      if (bufferedLogLines.length > 2000) bufferedLogLines = bufferedLogLines.slice(-2000)
      emitLog(getWin, raw)
    }
    proc.stdout?.on('data', (data) => bufferAndEmit(data.toString()))
    proc.stderr?.on('data', (data) => bufferAndEmit(data.toString()))
    attachChildProcessStreamErrorGuard(proc.stdout, (err) => {
      bufferAndEmit(`stdout stream error: ${err.message}\n`)
    })
    attachChildProcessStreamErrorGuard(proc.stderr, (err) => {
      bufferAndEmit(`stderr stream error: ${err.message}\n`)
    })

    proc.on('close', (code) => {
      const success = code === 0
      settle({ success, cancelled: false, error: success ? undefined : failureMessage })
    })
    proc.on('error', (err) => {
      settle({ success: false, error: err.message })
    })
  })
}

export function registerDeveloperHandlers(getWin: () => BrowserWindow | null) {
  ipcMain.handle('developer:info', async (_e, modelPath: string) => {
    return runQuickCommand('info', [modelPath])
  })

  ipcMain.handle('developer:doctor', async (_e, modelPath: string, options: { noInference?: boolean }) => {
    const args = ['doctor', modelPath]
    if (options?.noInference) args.push('--no-inference')
    return runStreamingCommand(getWin, args, 'Doctor found issues')
  })

  ipcMain.handle('developer:convert', async (_e, args: {
    model: string
    output?: string
    bits: number
    groupSize: number
    mode?: string
    dtype?: string
    force?: boolean
    skipVerify?: boolean
    trustRemoteCode?: boolean
    jangProfile?: string
    jangMethod?: string
    calibrationMethod?: string
    imatrixPath?: string
    useAwq?: boolean
    awqAlpha?: number
  }) => {
    const cliArgs = ['convert', args.model]

    if (args.jangProfile) {
      // JANG adaptive quantization
      cliArgs.push('--jang-profile', args.jangProfile)
      if (args.jangMethod) cliArgs.push('--jang-method', args.jangMethod)
      if (args.calibrationMethod && args.calibrationMethod !== 'weights') {
        cliArgs.push('--calibration-method', args.calibrationMethod)
      }
      if (args.imatrixPath) cliArgs.push('--imatrix-path', args.imatrixPath)
      if (args.useAwq) cliArgs.push('--use-awq')
      if (args.awqAlpha !== undefined && args.awqAlpha !== 0.25) {
        cliArgs.push('--awq-alpha', args.awqAlpha.toString())
      }
    } else {
      // MLX uniform quantization
      cliArgs.push('--bits', args.bits.toString(), '--group-size', args.groupSize.toString())
      if (args.mode && args.mode !== 'default') cliArgs.push('--mode', args.mode)
      if (args.dtype) cliArgs.push('--dtype', args.dtype)
    }

    if (args.output) cliArgs.push('--output', args.output)
    if (args.force) cliArgs.push('--force')
    if (args.skipVerify) cliArgs.push('--skip-verify')
    if (args.trustRemoteCode) cliArgs.push('--trust-remote-code')
    return runStreamingCommand(getWin, cliArgs, 'Conversion failed')
  })

  ipcMain.handle('developer:cancelOp', async () => {
    if (operationLifecycle.cancel()) return { success: true }
    return { success: false, error: 'No active operation' }
  })

  // Query whether a conversion/operation is currently running (for reconnection after navigation)
  ipcMain.handle('developer:isRunning', async () => {
    return { running: operationLifecycle.isRunning }
  })

  // Get buffered log lines from the current/last operation (for reconnection after navigation)
  ipcMain.handle('developer:getBufferedLogs', async () => {
    return {
      lines: bufferedLogLines,
      running: operationLifecycle.isRunning,
      result: operationLifecycle.lastResult,
      operation: bufferedOperationDescriptor,
    }
  })

  ipcMain.handle('developer:browseOutputDir', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory', 'createDirectory'],
      title: 'Select Output Directory',
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return result.filePaths[0]
  })
}

export function killActiveOperation() {
  operationLifecycle.cancel()
}

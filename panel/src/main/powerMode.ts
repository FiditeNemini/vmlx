import { execFile } from 'node:child_process'
import { parsePowerMode, type PowerModeStatus } from '../shared/powerMode'

function readPmset(argument: 'custom' | 'batt'): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile('/usr/bin/pmset', ['-g', argument], {
      timeout: 2000, maxBuffer: 64 * 1024, encoding: 'utf8',
    }, (error, stdout) => error ? reject(error) : resolve(stdout))
  })
}

export function createPowerModeReader(
  platform = process.platform,
  read: (argument: 'custom' | 'batt') => Promise<string> = readPmset,
): () => Promise<PowerModeStatus> {
  let pending: Promise<PowerModeStatus> | undefined
  return () => {
    if (platform !== 'darwin') return Promise.resolve({ mode: 'unsupported', source: 'unknown' })
    if (!pending) {
      // Coalesce concurrent focus/visibility events. No timer or token polling.
      pending = Promise.all([read('custom'), read('batt')])
        .then(([custom, battery]) => parsePowerMode(custom, battery))
        .catch((): PowerModeStatus => ({ mode: 'unknown', source: 'unknown' }))
        .finally(() => { pending = undefined })
    }
    return pending
  }
}

export const getPowerMode = createPowerModeReader()

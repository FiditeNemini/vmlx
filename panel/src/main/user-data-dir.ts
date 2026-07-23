import { app } from 'electron'
import { mkdirSync } from 'fs'
import { resolve } from 'path'

function valueFromArgv(argv: string[]): string | undefined {
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg.startsWith('--vmlx-user-data-dir=')) {
      return arg.slice('--vmlx-user-data-dir='.length)
    }
    if (arg === '--vmlx-user-data-dir') {
      return argv[i + 1]
    }
  }
  return undefined
}

export function resolveUserDataDirOverride(
  argv = process.argv,
  env = process.env,
): string | undefined {
  const raw =
    valueFromArgv(argv) ||
    env.VMLX_USER_DATA_DIR ||
    env.VMLINUX_USER_DATA_DIR
  if (!raw || !raw.trim()) return undefined
  return resolve(raw)
}

export function ensureUserDataDirExists(dir: string): void {
  mkdirSync(dir, { recursive: true })
}

export function applyUserDataDirOverride(): string | undefined {
  const dir = resolveUserDataDirOverride()
  if (!dir) return undefined
  // Electron accepts a userData override before the directory exists, but
  // better-sqlite3 does not create missing parent directories.  Ensure a
  // first-run isolated profile has a real storage root before database.ts is
  // imported and opens chats.db.
  ensureUserDataDirExists(dir)
  app.setPath('userData', dir)
  console.log(`[STARTUP] Using vMLX userData override: ${dir}`)
  return dir
}

applyUserDataDirOverride()

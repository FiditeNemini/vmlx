// One-time dated warning for Qwen3.8-Flash-Next JANG models (model_type
// qwen4_exp — base JANGQ-AI / OsaurusAI and dealignai CRACK alike):
// the models were re-uploaded with an MTP fix, and users who downloaded
// before that date must REDOWNLOAD THE ENTIRE MODEL to get it.
//
// This is deliberately the dumbest possible mechanism: no network requests,
// no version detection, no file inspection, and ZERO interaction with how
// MTP is detected, loaded, or used. When such a model is loaded, the
// renderer shows the warning once; any interaction persists the dismissal
// for this fix date, so it never reappears. Bumping MTP_FIX_DATE for a
// future fix shows the warning once more.

import { BrowserWindow } from 'electron'

import { db } from './database'
import { detectModelConfigFromDir } from './model-config-registry'

// The date the fixed Flash-Next bundles were published. Bump on a future fix.
export const MTP_FIX_DATE = '2026-09-04'

export interface MtpComponentUpdate {
  repoId: string
  modelPath: string
  /** The fix date this warning is about (doubles as the dismissal key). */
  remoteFingerprint: string
}

function dismissKey(repoId: string): string {
  return `mtp_component_dismissed:${repoId}`
}

// At most one warning per model per app run, on top of the persisted
// dismissal — repeated loads can never stack popups.
const promptedThisRun = new Set<string>()

function isFlashNextQwen4Exp(modelPath: string): boolean {
  try {
    // MODEL_TYPE_TO_FAMILY maps qwen4_exp and qwen4_exp_text -> qwen4-exp,
    // which covers every Flash-Next tier regardless of publishing org.
    return (
      String(detectModelConfigFromDir(modelPath)?.family ?? '').toLowerCase() ===
      'qwen4-exp'
    )
  } catch {
    return false
  }
}

// <org>/<name> from the bundle path — matches how the model scanner ids
// user-dir bundles (JANGQ-AI/..., dealignai/..., OsaurusAI/...).
function repoIdFromModelPath(modelPath: string): string | null {
  const parts = modelPath.split('/').filter(Boolean)
  if (parts.length < 2) return null
  return `${parts[parts.length - 2]}/${parts[parts.length - 1]}`
}

/**
 * Fire-and-forget: when the user loads a Flash-Next JANG model, show the
 * one-time dated redownload warning unless it was already dismissed for
 * this fix date. Never throws; never blocks or delays the load.
 */
export function checkMtpComponentUpdateOnLoad(
  getWindow: () => BrowserWindow | null,
  modelPath: string,
): void {
  if (process.env.VMLX_SKIP_MTP_COMPONENT_CHECK === '1') return
  void (async () => {
    try {
      if (!modelPath || !isFlashNextQwen4Exp(modelPath)) return
      const repoId = repoIdFromModelPath(modelPath)
      if (!repoId) return
      const runKey = `${repoId}@${MTP_FIX_DATE}`
      if (promptedThisRun.has(runKey)) return
      let dismissed: string | undefined
      try {
        dismissed = db.getSetting(dismissKey(repoId))
      } catch {
        dismissed = undefined
      }
      if (dismissed === MTP_FIX_DATE) return

      const win = getWindow()
      if (win && !win.isDestroyed()) {
        promptedThisRun.add(runKey)
        const update: MtpComponentUpdate = {
          repoId,
          modelPath,
          remoteFingerprint: MTP_FIX_DATE,
        }
        win.webContents.send('models:mtpComponentUpdateAvailable', update)
      }
    } catch (err) {
      console.log(
        `[MTP-UPDATE] warning skipped: ${(err as Error)?.message ?? err}`,
      )
    }
  })()
}

export function dismissMtpComponentUpdate(
  repoId: string,
  remoteFingerprint: string,
): void {
  try {
    db.setSetting(dismissKey(repoId), remoteFingerprint)
  } catch (err) {
    console.log(`[MTP-UPDATE] dismiss failed: ${(err as Error)?.message ?? err}`)
  }
}

// Exposed for tests.
export const __test = { repoIdFromModelPath }

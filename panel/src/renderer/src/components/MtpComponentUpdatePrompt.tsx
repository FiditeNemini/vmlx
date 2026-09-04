import { useEffect, useState } from 'react'
import { useTranslation } from '../i18n'
import { Modal } from './ui/Modal'

interface MtpComponentUpdate {
  repoId: string
  modelPath: string
  remoteFingerprint: string
}

/**
 * One-time dated warning: Qwen3.8 Flash-Next JANG models were re-uploaded
 * with an MTP fix on the shown date; users who downloaded earlier must
 * redownload the entire model. Two choices — Download (kicks the standard
 * full-model redownload, visible in the Download tab) or Skip. Either choice
 * (including closing the dialog) counts as "shown once": the dismissal is
 * persisted for this fix date, so the warning never reappears; a future fix
 * (a bumped date) prompts once more.
 */
export function MtpComponentUpdatePrompt() {
  const { t } = useTranslation()
  const [update, setUpdate] = useState<MtpComponentUpdate | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const unsub = window.api.app.onMtpComponentUpdateAvailable(
      (data: MtpComponentUpdate) => setUpdate(data),
    )
    return unsub
  }, [])

  if (!update) return null

  const modelName = update.repoId.split('/').pop() ?? update.repoId

  const persistAndClose = async () => {
    try {
      await window.api.app.dismissMtpComponentUpdate(
        update.repoId,
        update.remoteFingerprint,
      )
    } finally {
      setUpdate(null)
    }
  }

  const handleDownload = async () => {
    if (starting) return // double-click guard
    setStarting(true)
    setError(null)
    try {
      await window.api.models.startDownload(update.repoId)
      await persistAndClose()
    } catch (err) {
      setError((err as Error)?.message ?? String(err))
      setStarting(false)
    }
  }

  return (
    <Modal title={t('mtpFix.title')} onClose={persistAndClose}>
      <div className="space-y-4 text-sm">
        <p>
          {t('mtpFix.body', {
            model: modelName,
            date: update.remoteFingerprint,
          })}
        </p>
        {error && <p className="text-destructive text-xs">{error}</p>}
        <div className="flex items-center justify-end gap-2">
          <button
            className="px-3 py-1.5 rounded text-xs border border-border hover:bg-accent"
            onClick={persistAndClose}
          >
            {t('mtpFix.skip')}
          </button>
          <button
            className="px-3 py-1.5 rounded text-xs bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            disabled={starting}
            onClick={handleDownload}
          >
            {starting ? t('mtpFix.starting') : t('mtpFix.download')}
          </button>
        </div>
      </div>
    </Modal>
  )
}

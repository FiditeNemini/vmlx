import { useEffect, useState } from 'react'
import { useTranslation } from '../i18n'
import type { MetalMemoryNotice as Notice } from '../../../shared/metalWiredLimit'

export function MetalMemoryNotice() {
  const { t } = useTranslation()
  const [notices, setNotices] = useState<Notice[]>([])
  const [copiedId, setCopiedId] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    let eventReceived = false
    const unsubscribe = window.api.sessions.onMemoryWarnings((value) => {
      eventReceived = true
      if (active) setNotices(value)
    })
    // Recover pending notices on renderer reload without racing newer events.
    void window.api.sessions.memoryWarnings().then((value: Notice[]) => {
      if (active && !eventReceived) setNotices(value)
    }).catch(() => {})
    return () => { active = false; unsubscribe() }
  }, [])
  const notice = notices[0]
  if (!notice) return null
  const m = notice.measurement
  const act = async (operation: () => Promise<unknown>) => {
    setError('')
    try { await operation() } catch { setError(t('metalMemory.actionFailed')) }
  }
  return (
    <section role="status" aria-live="polite" data-vmlx-section="metal-memory-warning"
      className="shrink-0 max-h-[45vh] overflow-y-auto border-b border-warning/50 bg-muted px-4 py-3 text-xs space-y-2">
      <p className="text-foreground [overflow-wrap:anywhere]">
        <strong>{t('metalMemory.title', { model: notice.modelName })}</strong>{' '}
        {t('metalMemory.measured', {
          active: (m.active_bytes / 1024 ** 3).toFixed(1),
          limit: (m.limit_bytes / 1024 ** 3).toFixed(1),
          time: new Date(m.measured_at_ms).toLocaleTimeString(),
        })}
      </p>
      <p className="text-muted-foreground">{t(m.reason === 'guard_rejection' ? 'metalMemory.guard' : 'metalMemory.reached')}</p>
      {notice.command ? <>
        <p>{t('metalMemory.manual')}</p>
        <code className="block whitespace-pre-wrap break-all select-text">{notice.command}</code>
      </> : <p>{t('metalMemory.noIncrease')}</p>}
      <div className="flex flex-wrap gap-2">
        {notice.command && <button type="button" data-vmlx-control="metal-memory-copy" className="border border-border px-3 py-1.5 hover:bg-accent"
          onClick={() => void act(async () => { await window.api.sessions.copyMemoryWarningCommand(notice.id); setCopiedId(notice.id) })}>
          {t(copiedId === notice.id ? 'metalMemory.copied' : 'metalMemory.copy')}
        </button>}
        <button type="button" data-vmlx-control="metal-memory-dismiss" className="border border-border px-3 py-1.5 hover:bg-accent"
          onClick={() => void act(() => window.api.sessions.dismissMemoryWarning(notice.id, false))}>{t('metalMemory.dismiss')}</button>
        <button type="button" data-vmlx-control="metal-memory-dismiss-model" className="border border-border px-3 py-1.5 hover:bg-accent"
          onClick={() => void act(() => window.api.sessions.dismissMemoryWarning(notice.id, true))}>{t('metalMemory.dismissModel')}</button>
      </div>
      {error && <p role="alert" className="text-destructive">{error}</p>}
    </section>
  )
}

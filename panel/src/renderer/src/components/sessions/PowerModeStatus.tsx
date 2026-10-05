import { useEffect, useState } from 'react'
import { useTranslation } from '../../i18n'
import type { PowerModeStatus as Status } from '../../../../shared/powerMode'

export function PowerModeStatus() {
  const { t } = useTranslation()
  const [status, setStatus] = useState<Status | null>(null)
  useEffect(() => {
    let disposed = false
    let loading = false
    const refresh = async () => {
      if (document.visibilityState !== 'visible' || loading) return
      loading = true
      setStatus(null)
      try {
        const next = await window.api.app.getPowerMode()
        if (!disposed) setStatus(next)
      } catch {
        if (!disposed) setStatus({ mode: 'unknown', source: 'unknown' })
      } finally { loading = false }
    }
    void refresh()
    window.addEventListener('focus', refresh)
    document.addEventListener('visibilitychange', refresh)
    return () => {
      disposed = true
      window.removeEventListener('focus', refresh)
      document.removeEventListener('visibilitychange', refresh)
    }
  }, [])
  const label = t(`sessions.config.powerMode_${status?.mode ?? 'checking'}`)
  const source = status?.source === 'ac' ? t('sessions.config.powerModeAc')
    : status?.source === 'battery' ? t('sessions.config.powerModeBattery') : ''
  return <div className="text-xs text-muted-foreground" data-vmlx-control="power-mode-status" role="status">
    <span className={status?.mode === 'low' ? 'text-amber-500' : ''}>{t('sessions.config.powerModeLabel', { mode: label })}{source}</span>
    {status && status.mode !== 'high' && status.mode !== 'unsupported' && <p className="mt-1">
      {t('sessions.config.powerModeHint')}
    </p>}
    <p className="mt-1">{t('sessions.config.powerModeScope')}</p>
  </div>
}

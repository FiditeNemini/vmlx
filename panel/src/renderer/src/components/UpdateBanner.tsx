import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { useTranslation } from '../i18n'

interface UpdateInfo {
  currentVersion: string
  latestVersion: string
  url: string
  notes?: string
}

export function UpdateBanner() {
  const { t } = useTranslation()
  const [update, setUpdate] = useState<UpdateInfo | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const unsub = window.api.app.onUpdateAvailable((data: UpdateInfo) => {
      const prev = localStorage.getItem('vmlx-dismissed-update')
      if (prev !== data.latestVersion) setUpdate(data)
    })
    return unsub
  }, [])

  const handleDismiss = () => {
    setDismissed(true)
    if (update) localStorage.setItem('vmlx-dismissed-update', update.latestVersion)
  }

  if (!update || dismissed) return null

  return (
    <div className="flex items-center gap-3 px-4 py-1.5 bg-accent/50 border-b border-border text-xs">
      <span className="text-foreground">
        <strong>{t('update.banner.versionLabel', { version: update.latestVersion })}</strong>{' '}
        {t('update.banner.available')}
        {update.notes && <span className="text-muted-foreground ml-1">— {update.notes}</span>}
      </span>
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault()
          window.open(update.url)
        }}
        className="text-primary hover:text-primary/80 font-medium"
      >
        {t('update.banner.download')}
      </a>
      <span className="text-muted-foreground">|</span>
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault()
          window.open('https://github.com/jjang-ai/mlxstudio')
        }}
        className="text-muted-foreground hover:text-foreground"
      >
        {t('update.banner.starOnGitHub')}
      </a>
      <button
        onClick={handleDismiss}
        className="ml-auto text-muted-foreground hover:text-foreground"
        title={t('update.banner.dismissTitle')}
        aria-label={t('update.banner.dismissTitle')}
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

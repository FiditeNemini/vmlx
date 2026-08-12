import { useState } from 'react'
import { useTranslation } from '../../i18n'
import { Plus, Trash2, ChevronLeft, ImageIcon } from 'lucide-react'
import { useRelativeTimeTick } from '../ui/use-relative-time'
import type { ImageSessionInfo } from './ImageTab'

interface ImageHistoryProps {
  sessions: ImageSessionInfo[]
  currentId: string | null
  onSelect: (sessionId: string) => void
  onNew: () => void
  onDelete: (sessionId: string) => void
  onCollapse: () => void
}

export function ImageHistory({ sessions, currentId, onSelect, onNew, onDelete, onCollapse }: ImageHistoryProps) {
  const { t, locale } = useTranslation()
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  // Relative labels ("5m ago") go stale while mounted; re-render on the shared tick.
  useRelativeTimeTick()

  const handleDelete = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    if (confirmDeleteId === sessionId) {
      onDelete(sessionId)
      setConfirmDeleteId(null)
    } else {
      setConfirmDeleteId(sessionId)
      // Auto-clear confirmation after 3s
      setTimeout(() => setConfirmDeleteId(null), 3000)
    }
  }

  return (
    <div className="w-56 border-r border-border flex flex-col bg-background/50 flex-shrink-0">
      {/* Header */}
      <div className="p-3 border-b border-border flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{t('image.history.title')}</span>
        <div className="flex items-center gap-1">
          <button
            onClick={onNew}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title={t('image.history.newTitle')}
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onCollapse}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title={t('image.history.hideTitle')}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-auto">
        {sessions.length === 0 ? (
          <div className="p-4 text-center text-xs text-muted-foreground">
            <ImageIcon className="h-6 w-6 mx-auto mb-2 opacity-40" />
            <p>{t('image.history.empty')}</p>
          </div>
        ) : (
          <div className="py-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => onSelect(session.id)}
                className={`px-3 py-2 cursor-pointer group flex items-start justify-between transition-colors ${
                  currentId === session.id
                    ? 'bg-accent text-foreground'
                    : 'hover:bg-accent/50 text-muted-foreground hover:text-foreground'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1">
                    {session.sessionType === 'edit' ? (
                      <span className="text-[9px] px-1 py-0 rounded bg-violet-500/15 text-violet-400 flex-shrink-0">{t('image.history.editBadge')}</span>
                    ) : (
                      <span className="text-[9px] px-1 py-0 rounded bg-blue-500/15 text-blue-400 flex-shrink-0">{t('image.history.genBadge')}</span>
                    )}
                    <p className="text-xs font-medium truncate">{session.modelName}</p>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    {formatDate(session.updatedAt, t, locale)}
                  </p>
                </div>
                <button
                  onClick={(e) => handleDelete(e, session.id)}
                  className={`rounded flex-shrink-0 ml-1 transition-all ${
                    confirmDeleteId === session.id
                      ? 'text-destructive opacity-100 text-[9px] px-1 py-0.5 border border-destructive/40 bg-destructive/10'
                      : 'p-0.5 text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100'
                  }`}
                  title={confirmDeleteId === session.id ? t('image.history.confirmDeleteTitle') : t('image.history.deleteTitle')}
                >
                  {confirmDeleteId === session.id ? t('image.history.deleteConfirmShort') : <Trash2 className="h-3 w-3" />}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function formatDate(
  timestamp: number,
  t: (key: string, params?: Record<string, string | number>) => string,
  locale?: string,
): string {
  const date = new Date(timestamp)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)

  if (diffMins < 1) return t('image.history.justNow')
  if (diffMins < 60) return t('image.history.minutesAgo', { n: diffMins })

  const diffHours = Math.floor(diffMins / 60)
  if (diffHours < 24) return t('image.history.hoursAgo', { n: diffHours })

  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return t('chat.list.daysAgo', { n: diffDays })

  // Hardcoded 'en-US' printed English month names in every language.
  return date.toLocaleDateString(locale || [], { month: 'short', day: 'numeric' })
}

import { useState } from 'react'
import { X } from 'lucide-react'
import { useTranslation } from '../../i18n'

interface DirectoryManagerProps {
  userDirs: string[]
  builtinDirs: string[]
  dirError: string | null
  onAdd: (path: string) => void
  onRemove: (path: string) => void
  onBrowse: () => void
  onClearError: () => void
  description?: string
}

export function DirectoryManager({
  userDirs,
  builtinDirs,
  dirError,
  onAdd,
  onRemove,
  onBrowse,
  onClearError,
  description
}: DirectoryManagerProps) {
  const { t } = useTranslation()
  const [manualPath, setManualPath] = useState('')

  const handleAddManualPath = () => {
    const path = manualPath.trim()
    if (!path) return
    onAdd(path)
    setManualPath('')
  }

  return (
    <>
      <h3 className="text-sm font-semibold mb-3">{t('sessions.directories.title')}</h3>
      {description && (
        <p className="text-xs text-muted-foreground mb-3">{description}</p>
      )}

      {/* Built-in directories */}
      {builtinDirs.length > 0 && (
        <div className="mb-3">
          <span className="text-xs text-muted-foreground uppercase tracking-wider">{t('sessions.directories.default')}</span>
          {builtinDirs.map(dir => (
            <div key={dir} className="flex items-center gap-2 mt-1 px-2 py-1.5 bg-muted/50 rounded text-xs text-muted-foreground">
              <span className="truncate flex-1" title={dir}>{dir}</span>
              <span className="text-xs opacity-50 flex-shrink-0">{t('sessions.directories.builtIn')}</span>
            </div>
          ))}
        </div>
      )}

      {/* User directories */}
      {userDirs.length > 0 && (
        <div className="mb-3">
          <span className="text-xs text-muted-foreground uppercase tracking-wider">{t('sessions.directories.custom')}</span>
          {userDirs.map(dir => (
            <div key={dir} className="flex items-center gap-2 mt-1 px-2 py-1.5 bg-muted/50 rounded text-xs">
              <span className="truncate flex-1" title={dir}>{dir}</span>
              <button
                onClick={() => onRemove(dir)}
                className="text-destructive hover:text-destructive/80 flex-shrink-0"
                title={t('sessions.directories.removeTitle')}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Add directory */}
      <div className="flex items-center gap-2 mt-2">
        <input
          type="text"
          placeholder={t('sessions.directories.placeholder')}
          value={manualPath}
          onChange={(e) => { setManualPath(e.target.value); onClearError() }}
          onKeyDown={(e) => { if (e.key === 'Enter') handleAddManualPath() }}
          className="flex-1 px-2 py-1.5 bg-background border border-input rounded text-xs"
        />
        <button
          onClick={handleAddManualPath}
          disabled={!manualPath.trim()}
          className="px-2 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
        >
          {t('sessions.directories.add')}
        </button>
        <button
          onClick={onBrowse}
          className="px-2 py-1.5 text-xs border border-border rounded hover:bg-accent"
        >
          {t('sessions.directories.browse')}
        </button>
      </div>
      {dirError && (
        <p className="text-xs text-destructive mt-1">{dirError}</p>
      )}
    </>
  )
}

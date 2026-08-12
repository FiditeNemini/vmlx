import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Loader2, Terminal } from 'lucide-react'
import { useTranslation } from '../../i18n'

interface LogViewerProps {
  logLines: string[]
  running?: boolean
  defaultOpen?: boolean
}

export function LogViewer({ logLines, running, defaultOpen = false }: LogViewerProps) {
  const { t } = useTranslation()
  const [showLog, setShowLog] = useState(defaultOpen)
  const logRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (logRef.current && showLog) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logLines, showLog])

  if (logLines.length === 0) return null

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setShowLog(!showLog)}
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-muted hover:bg-muted/80 transition-colors text-left"
      >
        {running
          ? <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
          : <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
        }
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex-1">
          {running ? t('tools.logViewer.outputLog') : t('tools.logViewer.verboseOutput')} {t('tools.logViewer.lineCount', { n: logLines.length })}
        </span>
        {showLog
          ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
          : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        }
      </button>
      {showLog && (
        <pre
          ref={logRef}
          className="p-4 text-xs font-mono whitespace-pre-wrap overflow-auto max-h-[40vh] bg-background"
        >
          {logLines.join('\n')}
        </pre>
      )}
    </div>
  )
}

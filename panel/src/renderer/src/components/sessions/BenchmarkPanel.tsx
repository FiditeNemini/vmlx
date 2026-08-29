import { useEffect, useState } from 'react'

import { useTranslation } from '../../i18n'

type BenchmarkProfileId = 'peak' | 'representative'
type BenchmarkScenarioKind = 'decode' | 'prefill' | 'mixed'

interface BenchmarkMtpSnapshot {
  runtimeActive?: boolean
  effectiveDepth?: number
  telemetryState: 'engaged' | 'skipped' | 'stale' | 'missing'
  finalDepth?: number
  draftedTokens?: number
  acceptedTokens?: number
  acceptanceRate?: number
  skipReason?: string
}

interface BenchmarkPanelProps {
  sessionId: string
  endpoint: { host: string; port: number }
  modelPath: string
  modelName?: string
  sessionStatus: string
}

interface PromptResult {
  label: string
  scenarioId?: string
  kind?: BenchmarkScenarioKind
  profileId?: BenchmarkProfileId | 'legacy'
  profileLabel?: string
  familyLabel?: string
  trial?: number
  repetitions?: number
  ttft: number
  tps: number
  promptTokens: number
  cachedPromptTokens?: number
  uncachedPromptTokens?: number
  completionTokens: number
  totalTime: number
  decodeTime?: number
  ppSpeed: number
  maxTokens?: number
  temperature?: number
  thinkingDisabled?: boolean
  requestId?: string
  mtp?: BenchmarkMtpSnapshot
  error?: string
}

function mtpSummary(snapshot?: BenchmarkMtpSnapshot): string | null {
  if (!snapshot) return null
  if (snapshot.telemetryState === 'engaged') {
    const depth = snapshot.finalDepth || snapshot.effectiveDepth
    const acceptance =
      snapshot.acceptanceRate != null
        ? `${(snapshot.acceptanceRate * 100).toFixed(0)}%`
        : snapshot.acceptedTokens != null && snapshot.draftedTokens
          ? `${snapshot.acceptedTokens}/${snapshot.draftedTokens}`
          : null
    return [`MTP D${depth || '?'}`, acceptance ? `accept=${acceptance}` : null]
      .filter(Boolean)
      .join(' ')
  }
  if (snapshot.telemetryState === 'skipped') {
    return `AR · MTP skipped=${snapshot.skipReason || 'unspecified'}`
  }
  if (snapshot.runtimeActive === false) return 'AR · no native MTP runtime'
  if (snapshot.telemetryState === 'stale') return 'MTP telemetry stale'
  return snapshot.runtimeActive ? 'MTP telemetry missing' : null
}

interface BenchmarkRun {
  id: string
  sessionId: string
  modelPath: string
  modelName?: string
  profileId?: BenchmarkProfileId | 'legacy'
  profileLabel?: string
  familyLabel?: string
  results: PromptResult[]
  createdAt: number
}

function positiveValues(
  results: PromptResult[],
  metric: 'tps' | 'ppSpeed',
  kind?: BenchmarkScenarioKind,
): number[] {
  return results
    .filter((result) => !kind || result.kind === kind)
    .map((result) => result[metric])
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b)
}

function bestValue(values: number[]): number {
  return values.length ? values[values.length - 1] : 0
}

function medianValue(values: number[]): number {
  if (!values.length) return 0
  const middle = Math.floor(values.length / 2)
  return values.length % 2
    ? values[middle]
    : (values[middle - 1] + values[middle]) / 2
}

function HeadlineCards({ results }: { results: PromptResult[] }) {
  const { t } = useTranslation()
  const decode = positiveValues(results, 'tps', 'decode')
  const prefill = positiveValues(results, 'ppSpeed', 'prefill')
  if (!decode.length && !prefill.length) return null

  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          {t('sessions.benchmark.peakDecode')}
        </div>
        <div className="mt-1 font-mono text-2xl font-semibold text-primary">
          {decode.length ? bestValue(decode).toFixed(1) : '—'}
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            t/s
          </span>
        </div>
        {decode.length > 1 && (
          <div className="mt-1 text-[10px] text-muted-foreground">
            {t('sessions.benchmark.median')}: {medianValue(decode).toFixed(1)}{' '}
            t/s
          </div>
        )}
      </div>
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          {t('sessions.benchmark.peakPrefill')}
        </div>
        <div className="mt-1 font-mono text-2xl font-semibold text-primary">
          {prefill.length ? bestValue(prefill).toFixed(0) : '—'}
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            pp/s
          </span>
        </div>
        {prefill.length > 1 && (
          <div className="mt-1 text-[10px] text-muted-foreground">
            {t('sessions.benchmark.median')}: {medianValue(prefill).toFixed(0)}{' '}
            pp/s
          </div>
        )}
      </div>
    </div>
  )
}

export function BenchmarkPanel({
  sessionId,
  endpoint,
  modelPath,
  modelName,
  sessionStatus,
}: BenchmarkPanelProps) {
  const { t } = useTranslation()
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<{
    current: number
    total: number
    label: string
  } | null>(null)
  const [currentResults, setCurrentResults] = useState<PromptResult[] | null>(
    null,
  )
  const [currentFamily, setCurrentFamily] = useState<string | null>(null)
  const [currentDisclosure, setCurrentDisclosure] = useState<string | null>(
    null,
  )
  const [history, setHistory] = useState<BenchmarkRun[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [flushCache, setFlushCache] = useState(true)
  const [profileId, setProfileId] = useState<BenchmarkProfileId>('peak')

  useEffect(() => {
    loadHistory()
  }, [modelPath])

  useEffect(() => {
    const unsub = window.api.benchmark.onProgress((data: any) => {
      if (data.sessionId === sessionId) {
        setProgress({
          current: data.current,
          total: data.total,
          label: data.label,
        })
      }
    })
    return unsub
  }, [sessionId])

  const loadHistory = async () => {
    try {
      const loaded = await window.api.benchmark.history(modelPath)
      setHistory(loaded)
    } catch {
      // History is optional; a database read failure must not block a new run.
    }
  }

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setCurrentResults(null)
    setCurrentFamily(null)
    setCurrentDisclosure(null)
    setProgress(null)

    try {
      const result = await window.api.benchmark.run(
        sessionId,
        endpoint,
        modelPath,
        modelName,
        { flushCache, profileId },
      )
      setCurrentResults(result.results)
      setCurrentFamily(result.familyLabel)
      setCurrentDisclosure(result.disclosure)
      await loadHistory()
    } catch (runError: any) {
      setError(runError.message || t('sessions.benchmark.failed'))
    } finally {
      setRunning(false)
      setProgress(null)
    }
  }

  const handleDelete = async (id: string) => {
    await window.api.benchmark.delete(id)
    setHistory((previous) => previous.filter((run) => run.id !== id))
  }

  if (sessionStatus !== 'running') {
    return (
      <div className="p-4 text-sm text-muted-foreground">
        {t('sessions.benchmark.sessionMustBeRunning')}
      </div>
    )
  }

  const selectedDisclosure =
    profileId === 'peak'
      ? t('sessions.benchmark.peakDisclosure')
      : t('sessions.benchmark.representativeDisclosure')

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      <div>
        <div className="mb-2 flex w-fit rounded-md border border-border bg-muted/40 p-0.5">
          {(['peak', 'representative'] as BenchmarkProfileId[]).map((id) => (
            <button
              key={id}
              type="button"
              disabled={running}
              onClick={() => setProfileId(id)}
              className={`rounded px-2.5 py-1 text-xs transition-colors disabled:opacity-50 ${
                profileId === id
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {id === 'peak'
                ? t('sessions.benchmark.profilePeak')
                : t('sessions.benchmark.profileRepresentative')}
            </button>
          ))}
        </div>
        <div className="mb-3 rounded border border-border/70 bg-muted/20 px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
          {selectedDisclosure}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={running}
            className="rounded bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {running
              ? t('sessions.benchmark.running')
              : t('sessions.view.benchTitle')}
          </button>
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={flushCache}
              onChange={(event) => setFlushCache(event.target.checked)}
              className="rounded border-border"
            />
            {t('sessions.benchmark.flushCacheFirst')}
          </label>
        </div>
        {progress && (
          <div className="mt-2 text-xs text-muted-foreground">
            {progress.label} ({progress.current}/{progress.total})
            <div className="mt-1 h-1.5 w-full rounded bg-muted">
              <div
                className="h-full rounded bg-primary transition-all"
                style={{
                  width: `${(progress.current / progress.total) * 100}%`,
                }}
              />
            </div>
          </div>
        )}
      </div>

      {currentResults && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] text-muted-foreground">
            <span>{currentFamily}</span>
            <span>{currentResults[0]?.profileLabel}</span>
          </div>
          <HeadlineCards results={currentResults} />
          <ResultsTable results={currentResults} />
          {currentDisclosure && (
            <div className="text-[10px] leading-relaxed text-muted-foreground">
              {currentDisclosure}
            </div>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            {showHistory ? t('app.about.hide') : t('app.about.show')}{' '}
            {t('sessions.benchmark.history', { n: history.length })}
          </button>
          {showHistory && (
            <div className="mt-2 max-h-80 space-y-3 overflow-auto">
              {history.map((run) => {
                const decodeRows = positiveValues(run.results, 'tps', 'decode')
                const prefillRows = positiveValues(
                  run.results,
                  'ppSpeed',
                  'prefill',
                )
                const decode = decodeRows.length
                  ? decodeRows
                  : positiveValues(run.results, 'tps')
                const prefill = prefillRows.length
                  ? prefillRows
                  : positiveValues(run.results, 'ppSpeed')
                return (
                  <div
                    key={run.id}
                    className="rounded border border-border bg-background p-3"
                  >
                    <div className="mb-2 flex items-center justify-between">
                      <div>
                        <div className="text-xs">
                          {run.profileLabel || 'Legacy'}
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          {run.familyLabel} ·{' '}
                          {new Date(run.createdAt).toLocaleString()}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono">
                          {decode.length
                            ? `${bestValue(decode).toFixed(1)} t/s`
                            : '—'}
                          {' · '}
                          {prefill.length
                            ? `${bestValue(prefill).toFixed(0)} pp/s`
                            : '—'}
                        </span>
                        <button
                          onClick={() => handleDelete(run.id)}
                          className="text-xs text-destructive hover:text-destructive/80"
                        >
                          ×
                        </button>
                      </div>
                    </div>
                    <ResultsTable results={run.results} compact />
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ResultsTable({
  results,
  compact,
}: {
  results: PromptResult[]
  compact?: boolean
}) {
  const { t } = useTranslation()
  const decodeRows = positiveValues(results, 'tps', 'decode')
  const prefillRows = positiveValues(results, 'ppSpeed', 'prefill')
  const allTps = decodeRows.length ? decodeRows : positiveValues(results, 'tps')
  const allPrefill = prefillRows.length
    ? prefillRows
    : positiveValues(results, 'ppSpeed')
  const validTtft = results
    .map((result) => result.ttft)
    .filter((value) => Number.isFinite(value) && value > 0)
  return (
    <div className="overflow-auto">
      <table className={`w-full text-xs ${compact ? '' : 'mt-2'}`}>
        <thead>
          <tr className="border-b border-border text-muted-foreground">
            <th className="py-1 pr-2 text-left">
              {t('sessions.benchmark.headerTest')}
            </th>
            <th className="px-1 py-1 text-right">TTFT</th>
            <th className="px-1 py-1 text-right">TPS</th>
            <th className="px-1 py-1 text-right">PP/s</th>
            <th className="py-1 pl-1 text-right">
              {t('sessions.benchmark.headerTime')}
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => {
            const trialLabel =
              result.repetitions && result.repetitions > 1
                ? ` #${result.trial || index + 1}`
                : ''
            const settings = [
              result.maxTokens != null ? `max=${result.maxTokens}` : null,
              result.temperature != null ? `temp=${result.temperature}` : null,
              result.thinkingDisabled ? 'thinking=off' : null,
              result.cachedPromptTokens
                ? `cached=${result.cachedPromptTokens}`
                : 'cache=0',
              mtpSummary(result.mtp),
              result.error || null,
            ]
              .filter(Boolean)
              .join(' · ')
            return (
              <tr
                key={`${result.scenarioId || result.label}-${index}`}
                className="border-b border-border/50"
              >
                <td
                  className="max-w-[160px] truncate py-1 pr-2"
                  title={`${result.label}${trialLabel} · ${settings}`}
                >
                  {result.label}
                  {trialLabel}
                </td>
                <td className="px-1 py-1 text-right font-mono">
                  {result.ttft > 0
                    ? `${(result.ttft * 1000).toFixed(0)}ms`
                    : '—'}
                </td>
                <td className="px-1 py-1 text-right font-mono font-medium">
                  {result.tps > 0 ? result.tps.toFixed(1) : '—'}
                </td>
                <td className="px-1 py-1 text-right font-mono">
                  {result.ppSpeed > 0 ? result.ppSpeed.toFixed(0) : '—'}
                </td>
                <td className="py-1 pl-1 text-right font-mono">
                  {result.totalTime > 0
                    ? `${result.totalTime.toFixed(1)}s`
                    : '—'}
                </td>
              </tr>
            )
          })}
          <tr className="font-medium">
            <td className="py-1 pr-2">{t('sessions.benchmark.rowBest')}</td>
            <td className="px-1 py-1 text-right font-mono">
              {validTtft.length
                ? `${(Math.min(...validTtft) * 1000).toFixed(0)}ms`
                : '—'}
            </td>
            <td className="px-1 py-1 text-right font-mono">
              {allTps.length ? bestValue(allTps).toFixed(1) : '—'}
            </td>
            <td className="px-1 py-1 text-right font-mono">
              {allPrefill.length ? bestValue(allPrefill).toFixed(0) : '—'}
            </td>
            <td className="py-1 pl-1 text-right font-mono">
              {results
                .reduce((sum, result) => sum + result.totalTime, 0)
                .toFixed(1)}
              s
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

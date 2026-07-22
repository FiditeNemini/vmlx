import { useState, useEffect, useRef } from 'react'
import { X } from 'lucide-react'
import { SessionConfigForm, SessionConfig, DEFAULT_CONFIG, SliderField, DSV4_PAGED_CACHE_BLOCK_SIZE, commitActiveSettingsInput } from './SessionConfigForm'
import { useInferenceMode } from '../layout/InferenceMode'
import { useTranslation } from '../../i18n'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  config: string
}

interface ServerSettingsDrawerProps {
  session: Session
  isRemote?: boolean
  onClose: () => void
  onSessionUpdate?: () => void
}

export function ServerSettingsDrawer({ session, isRemote, onClose, onSessionUpdate }: ServerSettingsDrawerProps) {
  const { defaultConfig } = useInferenceMode()
  const { t } = useTranslation()
  const [config, setConfig] = useState<SessionConfig>(defaultConfig)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [detectedCacheType, setDetectedCacheType] = useState<string>('kv')
  const [detectedUsePagedCache, setDetectedUsePagedCache] = useState<boolean | undefined>(undefined)
  const [detectedCacheSubtype, setDetectedCacheSubtype] = useState<string | undefined>()
  const [detectedFamily, setDetectedFamily] = useState<string | undefined>()
  const [detectedIsTurboQuant, setDetectedIsTurboQuant] = useState<boolean>(false)
  const [detectedIsMultimodal, setDetectedIsMultimodal] = useState<boolean>(false)
  const [detectedForceTextOnly, setDetectedForceTextOnly] = useState<boolean>(false)
  const [detectedMaxContext, setDetectedMaxContext] = useState<number | undefined>()
  const [detectedNativeMtp, setDetectedNativeMtp] = useState<any>(undefined)
  const [singleModelMode, setSingleModelMode] = useState(false)
  const restartingRef = useRef(false)
  restartingRef.current = restarting

  useEffect(() => {
    try {
      const stored = JSON.parse(session.config)
      // Always use DB columns as canonical source for host/port to prevent mismatch
      setConfig({ ...DEFAULT_CONFIG, ...stored, host: session.host, port: session.port })
    } catch {
      setConfig({ ...DEFAULT_CONFIG, host: session.host, port: session.port })
    }
    setDirty(false)
    setMessage(null)
    // Detect model cache type for feature gating
    if (session.modelPath) {
      window.api.models.detectConfig(session.modelPath)
        .then((det: any) => {
          if (det?.cacheType) setDetectedCacheType(det.cacheType)
          setDetectedUsePagedCache(det?.usePagedCache)
          setDetectedCacheSubtype(det?.cacheSubtype)
          if (det?.family && det.family !== 'unknown') setDetectedFamily(det.family)
          else setDetectedFamily(undefined)
          setDetectedIsTurboQuant(!!det?.isTurboQuant)
          setDetectedIsMultimodal(!!det?.isMultimodal)
          setDetectedForceTextOnly(!!det?.forceTextOnly)
          setDetectedNativeMtp(det?.nativeMtp)
          if (det?.maxContextLength) setDetectedMaxContext(det.maxContextLength)
        })
        .catch((err) => console.error('Failed to detect model config:', err))
    }
  }, [session.id, session.config, session.host, session.port])

  useEffect(() => {
    window.api.gateway?.getStatus?.()
      .then((status: any) => setSingleModelMode(!!status?.singleModelMode))
      .catch(() => {})
  }, [])

  useEffect(() => {
    const unsubscribe = window.api.gateway?.onSingleModelModeChanged?.((data: { singleModelMode: boolean }) => {
      setSingleModelMode(!!data?.singleModelMode)
    })
    return () => { unsubscribe?.() }
  }, [])

  // Listen for restart completion
  useEffect(() => {
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === session.id) {
        setRestarting(false)
        setMessage({ type: 'success', text: 'Restarted with new settings.' })
        onSessionUpdate?.()
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === session.id && restartingRef.current) {
        setRestarting(false)
        setMessage({ type: 'error', text: `Restart failed: ${data.error}` })
      }
    })
    return () => {
      unsubReady()
      unsubError()
    }
  }, [session.id])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setDirty(true)
    setMessage(null)
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const result = await window.api.sessions.update(session.id, config)
      if (result.success) {
        setDirty(false)
        setMessage({
          type: 'success',
          text: isRemote ? 'Saved. Applies to next request.' : (
            result.restartRequired ? `Saved. Restart to apply (${result.changedKeys?.join(', ')}).` : 'Settings saved.'
          )
        })
        onSessionUpdate?.()
      } else {
        setMessage({ type: 'error', text: result.error || 'Failed to save' })
      }
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  const handleSaveAndRestart = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const saveResult = await window.api.sessions.update(session.id, config)
      if (!saveResult.success) {
        setMessage({ type: 'error', text: saveResult.error || 'Failed to save' })
        setSaving(false)
        return
      }
      setDirty(false)
      setRestarting(true)
      setMessage({ type: 'success', text: 'Stopping...' })

      const stopResult = await window.api.sessions.stop(session.id)
      if (!stopResult.success) {
        setMessage({ type: 'error', text: `Failed to stop: ${stopResult.error}` })
        setRestarting(false)
        setSaving(false)
        return
      }

      // sessions.stop resolves only after stopSession has killed the process,
      // cleared its durable PID, and emitted session:stopped. Subscribing after
      // that await misses the event and adds a guaranteed timeout delay.
      setMessage({ type: 'success', text: 'Starting with new settings...' })
      const startResult = await window.api.sessions.start(session.id)
      if (!startResult.success) {
        setMessage({ type: 'error', text: `Failed to start: ${startResult.error}` })
        setRestarting(false)
      }
      onSessionUpdate?.()
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
      setRestarting(false)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    const base = { ...DEFAULT_CONFIG, host: config.host, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (session.modelPath) {
      try {
        const detected = await window.api.models.detectConfig(session.modelPath)
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = detected.enableAutoToolChoice
          if (detected.family === 'deepseek-v4') {
            base.dsv4PrefixCache = true
            base.dsv4PoolQuant = true
            base.enablePrefixCache = true
            base.usePagedCache = true
            base.enableBlockDiskCache = true
            base.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
          } else if (detected.family === 'openpangu_v2') {
            base.enablePrefixCache = true
            base.usePagedCache = false
            base.enableDiskCache = true
            base.enableBlockDiskCache = false
            base.noMemoryAwareCache = false
            base.kvCacheQuantization = 'none'
          } else {
            base.usePagedCache = detected.usePagedCache
            base.enableDiskCache = false
            base.enableBlockDiskCache = true
          }
          setDetectedFamily(detected.family)
          setDetectedCacheSubtype(detected.cacheSubtype)
          setDetectedIsTurboQuant(!!detected.isTurboQuant)
          setDetectedIsMultimodal(!!detected.isMultimodal)
          setDetectedForceTextOnly(!!detected.forceTextOnly)
        } else {
          setDetectedFamily(undefined)
          setDetectedCacheSubtype(undefined)
          setDetectedIsTurboQuant(false)
          setDetectedIsMultimodal(false)
          setDetectedForceTextOnly(false)
        }
      } catch (_) {
        setDetectedFamily(undefined)
        setDetectedCacheSubtype(undefined)
        setDetectedIsTurboQuant(false)
        setDetectedIsMultimodal(false)
        setDetectedForceTextOnly(false)
      }
    }
    setConfig(base)
    setDirty(true)
    setMessage(null)
  }

  const handleGatewaySingleModelModeToggle = async () => {
    const next = !singleModelMode
    setSingleModelMode(next)
    try {
      const status = await window.api.gateway?.setSingleModelMode?.(next)
      if (typeof status?.singleModelMode === 'boolean') {
        setSingleModelMode(status.singleModelMode)
      }
    } catch (e) {
      setSingleModelMode(!next)
      setMessage({ type: 'error', text: (e as Error).message })
    }
  }

  const isRunning = session.status === 'running' || session.status === 'loading'

  return (
    <div className="w-full max-w-96 h-full border-l border-border bg-card flex flex-col overflow-hidden flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border flex-shrink-0">
        <span className="font-medium text-sm">{isRemote ? t('sessions.view.connectionTitle') : t('sessions.view.serverSettingsTitle')}</span>
        <button onClick={onClose} aria-label={t('common.close')} title={t('common.close')} className="text-muted-foreground hover:text-foreground text-sm px-1">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* Status message */}
        {message && (
          <div className={`p-2 rounded text-xs ${message.type === 'success'
              ? 'bg-primary/10 border border-primary/30 text-primary'
              : 'bg-destructive/10 border border-destructive/30 text-destructive'
            }`}>
            {message.text}
          </div>
        )}

        {isRunning && !restarting && !isRemote && (
          <div className="p-2 bg-warning/10 border border-warning/30 rounded text-xs text-warning">
            {t('sessions.settings.runningWarning')}
          </div>
        )}

        {!isRemote && (
          <div className="p-3 rounded border border-border bg-background/60 space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium">{t('main.tray.singleModelMode')}</div>
                <div className="text-xs text-muted-foreground">
                  {singleModelMode ? t('api.singleModelModeOn') : t('api.singleModelModeOff')}
                </div>
              </div>
              <button
                onClick={handleGatewaySingleModelModeToggle}
                aria-label={t('main.tray.singleModelMode')}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors flex-shrink-0 ${singleModelMode ? 'bg-primary' : 'bg-muted'}`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${singleModelMode ? 'translate-x-[18px]' : 'translate-x-0.5'}`}
                />
              </button>
            </div>
          </div>
        )}

        {/* Config Form — remote sessions only show timeout */}
        {isRemote ? (
          <div className="space-y-3">
            <SliderField
              label={t('sessions.config.timeout')}
              tooltip="Maximum time to wait for a response from the remote server before timing out. Increase this for slow models, long generations, or high-latency connections. Default 300s (5 minutes)."
              value={config.timeout}
              onChange={v => handleChange('timeout', v)}
              min={10}
              max={3600}
              step={10}
              defaultValue={DEFAULT_CONFIG.timeout}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.config.timeoutNoLimit')}
            />
          </div>
        ) : (
          <SessionConfigForm config={config} onChange={handleChange} detectedCacheType={detectedCacheType} detectedUsePagedCache={detectedUsePagedCache} detectedCacheSubtype={detectedCacheSubtype} detectedFamily={detectedFamily} detectedIsTurboQuant={detectedIsTurboQuant} detectedIsMultimodal={detectedIsMultimodal} detectedForceTextOnly={detectedForceTextOnly} detectedMaxContext={detectedMaxContext} detectedNativeMtp={detectedNativeMtp} modelType={(() => { try { return JSON.parse(session.config || '{}').modelType } catch { return undefined } })()} imageMode={(() => { try { return JSON.parse(session.config || '{}').imageMode } catch { return undefined } })()} sessionId={session.id} modelIdentity={`${session.modelName || ''} ${session.modelPath}`} />
        )}
      </div>

      {/* Footer Actions */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3 border-t border-border flex-shrink-0">
        <button
          onPointerDown={commitActiveSettingsInput}
          onClick={handleSave}
          disabled={!dirty || saving || restarting}
          className="flex-1 px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
        >
          {saving && !restarting
            ? t('common.saving')
            : t(isRunning ? 'sessions.settings.saveForNextRestart' : 'common.save')}
        </button>
        {isRunning && !isRemote && (
          <button
            onPointerDown={commitActiveSettingsInput}
            onClick={handleSaveAndRestart}
            disabled={saving || restarting}
            className="flex-1 px-3 py-1.5 text-sm bg-success text-success-foreground rounded hover:bg-success/90 disabled:opacity-40"
          >
            {restarting ? t('sessions.settings.restarting') : t('sessions.settings.saveAndRestart')}
          </button>
        )}
        <button
          onClick={handleReset}
          disabled={restarting}
          className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent disabled:opacity-40"
        >
          {t('common.reset')}
        </button>
      </div>
    </div>
  )
}

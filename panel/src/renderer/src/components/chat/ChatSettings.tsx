import { useState, useEffect, useRef, useCallback } from 'react'
import { AlertTriangle, X, Save, Trash2, Star } from 'lucide-react'
import { useToast } from '../Toast'
import { useTranslation } from '../../i18n'
import { buildChatSettingsCompatibilityWarnings } from './chatSettingsCompatibility'
import { buildChatSettingsResetOverrides } from '../../../../shared/chatSettingsResetPolicy'
import { applyEffectiveSessionGenerationDefaults } from '../../../../shared/effectiveGenerationDefaults'
import {
  reasoningParserIsEnabled,
  resolveEffectiveReasoningParser,
} from '../../../../shared/reasoningParserAliases'

interface ChatProfile {
  id: string
  name: string
  overrides: ChatOverrides
  isDefault: boolean
  createdAt: number
}

interface ChatOverrides {
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  maxTokens?: number
  maxThinkingTokens?: number
  repeatPenalty?: number
  systemPrompt?: string
  stopSequences?: string
  wireApi?: 'completions' | 'responses'
  maxToolIterations?: number
  builtinToolsEnabled?: boolean
  workingDirectory?: string
  enableThinking?: boolean
  reasoningEffort?: 'low' | 'medium' | 'high' | 'max'
  hideToolStatus?: boolean
  webSearchEnabled?: boolean
  braveSearchEnabled?: boolean
  fetchUrlEnabled?: boolean
  fileToolsEnabled?: boolean
  searchToolsEnabled?: boolean
  shellEnabled?: boolean
  toolResultMaxChars?: number
  gitEnabled?: boolean
  utilityToolsEnabled?: boolean
}

interface SessionInfo {
  modelName?: string
  modelPath: string
  host: string
  port: number
  status: string
  pid?: number
  type?: 'local' | 'remote'
  remoteUrl?: string
  modelType?: 'text' | 'image'
  config?: string
}

interface ChatSettingsProps {
  chatId: string
  session: SessionInfo
  reasoningParser?: string
  onClose: () => void
  onOverridesChanged?: () => void
}

function formatTopK(value: number): string {
  const rounded = Math.round(value)
  if (rounded <= 0) return 'Off'
  return rounded.toString()
}

const CHAT_TOP_K_SLIDER_DEFAULT_MAX = 200
const CHAT_TOP_K_HARD_MAX = 1_000_000

export function ChatSettings({ chatId, session, reasoningParser, onClose, onOverridesChanged }: ChatSettingsProps) {
  const { showToast } = useToast()
  const { t } = useTranslation()
  const [overrides, setOverrides] = useState<ChatOverrides>({})
  const [modelDefaults, setModelDefaults] = useState<Partial<ChatOverrides>>({})
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [profiles, setProfiles] = useState<ChatProfile[]>([])
  const [profileName, setProfileName] = useState('')
  const [showProfileSave, setShowProfileSave] = useState(false)
  const [detectedFamily, setDetectedFamily] = useState<string | undefined>(undefined)
  const [detectedToolParser, setDetectedToolParser] = useState<string | undefined>(undefined)
  const [detectedReasoningParser, setDetectedReasoningParser] = useState<string | undefined>(undefined)
  const [detectedSupportsThinking, setDetectedSupportsThinking] = useState<boolean | undefined>(undefined)
  const [detectedSupportsInstructMode, setDetectedSupportsInstructMode] = useState<boolean | undefined>(undefined)
  const [detectedReasoningEfforts, setDetectedReasoningEfforts] = useState<Array<'low' | 'medium' | 'high' | 'max'> | undefined>(undefined)
  const [thinkingBudgetSupported, setThinkingBudgetSupported] = useState<boolean | undefined>(undefined)
  const [supportsThinkingBudget, setSupportsThinkingBudget] = useState<boolean | undefined>(undefined)
  const [savedChatModelPath, setSavedChatModelPath] = useState<string | undefined>(undefined)
  const [messageCount, setMessageCount] = useState(0)
  const loadRequestRef = useRef(0)
  const isRemote = session.type === 'remote'
  const effectiveWireApi = overrides.wireApi ?? (isRemote ? 'completions' : 'responses')
  const configuredReasoningParser = (() => {
    try {
      const config = session.config ? JSON.parse(session.config) : {}
      return config.reasoningParser ?? reasoningParser
    } catch {
      return reasoningParser
    }
  })()
  const resolvedReasoningParser = resolveEffectiveReasoningParser({
    configuredParser: configuredReasoningParser,
    detectedParser: detectedReasoningParser,
    supportsThinking: detectedSupportsThinking,
  })
  const effectiveReasoningParser = reasoningParserIsEnabled(resolvedReasoningParser)
    ? resolvedReasoningParser
    : undefined
  const thinkingSupported = resolvedReasoningParser !== 'none' && (
    detectedFamily === 'deepseek-v4' ||
    detectedSupportsThinking === true ||
    (detectedSupportsThinking !== false && !!effectiveReasoningParser)
  )
  const thinkingOffSupported = detectedSupportsInstructMode !== false
  const displayedEnableThinking = thinkingSupported ? overrides.enableThinking : undefined
  const displayedTopK = Math.max(0, Math.round(overrides.topK ?? modelDefaults.topK ?? 0))
  const topKSliderMax = Math.min(
    CHAT_TOP_K_HARD_MAX,
    Math.max(CHAT_TOP_K_SLIDER_DEFAULT_MAX, displayedTopK),
  )
  const thinkingDisabledClass = thinkingSupported ? '' : ' opacity-50 cursor-not-allowed'
  const showReasoningEffort = (detectedReasoningEfforts?.length ?? 0) > 0 || detectedFamily === 'hy3' || effectiveReasoningParser === 'openai_gptoss' || effectiveReasoningParser === 'mistral'
  const showLowEffort = detectedReasoningEfforts ? detectedReasoningEfforts.includes('low') : effectiveReasoningParser !== 'mistral'
  const showMediumEffort = detectedReasoningEfforts ? detectedReasoningEfforts.includes('medium') : effectiveReasoningParser !== 'mistral' && detectedFamily !== 'hy3'
  const showHighEffort = detectedReasoningEfforts ? detectedReasoningEfforts.includes('high') : true
  const dsv4MaxEnabled = detectedFamily === 'deepseek-v4'

  const loadProfiles = useCallback(() => {
    window.api.chat.getProfiles().then((p: ChatProfile[]) => setProfiles(p))
  }, [])

  useEffect(() => {
    const requestId = ++loadRequestRef.current
    let active = true
    const stillCurrent = () => active && loadRequestRef.current === requestId
    void (async () => {
      const saved = await window.api.chat.getOverrides(chatId) as ChatOverrides | null
      if (!stillCurrent()) return
      let nextSavedChatModelPath: string | undefined
      let nextMessageCount = 0
      try {
        const [chat, messages] = await Promise.all([
          window.api.chat.get(chatId),
          window.api.chat.getMessages(chatId),
        ])
        if (!stillCurrent()) return
        nextSavedChatModelPath = chat?.modelPath
        nextMessageCount = Array.isArray(messages) ? messages.length : 0
      } catch (_) {
        if (!stillCurrent()) return
      }
      // Pull recommended defaults from model metadata. If the bundle does not
      // declare a value, leave it unset so the engine resolves its own fallback.
      let detectedModelDefaults: Partial<ChatOverrides> = {}
      let nextThinkingBudgetSupported: boolean | undefined
      let nextDetectedFamily: string | undefined
      let nextDetectedToolParser: string | undefined
      let nextDetectedReasoningParser: string | undefined
      let nextDetectedSupportsThinking: boolean | undefined
      let nextDetectedSupportsInstructMode: boolean | undefined
      let nextDetectedReasoningEfforts: Array<'low' | 'medium' | 'high' | 'max'> | undefined
      let nextSupportsThinkingBudget: boolean | undefined
      if (session.modelPath) {
        try {
          const gen = await window.api.models.getGenerationDefaults(session.modelPath)
          if (!stillCurrent()) return
          if (gen) {
            if (gen.temperature != null) detectedModelDefaults.temperature = gen.temperature
            if (gen.topP != null) detectedModelDefaults.topP = gen.topP
            if (gen.topK != null) detectedModelDefaults.topK = gen.topK
            if (gen.minP != null) detectedModelDefaults.minP = gen.minP
            if (gen.repeatPenalty != null) detectedModelDefaults.repeatPenalty = gen.repeatPenalty
            if (gen.maxNewTokens != null) detectedModelDefaults.maxTokens = gen.maxNewTokens
            if (gen.maxThinkingTokens != null) detectedModelDefaults.maxThinkingTokens = gen.maxThinkingTokens
            nextThinkingBudgetSupported = gen.thinkingBudgetSupported
          }
        } catch (_) {}
        try {
          const detected = await window.api.models.detectConfig(session.modelPath)
          if (!stillCurrent()) return
          nextDetectedFamily = detected?.family
          nextDetectedToolParser = detected?.toolParser
          nextDetectedReasoningParser = detected?.reasoningParser
          nextDetectedSupportsThinking = detected?.supportsThinking
          nextDetectedSupportsInstructMode = detected?.supportsInstructMode
          nextDetectedReasoningEfforts = detected?.supportedReasoningEfforts
          nextSupportsThinkingBudget = detected?.supportsThinkingBudget
          detectedModelDefaults = applyEffectiveSessionGenerationDefaults(
            detectedModelDefaults,
            session.config,
            detected?.nativeMtp,
          )
        } catch (_) {}
      }
      // Saved non-null overrides win over model defaults. SQL NULL means the
      // field is unset, not an explicit request to mask the model/app default.
      const savedExplicit = Object.fromEntries(
        Object.entries(saved || {}).filter(([, v]) => v !== null && v !== undefined),
      ) as ChatOverrides
      if (!stillCurrent()) return
      setSavedChatModelPath(nextSavedChatModelPath)
      setMessageCount(nextMessageCount)
      setThinkingBudgetSupported(nextThinkingBudgetSupported)
      setDetectedFamily(nextDetectedFamily)
      setDetectedToolParser(nextDetectedToolParser)
      setDetectedReasoningParser(nextDetectedReasoningParser)
      setDetectedSupportsThinking(nextDetectedSupportsThinking)
      setDetectedSupportsInstructMode(nextDetectedSupportsInstructMode)
      setDetectedReasoningEfforts(nextDetectedReasoningEfforts)
      setSupportsThinkingBudget(nextSupportsThinkingBudget)
      setModelDefaults(detectedModelDefaults)
      setOverrides(savedExplicit)
      setDirty(false)
    })()
    loadProfiles()
    return () => { active = false }
  }, [chatId, session.modelPath, session.config, loadProfiles])

  const update = <K extends keyof ChatOverrides>(key: K, value: ChatOverrides[K]) => {
    setOverrides(prev => ({ ...prev, [key]: value }))
    setDirty(true)
  }

  const updateThinkingMode = (
    enableThinking: boolean | undefined,
    reasoningEffort?: ChatOverrides['reasoningEffort']
  ) => {
    setOverrides(prev => ({ ...prev, enableThinking, reasoningEffort }))
    setDirty(true)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await window.api.chat.setOverrides(chatId, overrides)
      setDirty(false)
      onOverridesChanged?.()
    } catch (e) {
      showToast('error', t('chat.settings.saveFailedToast'), (e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    // Reset sampling fields to inheritance, not a sticky copy of today's bundle
    // values. The visible controls continue to render modelDefaults, while the
    // request omits the fields so the engine applies the live bundle contract.
    const defaults: ChatOverrides = buildChatSettingsResetOverrides(overrides) as ChatOverrides
    await window.api.chat.setOverrides(chatId, defaults)
    setOverrides(defaults)
    setDirty(false)
    onOverridesChanged?.()
  }

  const handleLoadProfile = async (profile: ChatProfile) => {
    setOverrides(profile.overrides)
    setDirty(true)
    showToast('success', t('chat.settings.loadedToast', { name: profile.name }))
  }

  const handleSaveProfile = async () => {
    if (!profileName.trim()) return
    try {
      await window.api.chat.saveProfile(profileName.trim(), overrides)
      setProfileName('')
      setShowProfileSave(false)
      loadProfiles()
      showToast('success', t('chat.settings.savedProfileToast', { name: profileName.trim() }))
    } catch (e) {
      showToast('error', t('chat.settings.saveProfileFailedToast'), (e as Error).message)
    }
  }

  const handleDeleteProfile = async (id: string, name: string) => {
    if (!confirm(t('chat.settings.deleteProfileConfirm', { name }))) return
    await window.api.chat.deleteProfile(id)
    loadProfiles()
    showToast('success', t('chat.settings.deletedToast', { name }))
  }

  const handleSetDefault = async (profile: ChatProfile) => {
    await window.api.chat.updateProfile(profile.id, profile.name, profile.overrides, !profile.isDefault)
    loadProfiles()
    showToast('success', t(profile.isDefault ? 'chat.settings.removeDefaultTitle' : 'chat.settings.setDefaultTitle'))
  }

  const shortModel = session.modelName || session.modelPath.split('/').pop() || session.modelPath
  const localizedStatus = session.status === 'running'
    ? t('status.running')
    : session.status === 'loading'
      ? t('status.loading')
      : session.status === 'standby'
        ? t('status.sleeping')
        : session.status === 'error'
          ? t('status.error')
          : t('status.stopped')
  const isImageModel = session.modelType === 'image'
  const compatibilityWarnings = buildChatSettingsCompatibilityWarnings({
    messageCount,
    savedChatModelPath,
    currentModelPath: session.modelPath,
    overrides,
    reasoningParser: effectiveReasoningParser,
    toolParser: detectedToolParser,
    detectedFamily,
  })

  return (
    <div className="w-full max-w-80 h-full border-l border-border bg-card flex flex-col overflow-hidden flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border flex-shrink-0">
        <span className="font-medium text-sm">{t('chat.settings.header')}</span>
        <button
          onClick={() => {
            if (dirty && !confirm(t('chat.settings.discardConfirm'))) return
            onClose()
          }}
          className="text-muted-foreground hover:text-foreground text-sm px-1"
          aria-label={t('common.close')}
          title={t('common.close')}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-5">
        {/* Server Info */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('chat.settings.serverInfo')}</h3>
          {(() => {
            const baseUrl = isRemote && session.remoteUrl
              ? session.remoteUrl.replace(/\/+$/, '')
              : `http://${session.host}:${session.port}`
            const isImage = session.modelType === 'image'
            const apiUrl = isImage
              ? `${baseUrl}/v1/images/generations`
              : effectiveWireApi === 'responses'
                ? `${baseUrl}/v1/responses`
                : `${baseUrl}/v1/chat/completions`
            return (
              <div className="space-y-1.5 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t('chat.settings.model')}</span>
                  <span className="text-right truncate ml-2 max-w-[180px]" title={session.modelPath}>{shortModel}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t('chat.settings.endpoint')}</span>
                  <span className="font-mono text-xs truncate ml-2 max-w-[180px]" title={baseUrl}>
                    {isRemote ? baseUrl : `http://${session.host}:${session.port}`}
                  </span>
                </div>
                {isRemote && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">{t('chat.settings.type')}</span>
                    <span className="text-xs text-muted-foreground">{t('chat.settings.typeRemote')}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t('chat.settings.status')}</span>
                  <span className={session.status === 'running' ? 'text-primary' : 'text-destructive'}>
                    {localizedStatus}
                  </span>
                </div>
                {session.pid && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">{t('chat.settings.pid')}</span>
                    <span>{session.pid}</span>
                  </div>
                )}
                <div className="mt-2">
                  <span className="text-xs text-muted-foreground">{t('chat.settings.apiUrlHint')}</span>
                  <button
                    onClick={() => navigator.clipboard.writeText(apiUrl)}
                    className="block w-full text-left font-mono text-xs bg-background px-2 py-1.5 rounded border border-border mt-1 hover:bg-accent truncate"
                    title={t('chat.settings.apiUrlTooltip')}
                  >
                    {apiUrl}
                  </button>
                </div>
              </div>
            )
          })()}
        </div>

        <div className="border-t border-border" />

        {!isImageModel && compatibilityWarnings.length > 0 && (
          <div className="rounded border border-warning/30 bg-warning/10 p-3 text-xs text-warning">
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <div className="space-y-1.5">
                <div className="font-medium">Review saved chat settings</div>
                {compatibilityWarnings.map((warning) => (
                  <p key={warning} className="leading-snug">{warning}</p>
                ))}
              </div>
            </div>
          </div>
        )}

        {isImageModel && (
          <div>
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">{t('chat.settings.imageServer')}</h3>
            <p className="text-xs text-muted-foreground">
              {t('chat.settings.imageServerNote')}
            </p>
          </div>
        )}

        {!isImageModel && <>
        {/* Profiles */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('chat.settings.profiles')}</h3>
          <p className="text-[10px] text-muted-foreground/70 mb-2">
            {t('chat.settings.profilesInheritHint')}
          </p>
          <div className="space-y-2">
            {profiles.length > 0 && (
              <div className="space-y-1">
                {profiles.map(p => (
                  <div key={p.id} className="flex items-center gap-1 group">
                    <button
                      onClick={() => handleLoadProfile(p)}
                      className="flex-1 text-left text-xs px-2 py-1.5 rounded border border-border hover:bg-accent truncate"
                      title={t('chat.settings.loadProfileTitle', { name: p.name })}
                    >
                      {p.isDefault && <Star className="inline h-3 w-3 mr-1 text-yellow-500 fill-yellow-500" />}
                      {p.name}
                    </button>
                    <button
                      onClick={() => handleSetDefault(p)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:text-yellow-500"
                      title={p.isDefault ? t('chat.settings.removeDefaultTitle') : t('chat.settings.setDefaultTitle')}
                    >
                      <Star className={`h-3 w-3 ${p.isDefault ? 'fill-yellow-500 text-yellow-500' : ''}`} />
                    </button>
                    <button
                      onClick={() => handleDeleteProfile(p.id, p.name)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:text-destructive"
                      title={t('chat.settings.deleteProfileTitle')}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {showProfileSave ? (
              <div className="flex gap-1">
                <input
                  type="text"
                  value={profileName}
                  onChange={e => setProfileName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSaveProfile()}
                  placeholder={t('chat.settings.profileNamePlaceholder')}
                  className="flex-1 px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
                  autoFocus
                />
                <button onClick={handleSaveProfile} className="px-2 py-1 bg-primary text-primary-foreground rounded text-xs hover:bg-primary/90" disabled={!profileName.trim()}>
                  {t('chat.settings.save')}
                </button>
                <button onClick={() => { setShowProfileSave(false); setProfileName('') }} className="px-2 py-1 text-xs text-muted-foreground hover:text-foreground">
                  {t('chat.settings.cancel')}
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowProfileSave(true)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                <Save className="h-3 w-3" /> {t('chat.settings.saveAsProfile')}
              </button>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-1.5">
            {t('chat.settings.profilesFooter')}
          </p>
        </div>

        <div className="border-t border-border" />

        {/* Inference Settings */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">{t('chat.settings.inference')}</h3>
          <div className="space-y-4">
            <div className="border-b border-border pb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">{t('chat.settings.enableThinking')}</span>
              </div>
              {detectedFamily === 'deepseek-v4' ? (
                <div className="flex gap-1 bg-background rounded border border-border p-0.5">
                  <button
                    onClick={() => updateThinkingMode(undefined, undefined)}
                    className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                      overrides.enableThinking == null
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-accent text-muted-foreground'
                    }`}
                  >
                    {t('chat.settings.thinkingAuto')}
                  </button>
                  <button
                    onClick={() => updateThinkingMode(false, undefined)}
                    className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                      overrides.enableThinking === false
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-accent text-muted-foreground'
                    }`}
                  >
                    Instruct
                  </button>
                  <button
                    onClick={() => updateThinkingMode(true, undefined)}
                    className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                      overrides.enableThinking === true && (overrides.reasoningEffort !== 'max' || !dsv4MaxEnabled)
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-accent text-muted-foreground'
                    }`}
                  >
                    Reasoning
                  </button>
                  <button
                    disabled={!dsv4MaxEnabled}
                    title={undefined}
                    onClick={() => updateThinkingMode(true, 'max')}
                    className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                      overrides.enableThinking === true && overrides.reasoningEffort === 'max' && dsv4MaxEnabled
                        ? 'bg-primary text-primary-foreground'
                        : dsv4MaxEnabled ? 'hover:bg-accent text-muted-foreground' : 'text-muted-foreground opacity-50 cursor-not-allowed'
                    }`}
                  >
                    Max
                  </button>
                </div>
              ) : (
                <div className="flex gap-1 bg-background rounded border border-border p-0.5">
                  <button
                    disabled={!thinkingSupported}
                    onClick={() => updateThinkingMode(undefined, undefined)}
                    className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                      displayedEnableThinking == null
                        ? 'bg-primary text-primary-foreground'
                        : thinkingSupported ? 'hover:bg-accent text-muted-foreground' : 'text-muted-foreground opacity-50 cursor-not-allowed'
                    }${thinkingDisabledClass}`}
                  >
                    {t('chat.settings.thinkingAuto')}
                  </button>
                  <button
                    disabled={!thinkingSupported}
                    onClick={() => updateThinkingMode(true, overrides.reasoningEffort)}
                    className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                      displayedEnableThinking === true
                        ? 'bg-primary text-primary-foreground'
                        : thinkingSupported ? 'hover:bg-accent text-muted-foreground' : 'text-muted-foreground opacity-50 cursor-not-allowed'
                    }${thinkingDisabledClass}`}
                  >
                    {t('chat.settings.thinkingOn')}
                  </button>
                  {thinkingOffSupported && (
                    <button
                      disabled={!thinkingSupported}
                      onClick={() => updateThinkingMode(false, undefined)}
                      className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                        displayedEnableThinking === false
                          ? 'bg-primary text-primary-foreground'
                          : thinkingSupported ? 'hover:bg-accent text-muted-foreground' : 'text-muted-foreground opacity-50 cursor-not-allowed'
                      }${thinkingDisabledClass}`}
                    >
                      {t('chat.settings.thinkingOff')}
                    </button>
                  )}
                </div>
              )}
              <p className="text-xs text-muted-foreground mt-1.5">
                {t(thinkingOffSupported
                  ? 'chat.settings.thinkingHelp'
                  : 'chat.settings.thinkingNativeOnlyHelp')}
              </p>
              {detectedFamily !== 'deepseek-v4' && overrides.enableThinking !== false && showReasoningEffort && (
                <div className="mt-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs text-muted-foreground">{t('chat.settings.reasoningEffort')}</span>
                  </div>
                  <div className="flex gap-1 bg-background rounded border border-border p-0.5">
                    <button
                      onClick={() => update('reasoningEffort', undefined)}
                      className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                        overrides.reasoningEffort == null
                          ? 'bg-primary text-primary-foreground'
                          : 'hover:bg-accent text-muted-foreground'
                      }`}
                    >
                      {t('chat.settings.thinkingAuto')}
                    </button>
                    {showLowEffort && (
                      <>
                        <button
                          onClick={() => update('reasoningEffort', 'low')}
                          className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                            overrides.reasoningEffort === 'low'
                              ? 'bg-primary text-primary-foreground'
                              : 'hover:bg-accent text-muted-foreground'
                          }`}
                        >
                          {t('chat.settings.effortLow')}
                        </button>
                        {showMediumEffort && (
                          <button
                            onClick={() => update('reasoningEffort', 'medium')}
                            className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                              overrides.reasoningEffort === 'medium'
                                ? 'bg-primary text-primary-foreground'
                                : 'hover:bg-accent text-muted-foreground'
                            }`}
                          >
                            {t('chat.settings.effortMedium')}
                          </button>
                        )}
                      </>
                    )}
                    {showHighEffort && (
                      <button
                        onClick={() => update('reasoningEffort', 'high')}
                        className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                          overrides.reasoningEffort === 'high'
                            ? 'bg-primary text-primary-foreground'
                            : 'hover:bg-accent text-muted-foreground'
                        }`}
                      >
                        {t('chat.settings.effortHigh')}
                      </button>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {effectiveReasoningParser === 'mistral'
                      ? t('chat.settings.effortHelpMistral')
                      : t('chat.settings.effortHelpGeneric')}
                  </p>
                </div>
              )}
            </div>
            <SliderField
              label={t('chat.settings.temperature')}
              value={overrides.temperature ?? modelDefaults.temperature ?? 0}
              onChange={v => update('temperature', v)}
              min={0} max={2} step={0.05}
              help={t('chat.settings.temperatureHelp')}
            />
            <SliderField
              label={t('chat.settings.topP')}
              value={overrides.topP ?? modelDefaults.topP ?? 1}
              onChange={v => update('topP', v)}
              min={0} max={1} step={0.05}
              help={t('chat.settings.topPHelp')}
            />
            <NumberField
              label={t('chat.settings.maxTokens')}
              value={overrides.maxTokens}
              onChange={v => update('maxTokens', v)}
              placeholder={modelDefaults.maxTokens ? `${modelDefaults.maxTokens} (model default)` : t('chat.settings.maxTokensPlaceholder')}
              help={t('chat.settings.maxTokensHelp')}
            />
            {/* Show ONLY for families whose engine honors a top-level
                max_thinking_tokens reasoning-phase cap (registry supportsThinkingBudget)
                or whose chat TEMPLATE declares a budget marker (thinkingBudgetSupported).
                Families with a reasoning parser but no engine-side thinking-budget
                behavior (deepseek-v4, step-3.7-flash, etc.) ignore the field, so showing
                it would be untruthful. The supportsThinkingBudget flag covers Qwen3.5/3.6
                whose template has <think> but no budget marker. */}
            {(supportsThinkingBudget === true || thinkingBudgetSupported === true) && displayedEnableThinking !== false && (
              <NumberField
                label={t('chat.settings.maxThinkingTokens')}
                value={overrides.maxThinkingTokens}
                onChange={v => update('maxThinkingTokens', v)}
                placeholder={modelDefaults.maxThinkingTokens ? `${modelDefaults.maxThinkingTokens} (model default)` : t('chat.settings.maxThinkingTokensPlaceholder')}
                help={t('chat.settings.maxThinkingTokensHelp')}
              />
            )}
            <SliderField
              label={t('chat.settings.topK')}
              value={displayedTopK}
              // Zero is an explicit and useful override. Clearing it here made
              // a bundle top-k default immediately snap back on screen and in
              // the next request, so users could not actually disable top-k.
              onChange={v => update('topK', v)}
              min={0} max={topKSliderMax} step={1}
              help={t('chat.settings.topKHelp')}
              format={formatTopK}
            />
            <SliderField
              label={t('chat.settings.minP')}
              value={overrides.minP ?? modelDefaults.minP ?? 0}
              // Unlike top-k, zero must remain an explicit override: omitting
              // min_p lets the server restore a non-zero bundle default.
              onChange={v => update('minP', v)}
              min={0} max={1} step={0.01}
              help={t('chat.settings.minPHelp')}
            />
            <SliderField
              label={t('chat.settings.repetitionPenalty')}
              value={overrides.repeatPenalty ?? modelDefaults.repeatPenalty ?? 1.0}
              // 1.0 is an explicit neutral override, distinct from inheriting a
              // bundle-declared repetition penalty. Reset clears inheritance.
              onChange={v => update('repeatPenalty', v)}
              min={1.0} max={2.0} step={0.05}
              help={t('chat.settings.repetitionPenaltyHelp')}
            />
          </div>
        </div>

        <div className="border-t border-border" />

        {/* System Prompt with Templates */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('chat.settings.systemPrompt')}</h3>
          <PromptTemplateSelector
            value={overrides.systemPrompt ?? ''}
            onChange={(prompt) => update('systemPrompt', prompt || undefined)}
          />
          <p className="text-xs text-muted-foreground mt-1">{t('chat.settings.systemPromptHelp')}</p>
        </div>

        {/* Stop Sequences */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('chat.settings.stopSequences')}</h3>
          <input
            type="text"
            value={overrides.stopSequences ?? ''}
            onChange={e => update('stopSequences', e.target.value || undefined)}
            placeholder={t('chat.settings.stopSequencesPlaceholder')}
            className="w-full px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <p className="text-xs text-muted-foreground mt-1">{t('chat.settings.stopSequencesHelp')}</p>
        </div>

        <div className="border-t border-border" />

        {/* Wire Format */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('chat.settings.wireFormat')}</h3>
          <select
            value={effectiveWireApi}
            onChange={e => update('wireApi', e.target.value as 'completions' | 'responses')}
            className="w-full px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          >
            <option value="completions">{t('chat.settings.wireCompletions')}</option>
            <option value="responses">{t('chat.settings.wireResponses')}</option>
          </select>
          <p className="text-xs text-muted-foreground mt-1">{t('chat.settings.wireHelp')}</p>
        </div>

        <div className="border-t border-border" />

        {/* Agentic Settings */}
        <div>
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">{t('chat.settings.agentic')}</h3>
          <div className="space-y-4">
            <NumberField
              label={t('chat.settings.maxToolIterations')}
              value={overrides.maxToolIterations}
              onChange={v => update('maxToolIterations', v)}
              placeholder={t('chat.settings.maxToolIterationsPlaceholder')}
              help={t('chat.settings.maxToolIterationsHelp')}
            />

            <div className="border-t border-border pt-4">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={overrides.builtinToolsEnabled ?? false}
                  onChange={e => update('builtinToolsEnabled', e.target.checked || undefined)}
                  className="w-4 h-4 accent-primary"
                />
                <span className="font-medium">{t('chat.settings.enableBuiltinTools')}</span>
              </label>
              <p className="text-xs text-muted-foreground mt-1 ml-6">
                {t('chat.settings.enableBuiltinToolsHelp')}
              </p>

              {overrides.builtinToolsEnabled && (
                <div className="mt-3 ml-6 space-y-2">
                  <label className="text-sm">{t('chat.settings.workingDirectory')}</label>
                  {!overrides.workingDirectory && (
                    <div className="px-2 py-1.5 rounded text-[11px] bg-warning/10 border border-warning/30 text-warning leading-tight">
                      {t('chat.settings.workingDirectoryRequired')}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={overrides.workingDirectory ?? ''}
                      onChange={e => update('workingDirectory', e.target.value)}
                      spellCheck={false}
                      placeholder={t('chat.settings.workingDirectoryPlaceholder')}
                      className="flex-1 px-3 py-1.5 bg-background border border-input rounded text-sm font-mono truncate focus:outline-none"
                    />
                    <button
                      onClick={async () => {
                        const result = await window.api.chat.openDirectory()
                        if (result && !result.canceled && result.filePaths?.[0]) {
                          update('workingDirectory', result.filePaths[0])
                        }
                      }}
                      className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent whitespace-nowrap"
                    >
                      {t('chat.settings.browse')}
                    </button>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {t('chat.settings.workingDirectoryHelp')}
                  </p>

                  <div className="mt-3 pt-3 border-t border-border space-y-2">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{t('chat.settings.toolCategories')}</p>
                    <ToolToggle
                      label={t('chat.settings.toolFileIO')}
                      checked={overrides.fileToolsEnabled !== false}
                      onChange={v => update('fileToolsEnabled', v)}
                      help={t('chat.settings.toolFileIOHelp')}
                    />
                    <ToolToggle
                      label={t('chat.settings.toolSearch')}
                      checked={overrides.searchToolsEnabled !== false}
                      onChange={v => update('searchToolsEnabled', v)}
                      help={t('chat.settings.toolSearchHelp')}
                    />
                    <ToolToggle
                      label={t('chat.settings.toolShell')}
                      checked={overrides.shellEnabled !== false}
                      onChange={v => update('shellEnabled', v)}
                      help={t('chat.settings.toolShellHelp')}
                    />
                    <ToolToggle
                      label={t('chat.settings.toolWebSearch')}
                      checked={overrides.webSearchEnabled !== false}
                      onChange={v => update('webSearchEnabled', v)}
                      help={t('chat.settings.toolWebSearchHelp')}
                    />
                    <BraveSearchToggle
                      checked={overrides.braveSearchEnabled === true}
                      onChange={v => update('braveSearchEnabled', v)}
                    />
                    <ToolToggle
                      label={t('chat.settings.toolUrlFetch')}
                      checked={overrides.fetchUrlEnabled !== false}
                      onChange={v => update('fetchUrlEnabled', v)}
                      help={t('chat.settings.toolUrlFetchHelp')}
                    />
                    <ToolToggle
                      label={t('chat.settings.toolGit')}
                      checked={overrides.gitEnabled !== false}
                      onChange={v => update('gitEnabled', v)}
                      help={t('chat.settings.toolGitHelp')}
                    />
                    <ToolToggle
                      label={t('chat.settings.toolUtilities')}
                      checked={overrides.utilityToolsEnabled !== false}
                      onChange={v => update('utilityToolsEnabled', v)}
                      help={t('chat.settings.toolUtilitiesHelp')}
                    />
                  </div>

                  <div className="mt-3 pt-3 border-t border-border">
                    <label className="text-sm font-medium">{t('chat.settings.toolResultLimit')}</label>
                    <p className="text-xs text-muted-foreground mt-0.5 mb-2">
                      {t('chat.settings.toolResultLimitHelp')}
                    </p>
                    <div className="flex items-center gap-2">
                      <input
                        type="range"
                        min={500}
                        max={50000}
                        step={500}
                        value={overrides.toolResultMaxChars ?? 50000}
                        onChange={e => {
                          const v = Number(e.target.value)
                          update('toolResultMaxChars', v >= 50000 ? undefined : v)
                        }}
                        className="flex-1 accent-primary"
                      />
                      <span className="text-xs font-mono w-16 text-right tabular-nums text-muted-foreground">
                        {overrides.toolResultMaxChars ? `${(overrides.toolResultMaxChars / 1000).toFixed(1)}k` : '50k'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="border-t border-border pt-4">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={overrides.hideToolStatus === true}
                  onChange={e => update('hideToolStatus', e.target.checked)}
                  className="w-4 h-4 accent-primary"
                />
                <span className="font-medium">{t('chat.settings.hideToolStatus')}</span>
              </label>
              <p className="text-xs text-muted-foreground mt-1 ml-6">
                {t('chat.settings.hideToolStatusHelp')}
              </p>
            </div>
          </div>
        </div>
        </>}
      </div>

      {/* Footer Actions */}
      <div className="flex items-center gap-2 px-4 py-3 border-t border-border flex-shrink-0">
        <button
          onClick={handleSave}
          disabled={!dirty || saving}
          className="flex-1 px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
        >
          {saving ? t('chat.settings.saving') : t('chat.settings.save')}
        </button>
        <button
          onClick={handleReset}
          className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent"
        >
          {t('chat.settings.reset')}
        </button>
      </div>
    </div>
  )
}

// ─── Helper Components ────────────────────────────────────────────────────────

function SliderField({ label, value, onChange, min, max, step, help, format }: {
  label: string; value: number; onChange: (v: number) => void
  min: number; max: number; step: number; help: string
  format?: (v: number) => string
}) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="text-muted-foreground font-mono text-xs">{format ? format(value) : value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        min={min} max={max} step={step}
        className="w-full h-1.5 accent-primary"
      />
      <p className="text-xs text-muted-foreground mt-0.5">{help}</p>
    </div>
  )
}

function NumberField({ label, value, onChange, placeholder, help }: {
  label: string; value: number | undefined; onChange: (v: number | undefined) => void
  placeholder: string; help: string
}) {
  return (
    <div>
      <label className="text-sm">{label}</label>
      <input
        type="number"
        value={value ?? ''}
        onChange={e => onChange(e.target.value ? parseInt(e.target.value) : undefined)}
        placeholder={placeholder}
        className="w-full mt-1 px-3 py-1.5 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
      />
      <p className="text-xs text-muted-foreground mt-0.5">{help}</p>
    </div>
  )
}

function BraveSearchToggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  const { t } = useTranslation()
  const [key, setKey] = useState('')
  const [hasKey, setHasKey] = useState(false)
  const checkedRef = useRef(checked)
  const onChangeRef = useRef(onChange)
  checkedRef.current = checked
  onChangeRef.current = onChange

  useEffect(() => {
    window.api.settings.has('braveApiKey').then((exists: boolean) => {
      if (exists) setHasKey(true)
      else if (checkedRef.current) onChangeRef.current(false) // correct stale enabled state when key was deleted
    })
  }, [])

  const saveKey = async (val: string) => {
    const trimmed = val.trim()
    setKey(trimmed)
    if (trimmed) {
      await window.api.settings.set('braveApiKey', trimmed)
      setHasKey(true)
    } else {
      await window.api.settings.delete('braveApiKey')
      setHasKey(false)
      onChange(false) // auto-disable when key removed
    }
  }

  return (
    <div>
      <label className="flex items-start gap-2 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={checked}
          onChange={e => {
            if (e.target.checked && !hasKey) return // can't enable without key
            onChange(e.target.checked)
          }}
          disabled={!hasKey}
          className="w-3.5 h-3.5 accent-primary mt-0.5"
        />
        <span>
          <span>{t('chat.settings.braveSearch')}</span>
          <span className="block text-xs text-muted-foreground">
            {t(hasKey ? 'chat.settings.braveWithKey' : 'chat.settings.braveNoKey')}
          </span>
        </span>
      </label>
      <div className="ml-6 mt-1.5">
        <input
          type="password"
          value={key}
          onChange={e => setKey(e.target.value)}
          onBlur={e => saveKey(e.target.value)}
          placeholder={t('chat.settings.braveKeyPlaceholder')}
          className="w-full px-2 py-1 bg-background border border-input rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-ring"
        />
        <a
          href="https://brave.com/search/api/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[10px] text-primary hover:underline mt-0.5 inline-block"
        >
          {t('chat.settings.braveGetKey')}
        </a>
      </div>
    </div>
  )
}

function ToolToggle({ label, checked, onChange, help }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; help: string
}) {
  return (
    <label className="flex items-start gap-2 text-sm cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        className="w-3.5 h-3.5 accent-primary mt-0.5"
      />
      <span>
        <span>{label}</span>
        <span className="block text-xs text-muted-foreground">{help}</span>
      </span>
    </label>
  )
}

// ─── Prompt Template Selector (backed by SQLite prompt_templates table) ──────

interface PromptTemplate {
  id: string
  name: string
  content: string
  category: string
  isBuiltin: boolean
}

function PromptTemplateSelector({ value, onChange }: { value: string; onChange: (prompt: string) => void }) {
  const { t } = useTranslation()
  const [templates, setTemplates] = useState<PromptTemplate[]>([])
  const [saveName, setSaveName] = useState('')
  const [showSave, setShowSave] = useState(false)

  useEffect(() => {
    window.api.templates.list().then(setTemplates).catch((err) => console.error('Failed to load templates:', err))
  }, [])

  const builtins = templates.filter(tpl => tpl.isBuiltin)
  const customs = templates.filter(tpl => !tpl.isBuiltin)

  const handleSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value
    if (id === '__none__') { onChange(''); return }
    const tmpl = templates.find(tpl => tpl.id === id)
    if (tmpl) onChange(tmpl.content)
  }

  const handleSave = async () => {
    const name = saveName.trim()
    if (!name || !value.trim()) return
    const id = `custom-${Date.now()}`
    await window.api.templates.save({ id, name, content: value, category: 'custom' })
    setTemplates(await window.api.templates.list())
    setSaveName('')
    setShowSave(false)
  }

  const handleDeleteCustom = async (id: string) => {
    await window.api.templates.delete(id)
    setTemplates(await window.api.templates.list())
  }

  const currentMatch = templates.find(tpl => tpl.content === value)

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <select
          value={currentMatch?.id ?? '__custom__'}
          onChange={handleSelect}
          className="flex-1 text-xs px-2 py-1.5 bg-background border border-input rounded focus:outline-none focus:ring-1 focus:ring-ring"
        >
          <option value="__none__">{t('chat.settings.templateNone')}</option>
          {builtins.length > 0 && (
            <optgroup label={t('chat.settings.templateBuiltIn')}>
              {builtins.map(tpl => (
                <option key={tpl.id} value={tpl.id}>{tpl.name}</option>
              ))}
            </optgroup>
          )}
          {customs.length > 0 && (
            <optgroup label={t('chat.settings.templateCustomGroup')}>
              {customs.map(tpl => (
                <option key={tpl.id} value={tpl.id}>{tpl.name}</option>
              ))}
            </optgroup>
          )}
          {!currentMatch && value && <option value="__custom__">{t('chat.settings.templateCustomOption')}</option>}
        </select>
        <button
          onClick={() => setShowSave(!showSave)}
          className="text-xs px-2 py-1 border border-border rounded hover:bg-accent"
          title={t('chat.settings.saveTemplateTitle')}
        >
          {t('chat.settings.save')}
        </button>
      </div>

      {showSave && (
        <div className="flex gap-2">
          <input
            type="text"
            value={saveName}
            onChange={e => setSaveName(e.target.value)}
            placeholder={t('chat.settings.templateNamePlaceholder')}
            className="flex-1 text-xs px-2 py-1 bg-background border border-input rounded focus:outline-none focus:ring-1 focus:ring-ring"
            onKeyDown={e => e.key === 'Enter' && handleSave()}
          />
          <button onClick={handleSave} className="text-xs px-2 py-1 bg-primary text-primary-foreground rounded hover:bg-primary/90">
            {t('chat.settings.save')}
          </button>
        </div>
      )}

      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={t('chat.settings.systemPromptPlaceholder')}
        className="w-full h-24 resize-none px-3 py-2 bg-background border border-input rounded text-sm focus:outline-none focus:ring-1 focus:ring-ring"
      />

      {customs.length > 0 && (
        <div className="flex gap-1 flex-wrap">
          {customs.map(tpl => (
            <span key={tpl.id} className="text-xs bg-muted px-1.5 py-0.5 rounded flex items-center gap-1">
              {tpl.name}
              <button onClick={() => handleDeleteCustom(tpl.id)} className="text-destructive hover:text-destructive/80">x</button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

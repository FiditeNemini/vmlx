import { useState, useEffect, useCallback } from 'react'
import { CASUAL_CONFIG, EXPERT_CONFIG, SessionConfig } from '../sessions/SessionConfigForm'
import { useTranslation } from '../../i18n'

export type InferenceMode = 'casual' | 'expert'

export function useInferenceMode() {
  const [mode, setModeState] = useState<InferenceMode>('expert')

  useEffect(() => {
    window.api.settings.get('inference_mode').then((val: string | null) => {
      if (val === 'casual' || val === 'expert') setModeState(val)
    })
  }, [])

  const setMode = useCallback((m: InferenceMode) => {
    setModeState(m)
    window.api.settings.set('inference_mode', m)
  }, [])

  const defaultConfig: SessionConfig = mode === 'casual' ? CASUAL_CONFIG : EXPERT_CONFIG

  return { mode, setMode, defaultConfig }
}

export function InferenceModeToggle({ mode, onToggle }: { mode: InferenceMode; onToggle: (m: InferenceMode) => void }) {
  const { t } = useTranslation()
  const isCasual = mode === 'casual'
  return (
    <button
      onClick={() => onToggle(isCasual ? 'expert' : 'casual')}
      className={`w-full px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 border ${
        isCasual
          ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/25'
          : 'bg-blue-500/15 text-blue-400 border-blue-500/30 hover:bg-blue-500/25'
      }`}
      title={isCasual
        ? t('layout.inferenceMode.casualTitle')
        : t('layout.inferenceMode.expertTitle')}
    >
      {isCasual ? t('layout.inferenceMode.casual') : t('layout.inferenceMode.expert')}
    </button>
  )
}

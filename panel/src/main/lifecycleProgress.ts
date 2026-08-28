/**
 * The panel half of the engine-owned lifecycle-progress contract.
 *
 * The engine reports phase / completed / total / model_loaded / ready /
 * generation (vmlx_engine/load_progress.py) — mirrored as LOADPROGRESS
 * stdout lines during cold start (uvicorn serves nothing until lifespan
 * completes) and embedded in /health as `load_progress` (how an external
 * API-triggered JIT wake becomes visible). The panel maps phases to a
 * display percentage but never invents one, and RSS/Metal residency is a
 * separate diagnostic — never the percentage oracle.
 */

export interface LifecycleSnapshot {
  phase: string
  completed: number
  total: number
  model_loaded: boolean
  ready: boolean
  generation: number
}

export function parseLifecycleSnapshot(json: string): LifecycleSnapshot | null {
  try {
    const parsed = JSON.parse(json)
    if (!parsed || typeof parsed !== 'object') return null
    if (typeof parsed.phase !== 'string' || typeof parsed.generation !== 'number') return null
    return {
      phase: parsed.phase,
      completed: Number(parsed.completed) || 0,
      total: Number(parsed.total) || 0,
      model_loaded: parsed.model_loaded === true,
      ready: parsed.ready === true,
      generation: parsed.generation,
    }
  } catch {
    return null
  }
}

export type LifecycleDisplay =
  | { indeterminate: true }
  | { indeterminate: false; percent: number }

/**
 * Snapshot → display. Determinate percentages exist ONLY for measured
 * completed/total units (weight shards), capped at 99 so that 100 is
 * reserved exclusively for the engine's authoritative ready=true
 * (model_loaded + engine init + acceleration restoration + the readiness
 * barrier). Every phase without a real denominator renders as an
 * indeterminate animated bar — no invented numbers.
 */
export function lifecycleDisplay(snap: LifecycleSnapshot): LifecycleDisplay {
  if (snap.ready) return { indeterminate: false, percent: 100 }
  if (snap.phase === 'loading_weights' && snap.total > 0) {
    return {
      indeterminate: false,
      percent: Math.min(99, Math.round(100 * Math.min(1, snap.completed / Math.max(1, snap.total)))),
    }
  }
  return { indeterminate: true }
}

export function lifecyclePhaseLabel(snap: LifecycleSnapshot): {
  label: string
  labelKey: string
  labelParams?: Record<string, string | number>
} {
  if (snap.ready) {
    return { label: 'Model ready', labelKey: 'main.loadProgress.modelReady' }
  }
  switch (snap.phase) {
    case 'starting':
      return { label: 'Starting engine...', labelKey: 'main.loadProgress.initializing' }
    case 'loading_weights':
      if (snap.total > 0) {
        return {
          label: `Loading weights — shard ${snap.completed} of ${snap.total}`,
          labelKey: 'main.loadProgress.loadingWeightShards',
          labelParams: { completed: snap.completed, total: snap.total },
        }
      }
      return { label: 'Loading weights...', labelKey: 'main.loadProgress.loadingWeights' }
    case 'initializing_engine':
      return { label: 'Initializing engine...', labelKey: 'main.loadProgress.initializingEngine' }
    case 'restoring_acceleration':
      return { label: 'Restoring acceleration...', labelKey: 'main.loadProgress.restoringAcceleration' }
    default:
      return { label: 'Preparing...', labelKey: 'main.loadProgress.preparing' }
  }
}

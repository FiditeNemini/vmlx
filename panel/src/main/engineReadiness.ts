/**
 * Local-engine readiness from a /health response body.
 *
 * /health answers HTTP 200 in EVERY state — no_model while loading in
 * lifespan, standby_soft/standby_deep while sleeping, healthy when serving.
 * Treating any 200 as "ready to infer" let messages race a model that was
 * still loading or waking. Inference readiness requires the body to say so.
 */
export function localEngineReadyFromHealthBody(body: unknown): boolean {
  if (!body || typeof body !== 'object') return false
  const health = body as Record<string, unknown>
  if (health.status !== 'healthy') return false
  if (health.model_loaded !== true) return false
  if (health.wake_in_progress === true) return false
  // Engines speaking the lifecycle contract must also assert authoritative
  // readiness; older engines without the field fall back to status alone.
  const progress = health.load_progress
  if (progress && typeof progress === 'object') {
    if ((progress as Record<string, unknown>).ready !== true) return false
  }
  return true
}

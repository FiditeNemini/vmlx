/**
 * Why the temperature box must explain itself when a model has native MTP.
 *
 * 2026-08-17, Eric: "IF MTP IS TURNED ON IT SHOULD SHOW IN CHAT SETTINGS THE
 * TEMP SET TO 0 AND TELL USERS WHY" / "IF SET TO AUTO AND MODEL HAS MTP IT
 * PROPERLY LETS THEM KNOW IN THE CHAT SETTINGS IN THE TEMP BOX AREA AND THAT IT
 * IS DUE TO MTP BEING ON".
 *
 * Sampled native MTP now uses rejection-sampling acceptance. Auto therefore
 * preserves the bundle's sampling defaults; Deterministic is the explicit
 * opt-in that replaces omitted sampling values with greedy defaults.
 *
 * Three states worth surfacing, all keyed on the model ACTUALLY having MTP:
 *  - `pinned`   deterministic mode: sampling was replaced with greedy, and the
 *               0 in the box is a consequence of MTP, not a user choice.
 *  - `active`   auto mode at any temperature, or an explicit sampled override:
 *               MTP uses rejection-sampling acceptance.
 *  - `inactive` retained for callers that may later carry an explicit runtime
 *               gate result; the current resolver does not infer it from temp.
 */

export type MtpTemperatureNoticeKind = 'pinned' | 'active' | 'inactive'

export interface MtpTemperatureNotice {
  kind: MtpTemperatureNoticeKind
  /** The temperature that caused an `inactive` verdict. */
  temperature?: number
}

export interface MtpTemperatureNoticeInput {
  /** True only when the bundle really carries native MTP heads. */
  nativeMtpSupported: boolean
  /** Session `nativeMtpMode`: 'auto' | 'deterministic' | 'off'. */
  mode: string | undefined
  /** Effective temperature shown in the box. */
  temperature: number | undefined
}

/**
 * Read `nativeMtpMode` out of a session config (stored as a JSON string).
 * Anything unreadable or absent means the app default, which is `auto`.
 */
export function parseSessionNativeMtpMode(
  sessionConfig: string | Record<string, unknown> | undefined,
): string {
  if (!sessionConfig) return 'auto'
  let parsed: unknown = sessionConfig
  if (typeof sessionConfig === 'string') {
    try {
      parsed = JSON.parse(sessionConfig)
    } catch {
      return 'auto'
    }
  }
  if (!parsed || typeof parsed !== 'object') return 'auto'
  const mode = (parsed as Record<string, unknown>).nativeMtpMode
  return typeof mode === 'string' && mode ? mode : 'auto'
}

export function resolveMtpTemperatureNotice(
  input: MtpTemperatureNoticeInput,
): MtpTemperatureNotice | null {
  // No MTP heads -> temperature has nothing to do with MTP. Say nothing.
  if (!input.nativeMtpSupported) return null

  const mode = typeof input.mode === 'string' && input.mode ? input.mode : 'auto'
  // Explicitly disabled by the user: their choice, not a surprise.
  if (mode === 'off') return null

  const temperature = input.temperature
  if (temperature == null) return null
  if (mode === 'deterministic' && temperature === 0) return { kind: 'pinned' }
  return { kind: 'active', temperature }
}

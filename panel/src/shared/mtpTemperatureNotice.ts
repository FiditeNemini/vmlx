/**
 * Why the temperature box must explain itself when a model has native MTP.
 *
 * 2026-08-17, Eric: "IF MTP IS TURNED ON IT SHOULD SHOW IN CHAT SETTINGS THE
 * TEMP SET TO 0 AND TELL USERS WHY" / "IF SET TO AUTO AND MODEL HAS MTP IT
 * PROPERLY LETS THEM KNOW IN THE CHAT SETTINGS IN THE TEMP BOX AREA AND THAT IT
 * IS DUE TO MTP BEING ON".
 *
 * The trap this closes: native MTP only runs on GREEDY requests. In `auto` mode
 * the session preserves the bundle's own sampling defaults, and several MTP
 * bundles (dots3-note, the Qwen MTP builds) ship `temperature: 1.0`. So the
 * headline feature silently never engages, the user sees ordinary decode
 * speed, and nothing anywhere says why. A capability that quietly declines is
 * indistinguishable from one that is broken.
 *
 * Three states worth surfacing, all keyed on the model ACTUALLY having MTP:
 *  - `pinned`   deterministic mode: sampling was replaced with greedy, and the
 *               0 in the box is a consequence of MTP, not a user choice.
 *  - `active`   auto mode at temperature 0: MTP will run.
 *  - `inactive` auto mode above 0: MTP will NOT run, and this is the case that
 *               previously failed silently.
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

  // With MTP on (auto or deterministic) the session pins greedy sampling, so a
  // 0 in the box is a CONSEQUENCE of MTP and must say so. A non-zero value can
  // now only come from the user overriding it by hand — in which case MTP will
  // not run, and that is the case worth warning about.
  const temperature = input.temperature
  if (temperature == null) return null
  if (temperature > 0) return { kind: 'inactive', temperature }
  return { kind: 'pinned' }
}

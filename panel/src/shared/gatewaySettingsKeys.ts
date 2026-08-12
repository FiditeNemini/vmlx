/**
 * ONE spelling of the gateway settings keys read out of the settings table.
 *
 * `gateway_single_model_mode` was hand-written as a bare string literal at SIX
 * call sites in `src/main/` (sessions.ts ×5, tray.ts ×1) alongside a named
 * constant in api-gateway.ts. All seven agreed when this was extracted, so this
 * removes the opportunity rather than fixing a live divergence.
 *
 * It is worth removing because of HOW this key fails. Every reader compares
 * `db.getSetting(key) === 'true'`, and `getSetting` returns undefined for an
 * unknown key — so a typo in one literal does not throw, it silently evaluates
 * to `false` and that one site quietly stops enforcing single-model mode. The
 * user would see other models left resident with no error anywhere, and no test
 * would fail, because each site is covered by its own suite.
 */

/** Whether the gateway keeps only one model resident at a time. */
export const GATEWAY_SINGLE_MODEL_MODE_KEY = 'gateway_single_model_mode'

/**
 * True only for the exact stored value that means enabled.
 *
 * Centralised for the same reason as the key: the setting is stored as the
 * STRING `'true'`, and a reader that did a truthiness check instead would treat
 * the stored `'false'` as enabled.
 */
export function isGatewaySettingEnabled(value: string | undefined | null): boolean {
  return value === 'true'
}

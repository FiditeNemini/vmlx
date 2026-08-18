import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import {
  parseSessionNativeMtpMode,
  resolveMtpTemperatureNotice,
} from '../src/shared/mtpTemperatureNotice'

/**
 * 2026-08-17. Native MTP only runs on GREEDY requests. In `auto` mode a session
 * preserves the bundle's own sampling, and MTP bundles (dots3-note, the Qwen
 * MTP builds) ship temperature 1.0 — so the headline speed feature silently
 * never engaged and NOTHING in the UI said why. Observed live: dots3-note
 * decoding at 21.9 t/s with Native MTP Mode showing "Auto".
 *
 * Eric: "IF MTP IS TURNED ON IT SHOULD SHOW IN CHAT SETTINGS THE TEMP SET TO 0
 * AND TELL USERS WHY" / "IF SET TO AUTO AND MODEL HAS MTP IT PROPERLY LETS THEM
 * KNOW IN THE CHAT SETTINGS IN THE TEMP BOX AREA".
 */
describe('MTP temperature disclosure', () => {
  it('says nothing when the bundle has no MTP heads', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: false, mode: 'deterministic', temperature: 0 }),
    ).toBeNull()
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: false, mode: 'auto', temperature: 1 }),
    ).toBeNull()
  })

  it('explains the pinned 0 in deterministic mode', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'deterministic', temperature: 0 }),
    ).toEqual({ kind: 'pinned' })
  })

  it('THE REGRESSION: a hand-raised temperature must warn MTP is not running', () => {
    // Auto now pins greedy for MTP bundles, so a non-zero value can only come
    // from a manual override -- which silently disables MTP.
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', temperature: 1 }),
    ).toEqual({ kind: 'inactive', temperature: 1 })
    // even a small non-zero temperature disables MTP
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', temperature: 0.01 }),
    ).toEqual({ kind: 'inactive', temperature: 0.01 })
  })

  it('explains the pinned 0 on auto too, not just deterministic', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', temperature: 0 }),
    ).toEqual({ kind: 'pinned' })
  })

  it('stays silent when the user explicitly turned MTP off', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'off', temperature: 1 }),
    ).toBeNull()
  })

  it('treats a missing/!unreadable mode as auto, never as deterministic', () => {
    expect(parseSessionNativeMtpMode(undefined)).toBe('auto')
    expect(parseSessionNativeMtpMode('not json')).toBe('auto')
    expect(parseSessionNativeMtpMode('{}')).toBe('auto')
    expect(parseSessionNativeMtpMode(JSON.stringify({ nativeMtpMode: 'deterministic' }))).toBe('deterministic')
    expect(parseSessionNativeMtpMode({ nativeMtpMode: 'off' })).toBe('off')
  })

  it('is actually rendered in the temperature area of Chat Settings', () => {
    const source = readFileSync(
      resolve(__dirname, '../src/renderer/src/components/chat/ChatSettings.tsx'),
      'utf8',
    )
    expect(source).toContain('resolveMtpTemperatureNotice(')
    expect(source).toContain('data-testid="mtp-temperature-notice"')
    // all three states have copy wired
    expect(source).toContain('chat.settings.mtpTempPinned')
        expect(source).toContain('chat.settings.mtpTempInactive')
    // it must sit with the temperature control, not in some unrelated section
    const tempAt = source.indexOf("t('chat.settings.temperature')")
    const noticeAt = source.indexOf('data-testid="mtp-temperature-notice"')
    const topPAt = source.indexOf("t('chat.settings.topP')")
    expect(tempAt).toBeGreaterThan(-1)
    expect(noticeAt).toBeGreaterThan(tempAt)
    expect(noticeAt).toBeLessThan(topPAt)
  })

  it('ships user-facing copy that names MTP as the reason', () => {
    const en = JSON.parse(
      readFileSync(resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'), 'utf8'),
    )
    const settings = en.chat.settings
    expect(settings.mtpTempPinned).toMatch(/MTP/)
    expect(settings.mtpTempPinned).toMatch(/greedy/i)
    // the warning must tell the user what to DO about it
    expect(settings.mtpTempInactive).toMatch(/MTP/)
    expect(settings.mtpTempInactive).toMatch(/temperature to 0|Deterministic/i)
    expect(settings.mtpTempInactive).toContain('{{temperature}}')
  })
})

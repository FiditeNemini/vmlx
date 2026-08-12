import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

/**
 * The Reasoning Effort help text must not name an option this control does not
 * render.
 *
 * Caught live in the dev app on DSV4-Flash: the buttons are
 * `Default (Low) / Low / High / Max`, while the help below them read
 * "Auto lets the model decide". "Auto" is a real option — but of the SEPARATE
 * Enable Thinking control (Auto/On/Off) — so the copy sent the user looking for
 * a button that is not there.
 */
describe('reasoning effort help copy', () => {
  const locales = ['en', 'es', 'ja', 'ko', 'zh'] as const

  it('never advertises an Auto effort level', () => {
    for (const locale of locales) {
      const catalog = JSON.parse(
        readFileSync(
          resolve(__dirname, `../src/renderer/src/i18n/locales/${locale}.json`),
          'utf-8',
        ),
      )
      const help = catalog.chat.settings.effortHelpGeneric as string
      expect(typeof help).toBe('string')
      expect(help.length).toBeGreaterThan(0)
      // The effort control renders Default/Low/High/Max — never Auto.
      expect(/\bAuto\b/i.test(help)).toBe(false)
    }
  })

  it('is translated, not English copied into every catalog', () => {
    const read = (locale: string) =>
      JSON.parse(
        readFileSync(
          resolve(__dirname, `../src/renderer/src/i18n/locales/${locale}.json`),
          'utf-8',
        ),
      ).chat.settings.effortHelpGeneric as string
    const en = read('en')
    for (const locale of ['es', 'ja', 'ko', 'zh']) {
      expect(read(locale)).not.toBe(en)
    }
  })
})

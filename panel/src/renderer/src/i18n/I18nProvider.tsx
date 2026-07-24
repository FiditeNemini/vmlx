import { useCallback, useEffect, useState, type ReactNode } from 'react'
import en from './locales/en.json'
import zh from './locales/zh.json'
import ko from './locales/ko.json'
import ja from './locales/ja.json'
import es from './locales/es.json'
import type { Locale } from './catalog'
import { I18nContext } from './context'
import { translateFromCatalog } from './translate'

const locales: Record<Locale, Record<string, unknown>> = {
  en,
  zh,
  ko,
  ja,
  es,
}

// localStorage key — namespaced under the vMLX prefix to not collide with
// any other stored setting. If future code ever changes this, also update
// main/i18n.ts (which hard-codes the same name in the persistence DB row)
// and panel/tests/i18n-consistency.test.ts (which asserts the contract).
export const LOCALE_STORAGE_KEY = 'vmlx-locale'

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    try {
      const saved = localStorage.getItem(LOCALE_STORAGE_KEY)
      return saved && saved in locales ? (saved as Locale) : 'en'
    } catch {
      return 'en'
    }
  })

  // Mirror the initial locale to the main process once on mount so the
  // tray menu, system dialogs, and load-progress labels match what the
  // renderer shows — even before the user clicks the picker.
  useEffect(() => {
    try {
      const api = (window as any).api
      if (api?.i18n?.setLocale) api.i18n.setLocale(locale)
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const setLocale = useCallback((nextLocale: Locale) => {
    setLocaleState(nextLocale)
    try {
      localStorage.setItem(LOCALE_STORAGE_KEY, nextLocale)
    } catch {}
    // Push to main so tray/menu/dialogs update live alongside the React tree.
    try {
      const api = (window as any).api
      if (api?.i18n?.setLocale) api.i18n.setLocale(nextLocale)
    } catch {}
  }, [])

  // Translation lookup must never throw from inside a render function.
  const t = useCallback(
    (key: string, params?: Record<string, string | number>): string =>
      translateFromCatalog(locales[locale], en, key, params),
    [locale],
  )

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  )
}

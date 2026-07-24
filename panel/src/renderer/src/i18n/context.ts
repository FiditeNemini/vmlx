import { createContext, useContext, type Context } from 'react'
import en from './locales/en.json'
import type { Locale } from './catalog'
import { translateFromCatalog } from './translate'

export interface I18nContextType {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: string, params?: Record<string, string | number>) => string
}

const defaultValue: I18nContextType = {
  locale: 'en',
  setLocale: () => {},
  // A provider wiring failure must remain legible instead of exposing raw
  // translation keys throughout the shell.
  t: (key, params) => translateFromCatalog(en, en, key, params),
}

// React Fast Refresh can temporarily evaluate provider and consumer importers
// at different times. Keep one context object for the lifetime of this
// renderer global so those module instances cannot split onto distinct
// contexts during HMR.
const I18N_CONTEXT_GLOBAL = Symbol.for('net.vmlx.renderer.i18n-context')
const registry = globalThis as Record<PropertyKey, unknown>
const existing = registry[I18N_CONTEXT_GLOBAL] as
  | Context<I18nContextType>
  | undefined

export const I18nContext =
  existing ?? createContext<I18nContextType>(defaultValue)

registry[I18N_CONTEXT_GLOBAL] = I18nContext

export function useTranslation(): I18nContextType {
  return useContext(I18nContext)
}

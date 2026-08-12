export function getTranslation(
  catalog: Record<string, unknown> | undefined,
  path: string,
): string | undefined {
  if (!catalog || typeof path !== 'string') return undefined

  try {
    const keys = path.split('.')
    let current: unknown = catalog
    for (const key of keys) {
      if (current == null || typeof current !== 'object') return undefined
      current = (current as Record<string, unknown>)[key]
    }
    return typeof current === 'string' ? current : undefined
  } catch {
    return undefined
  }
}

export function interpolateTranslation(
  template: string,
  params?: Record<string, string | number>,
): string {
  if (!params) return template

  try {
    return template.replace(/\{(\w+)\}|\{\{(\w+)\}\}/g, (_, single, doubled) =>
      String(params[single || doubled] ?? `{${single || doubled}}`),
    )
  } catch {
    return template
  }
}

export function translateFromCatalog(
  catalog: Record<string, unknown> | undefined,
  fallbackCatalog: Record<string, unknown>,
  key: string,
  params?: Record<string, string | number>,
): string {
  try {
    if (typeof key !== 'string') return String(key ?? '')
    // `defaultValue` is honoured before falling back to the raw key. Load
    // progress arrives from the MAIN process as `{ labelKey, label }`, and the
    // call sites pass the English `label` as defaultValue precisely so a key
    // the renderer's catalog does not carry still shows words. Ignoring it put
    // a dotted key — `sessions.loadProgress.…` — in the progress bar whenever
    // main and renderer disagreed about the key set, which is exactly the
    // version-skew case the defaultValue exists for.
    const fallback =
      typeof params?.defaultValue === 'string' && params.defaultValue
        ? params.defaultValue
        : key
    const value =
      getTranslation(catalog, key) ??
      getTranslation(fallbackCatalog, key) ??
      fallback
    return interpolateTranslation(value, params)
  } catch {
    return key
  }
}

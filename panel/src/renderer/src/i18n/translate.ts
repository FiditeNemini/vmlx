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
    const value =
      getTranslation(catalog, key) ??
      getTranslation(fallbackCatalog, key) ??
      key
    return interpolateTranslation(value, params)
  } catch {
    return key
  }
}

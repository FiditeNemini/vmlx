import { readFileSync, statSync } from 'fs'
import { isAbsolute, join } from 'path'

/** Test-only fixture locations. Missing explicit inputs are errors, never skipped. */
export function loadLocalModelPaths(manifest: string | undefined, knownRows: string[]): Record<string, string> {
  if (!manifest) return {}
  const paths: unknown = JSON.parse(readFileSync(manifest, 'utf8'))
  if (!paths || typeof paths !== 'object' || Array.isArray(paths)
      || Object.keys(paths).some(name => !knownRows.includes(name))) {
    throw new Error('Local model manifest must map known high-risk row names')
  }
  for (const [name, value] of Object.entries(paths)) {
    if (typeof value !== 'string' || !isAbsolute(value)) {
      throw new Error(`${name}: local fixture path must be an absolute string`)
    }
    if (!statSync(value).isDirectory() || !statSync(join(value, 'config.json')).isFile()) {
      throw new Error(`${name}: explicit local fixture/config is missing`)
    }
  }
  return paths as Record<string, string>
}

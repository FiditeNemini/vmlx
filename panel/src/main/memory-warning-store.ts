import { createHash } from 'node:crypto'
import { realpathSync } from 'node:fs'
import { resolve } from 'node:path'
import { isMeasuredMetalPressure, measuredWiredLimitCommand, readMeasuredMetalMemory, type MetalMemoryNotice } from '../shared/metalWiredLimit'

const dismissalKey = (id: string) => `metal-memory-warning-dismissed-v1:${id}`

export function memoryWarningModelId(modelPath: string): string {
  let path = resolve(modelPath)
  try { path = realpathSync(path) } catch { /* deleted/unavailable path keeps its absolute identity */ }
  return createHash('sha256').update(path).digest('hex')
}

/** One notice per local bundle per app run; explicit opt-out survives restarts. */
export class MemoryWarningStore {
  private seen = new Set<string>()
  private notices = new Map<string, MetalMemoryNotice>()
  constructor(private settings: { getSetting(key: string): unknown; setSetting(key: string, value: string): unknown }) {}

  observe(context: { sessionId: string; modelPath: string; modelName: string; pid: number }, value: unknown, now = Date.now()): boolean {
    const measurement = readMeasuredMetalMemory(value, context.pid, now)
    if (!measurement || !isMeasuredMetalPressure(measurement)) return false
    const id = memoryWarningModelId(context.modelPath)
    if (this.seen.has(id) || this.settings.getSetting(dismissalKey(id)) === '1') return false
    this.seen.add(id)
    this.notices.set(id, { id, sessionId: context.sessionId, modelName: context.modelName, measurement, command: measuredWiredLimitCommand(measurement) })
    return true
  }

  list(): MetalMemoryNotice[] { return [...this.notices.values()] }
  get(id: string): MetalMemoryNotice | undefined { return this.notices.get(id) }
  dismiss(id: string, forModel: boolean): void {
    if (!this.notices.has(id)) return
    // Persist first: a failed save must leave the notice actionable.
    if (forModel) this.settings.setSetting(dismissalKey(id), '1')
    this.notices.delete(id)
  }
}

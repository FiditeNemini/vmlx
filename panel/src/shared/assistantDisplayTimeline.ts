export type AssistantDisplayItem =
  | { kind: 'content'; start: number; end: number }
  | { kind: 'reasoning'; segment: number; start: number; end: number }
  | { kind: 'tool'; callId: string }

export interface AssistantDisplayTimeline {
  version: 1
  items: AssistantDisplayItem[]
}

/** Record emitted display order; offsets refer to the existing text fields. */
export class AssistantDisplayTimelineRecorder {
  private items: AssistantDisplayItem[] = []
  private content = ''
  private reasoning: string[] = []
  private valid = true

  private text(kind: 'content' | 'reasoning', before: string, after: string, segment?: number) {
    if (!after.startsWith(before)) { this.valid = false; return }
    if (after.length === before.length) return
    const last = this.items[this.items.length - 1]
    if (last?.kind === kind && last.end === before.length
        && (kind === 'content' || (last.kind === 'reasoning' && last.segment === segment))) {
      last.end = after.length
    } else if (kind === 'content') {
      this.items.push({ kind, start: before.length, end: after.length })
    } else {
      this.items.push({ kind, segment: segment!, start: before.length, end: after.length })
    }
  }

  observeContent(content: string) {
    this.text('content', this.content, content)
    this.content = content
  }

  observeReasoning(segments: string[]) {
    if (segments.length < this.reasoning.length) this.valid = false
    segments.forEach((text, segment) => this.text('reasoning', this.reasoning[segment] || '', text, segment))
    this.reasoning = [...segments]
  }

  tool(callId: string | undefined) {
    if (!callId) { this.valid = false; return }
    if (!this.items.some(item => item.kind === 'tool' && item.callId === callId)) {
      this.items.push({ kind: 'tool', callId })
    }
  }

  snapshot(): AssistantDisplayTimeline | undefined {
    return this.valid ? { version: 1, items: this.items.map(item => ({ ...item })) } : undefined
  }

  /** Final display cleanup may trim whitespace; other edits have no known order. */
  finalize(content: string, reasoning: string[]): AssistantDisplayTimeline | undefined {
    if (!this.valid || this.content.trim() !== content.trim()
        || reasoning.some((text, index) => text !== (this.reasoning[index] || ''))
        || this.reasoning.some((text, index) => text !== (reasoning[index] || ''))) return undefined
    const leading = (text: string) => text.length - text.trimStart().length
    const shift = leading(content) - leading(this.content)
    const items = this.items.flatMap<AssistantDisplayItem>(item => {
      if (item.kind !== 'content') return [{ ...item }]
      const start = Math.max(0, Math.min(content.length, item.start + shift))
      const end = Math.max(start, Math.min(content.length, item.end + shift))
      return end > start ? [{ ...item, start, end }] : []
    })
    return { version: 1, items }
  }
}

/** Unknown/legacy rows do not acquire an inferred timeline. */
export function readAssistantDisplayTimeline(value: unknown): AssistantDisplayTimeline | undefined {
  if (!value || typeof value !== 'object') return undefined
  const timeline = value as AssistantDisplayTimeline
  if (timeline.version !== 1 || !Array.isArray(timeline.items)) return undefined
  if (!timeline.items.every(item => {
    if (!item || typeof item !== 'object') return false
    if (item.kind === 'tool') return typeof item.callId === 'string' && item.callId.length > 0
    if (item.kind !== 'content' && item.kind !== 'reasoning') return false
    return Number.isInteger(item.start) && Number.isInteger(item.end) && item.start >= 0 && item.end > item.start
      && (item.kind !== 'reasoning' || (Number.isInteger(item.segment) && item.segment >= 0))
  })) return undefined
  return timeline
}

export function validateAssistantDisplayTimeline(
  value: unknown, content: string, reasoning: string[],
): AssistantDisplayTimeline | undefined {
  const timeline = readAssistantDisplayTimeline(value)
  if (!timeline) return undefined
  let contentEnd = 0
  const reasoningEnds = new Map<number, number>()
  const calls = new Set<string>()
  for (const item of timeline.items) {
    if (item.kind === 'tool') {
      if (calls.has(item.callId)) return undefined
      calls.add(item.callId)
    } else if (item.kind === 'content') {
      if (item.start !== contentEnd || item.end > content.length) return undefined
      contentEnd = item.end
    } else {
      if (item.start !== (reasoningEnds.get(item.segment) || 0)
          || item.end > (reasoning[item.segment]?.length || 0)) return undefined
      reasoningEnds.set(item.segment, item.end)
    }
  }
  if (contentEnd !== content.length
      || reasoning.some((text, segment) => text.length !== (reasoningEnds.get(segment) || 0))) return undefined
  return timeline
}

import { describe, expect, it } from 'vitest'
import { historyIsStableAcrossToolAuthorization, shouldReplayHistoricalReasoning } from '../src/shared/nativeReasoningHistory'
import { toolCapabilityEpochInstruction } from '../src/main/tool-capability-epoch'
import { replayPersistedAssistantHistory } from '../src/shared/toolHistoryReplay'
import { requestsNoToolCalls, toolChoiceForCurrentTurn } from '../src/shared/toolAutoContinue'

const row = {
  content: 'Recorded result.',
  reasoningSegmentsJson: JSON.stringify(['Inspect first.', 'Report after result.']),
  toolCallsJson: JSON.stringify([{ phase: 'calling', toolName: 'read_file', toolCallId: 'call_1', iteration: 0 }]),
  toolCallsOaiJson: JSON.stringify([{ id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":"note.txt"}' } }]),
  toolResultsOaiJson: JSON.stringify([{ tool_call_id: 'call_1', content: 'Actual file text.' }]),
}

describe('native historical reasoning policy', () => {
  it.each(['qwen3.5', 'qwen4-exp', 'naive_n05_flash', 'glm5-next'])('keeps %s history stable while forbidding new tools on both wires', family => {
    const forbidden = requestsNoToolCalls('Answer from history. Do not use tools.')
    expect(forbidden).toBe(true)
    for (const responses of [false, true]) {
      const before = replayPersistedAssistantHistory(row, responses, {
        includeReasoning: shouldReplayHistoricalReasoning(family, false),
      })
      const after = replayPersistedAssistantHistory(row, responses, {
        includeReasoning: shouldReplayHistoricalReasoning(family, forbidden),
      })
      expect(after).toEqual(before)
      expect(toolChoiceForCurrentTurn(forbidden, [], responses ? 'responses' : 'chat')).toBe('none')
      if (responses) {
        expect(after.map(m => m.type)).toEqual(['reasoning', 'function_call', 'function_call_output', 'reasoning', 'output_text'])
        expect(after[2].output).toBe('Actual file text.')
      } else {
        expect(after.map(m => m.role)).toEqual(['assistant', 'tool', 'assistant'])
        expect(after[0].reasoning_content).toBe('Inspect first.')
        expect(after[1].content).toBe('Actual file text.')
        expect(after[2].reasoning_content).toBe('Report after result.')
      }
    }
  })

  // GLM-5.3 long chat (2026-10-12): tools were toggled off earlier, then a "Without tools: ..." turn dropped the leading
  // epoch note and the replayed reasoning; the whole 31.6k-token history re-prefilled (cached 0, 78 s TTFT).
  it('keeps the leading tool-epoch note independent of the current wording for history-stable families', () => {
    const prior = ['fp-tools-on']
    const forbidden = requestsNoToolCalls('Without tools: what stale count does the ledger list for amber-787-Z?')
    expect(forbidden).toBe(true)
    const note = (family: string, forbids: boolean) => toolCapabilityEpochInstruction(
      prior, 'fp-tools-off', [], [], forbids && !historyIsStableAcrossToolAuthorization(family))
    expect(note('glm5-next', forbidden)).toBeDefined()
    expect(note('glm5-next', forbidden)).toEqual(note('glm5-next', false))
    // Legacy families keep the LFM2-era policy: an explicit prohibition replaces the generic note.
    expect(note('lfm2', forbidden)).toBeUndefined()
    expect(note('lfm2', false)).toBeDefined()
  })

  it.each([undefined, 'qwen3', 'qwen3.5-moe', 'unrelated'])('retains the legacy explicit-prohibition policy for %s', family => {
    expect(shouldReplayHistoricalReasoning(family, true)).toBe(false)
    expect(shouldReplayHistoricalReasoning(family, false)).toBe(true)
  })
})

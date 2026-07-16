import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  appendReasoningDelta,
  markReasoningToolBoundary,
  reconcileReasoningSummaryDone,
  reasoningSegmentsForDisplay,
  visibleReasoningSegments,
} from '../src/shared/interleavedReasoning'

describe('interleaved reasoning segments', () => {
  it('keeps separate reasoning segments around tool calls instead of replacing the first segment', () => {
    let segments = appendReasoningDelta([], 'Plan first.')
    segments = markReasoningToolBoundary(segments)
    segments = appendReasoningDelta(segments, 'After tool, inspect result.')
    segments = markReasoningToolBoundary(segments)
    segments = appendReasoningDelta(segments, 'Final synthesis.')

    expect(visibleReasoningSegments(segments)).toEqual([
      'Plan first.',
      'After tool, inspect result.',
      'Final synthesis.',
    ])
  })

  it('does not add empty duplicate segment boundaries for repeated tool status updates', () => {
    let segments = appendReasoningDelta([], 'Need shell.')
    segments = markReasoningToolBoundary(segments)
    segments = markReasoningToolBoundary(segments)
    segments = markReasoningToolBoundary(segments)

    expect(segments).toEqual(['Need shell.', ''])
    expect(visibleReasoningSegments(segments)).toEqual(['Need shell.'])
  })

  it('replaces old reasoning segments during live interleaved streaming, then can show all after completion', () => {
    let segments = appendReasoningDelta([], 'First plan before tools.')
    segments = markReasoningToolBoundary(segments)
    segments = appendReasoningDelta(segments, 'Second plan after tool results.')

    expect(reasoningSegmentsForDisplay(segments, { liveReplace: true })).toEqual([
      'Second plan after tool results.',
    ])
    expect(reasoningSegmentsForDisplay(segments, { liveReplace: false })).toEqual([
      'First plan before tools.',
      'Second plan after tool results.',
    ])
  })

  it('marks a resumed reasoning segment as active again in the renderer', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatInterface.tsx', 'utf8')

    expect(source).toContain('setReasoningDoneMap(prev => ({ ...prev, [data.messageId]: false }))')
  })

  it('adopts a longer authoritative Responses reasoning summary', () => {
    const segments = reconcileReasoningSummaryDone(
      ['The user wants me to call the'],
      'The user wants me to call the file_info tool exactly once.\nPlan: execute it.',
    )

    expect(segments).toEqual([
      'The user wants me to call the file_info tool exactly once.\nPlan: execute it.',
    ])
  })

  it('updates only the current segment after a tool boundary', () => {
    const segments = reconcileReasoningSummaryDone(
      ['First tool plan.', '', 'Inspect'],
      'Inspect the first tool result before answering.',
    )

    expect(segments).toEqual([
      'First tool plan.',
      '',
      'Inspect the first tool result before answering.',
    ])
  })

  it('fills an empty tool boundary when only the terminal summary arrived', () => {
    expect(
      reconcileReasoningSummaryDone(
        ['First tool plan.', ''],
        'Inspect the tool result.',
      ),
    ).toEqual(['First tool plan.', 'Inspect the tool result.'])
  })

  it('rejects terminal reasoning that contains raw tool-control markup', () => {
    const streamed = ['Safe planning prefix.']

    expect(
      reconcileReasoningSummaryDone(
        streamed,
        'Safe planning prefix.<tool_call>{"name":"file_info"}</tool_call>',
      ),
    ).toBe(streamed)
    expect(
      reconcileReasoningSummaryDone(
        streamed,
        'Safe planning prefix.<function=file_info><parameter=path>x</parameter></function>',
      ),
    ).toBe(streamed)
  })

  it('does not replace a streamed segment with unrelated terminal text', () => {
    const streamed = ['Keep this streamed reasoning.']
    expect(
      reconcileReasoningSummaryDone(streamed, 'A different summary.'),
    ).toBe(streamed)
  })
})
